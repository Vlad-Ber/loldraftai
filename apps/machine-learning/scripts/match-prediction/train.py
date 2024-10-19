# scripts/train.py
import multiprocessing
import signal
import os
import copy
import pickle
import datetime
import time
import cProfile
import pstats
import io
from typing import Any, Dict, List, Optional, Tuple
from pstats import SortKey

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.data import DataLoader
import argparse


from utils.match_prediction.match_dataset import MatchDataset
from utils.match_prediction.model import MatchOutcomeModel
from utils import DATA_DIR
from utils.match_prediction import (
    get_best_device,
    ENCODERS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_BATCH_SIZE,
)
from utils.match_prediction.column_definitions import (
    COLUMNS,
    CATEGORICAL_COLUMNS,
    ColumnType,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.train import (
    get_optimizer_grouped_parameters,
    set_random_seeds,
    get_num_champions,
)

# Enable cuDNN auto-tuner, source https://x.com/karpathy/status/1299921324333170689/photo/1
torch.backends.cudnn.benchmark = True
# should use TensorFloat-32, which is faster that "highest" precision
torch.set_float32_matmul_precision("high")


device = get_best_device()

if device.type == "mps":
    DATALOADER_WORKERS = (
        1  # Fastest with 1 or 2 might be because of mps performance cores
    )
    PREFETCH_FACTOR = 1
else:
    # cuda/runpod config
    # programmaticaly determined optimal value
    num_cpus = multiprocessing.cpu_count()
    DATALOADER_WORKERS = max(1, min(num_cpus - 1, 8))  # Use at most 8 workers
    PREFETCH_FACTOR = 2


def save_model(model: MatchOutcomeModel, timestamp: Optional[str] = None) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_timestamp_path = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pth"
    torch.save(model.state_dict(), model_timestamp_path)
    print(f"Model saved to {model_timestamp_path}")
    return model_timestamp_path


def cleanup() -> None:
    if "model" in globals():
        save_model(model)
    if wandb.run is not None:
        wandb.finish()


def signal_handler(signum: int, frame: Any) -> None:
    print("Received interrupt signal. Saving model and exiting...")
    cleanup()
    exit(0)


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    collated = {col: [] for col in COLUMNS}
    collated_labels = {task: [] for task in TASKS}

    for item in batch:
        for col, col_def in COLUMNS.items():
            collated[col].append(item[col])
        for task in TASKS:
            collated_labels[task].append(item[task])

    for col, col_def in COLUMNS.items():
        if col_def.column_type == ColumnType.LIST:
            collated[col] = torch.stack(collated[col])
        elif col_def.column_type == ColumnType.CATEGORICAL:
            collated[col] = torch.tensor(collated[col], dtype=torch.long)
        elif col_def.column_type == ColumnType.NUMERICAL:
            collated[col] = torch.tensor(collated[col], dtype=torch.float)

    for task_name, task_def in TASKS.items():
        dtype = (
            torch.float
            if task_def.task_type != TaskType.MULTICLASS_CLASSIFICATION
            else torch.long
        )
        collated_labels[task_name] = torch.tensor(
            collated_labels[task_name], dtype=dtype
        )

    return collated, collated_labels


def init_model(
    config: TrainingConfig,
    num_champions: int,
    continue_training: bool,
    load_path: Optional[str] = None,
):
    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    # Initialize the model
    model = MatchOutcomeModel(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_transformer_layers=config.num_transformer_layers,
        dropout=config.dropout,
    )

    if continue_training:
        load_path = load_path or MODEL_PATH
        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            model.load_state_dict(torch.load(load_path, weights_only=True))
        else:
            print(f"No saved model found at {load_path}. Starting from scratch.")

    # Create a dictionary with model parameters
    model_params = {
        "num_categories": num_categories,
        "num_champions": num_champions,
        **config.to_dict(),
    }

    # Save model config
    with open(MODEL_CONFIG_PATH, "wb") as f:
        pickle.dump(model_params, f)

    model.to(device)
    if device != torch.device("mps"):
        print("Compiling model")
        model = torch.compile(model)
        print("Model compiled")

    if config.log_wandb:
        wandb.watch(model, log_freq=1000)

    return model


def train_epoch(
    model: MatchOutcomeModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[float, int]:
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    task_names = list(TASKS.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in TASKS.values()], device=device
    )

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(features)
            losses = torch.stack(
                [
                    criterion[task_name](outputs[task_name], labels[task_name])
                    for task_name in task_names
                ]
            )
            total_loss = (losses * task_weights).sum() / config.accumulation_steps

        total_loss.backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if (
                (batch_idx + 1) // config.accumulation_steps
            ) % 20 == 0 and config.log_wandb:
                log_training_step(
                    epoch, batch_idx, grad_norm, losses, task_names, config
                )

        epoch_loss += total_loss.item()
        epoch_steps += 1

    return epoch_loss, epoch_steps


def validate(
    model: MatchOutcomeModel,
    test_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[Optional[float], Dict[str, float]]:
    model.eval()
    metric_accumulators = {
        task_name: torch.zeros(2).to(device) for task_name in TASKS.keys()
    }
    num_samples = 0
    total_loss = 0.0
    total_steps = 0
    task_names = list(TASKS.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in TASKS.values()], device=device
    )

    with torch.no_grad():
        for features, labels in test_loader:
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features)
                if config.calculate_val_loss:
                    losses = torch.stack(
                        [
                            criterion[task_name](outputs[task_name], labels[task_name])
                            for task_name in task_names
                        ]
                    )
                    total_loss += (losses * task_weights).sum()

            batch_size = next(iter(labels.values())).size(0)
            num_samples += batch_size
            total_steps += 1

            update_metric_accumulators(
                metric_accumulators, outputs, labels, batch_size, config
            )

    metrics = calculate_final_metrics(metric_accumulators)
    avg_loss = total_loss / total_steps if config.calculate_val_loss else None

    return avg_loss, metrics


def log_training_step(
    epoch: int,
    batch_idx: int,
    grad_norm: float,
    losses: torch.Tensor,
    task_names: List[str],
    config: TrainingConfig,
) -> None:
    if config.log_wandb:
        log_data = {
            "epoch": epoch + 1,
            "batch": (batch_idx + 1) // config.accumulation_steps,
            "grad_norm": grad_norm,
        }
        log_data.update(
            {f"train_loss_{k}": v.item() for k, v in zip(task_names, losses)}
        )
        wandb.log(log_data)


def update_metric_accumulators(
    metric_accumulators: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    batch_size: int,
    config: TrainingConfig,
) -> None:
    for task_name, task_def in TASKS.items():
        if config.calculate_val_win_prediction_only and task_name != "win_prediction":
            continue
        task_output = outputs[task_name]
        task_label = labels[task_name]

        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            probs = torch.sigmoid(task_output)
            preds = (probs >= 0.5).float()
            correct = (preds == task_label).float().sum()
            metric_accumulators[task_name][0] += correct
            metric_accumulators[task_name][1] += batch_size
        elif task_def.task_type == TaskType.REGRESSION:
            mse = nn.functional.mse_loss(task_output, task_label, reduction="sum")
            metric_accumulators[task_name][0] += mse
            metric_accumulators[task_name][1] += batch_size


def calculate_final_metrics(
    metric_accumulators: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    # Calculate final metrics
    metrics = {}
    for task_name, accumulator in metric_accumulators.items():
        if TASKS[task_name].task_type == TaskType.BINARY_CLASSIFICATION:
            metrics[task_name] = (accumulator[0] / accumulator[1]).item()
        elif TASKS[task_name].task_type == TaskType.REGRESSION:
            metrics[task_name] = (accumulator[0] / accumulator[1]).item()
    return metrics


def log_validation_metrics(
    val_metrics: Dict[str, float], config: TrainingConfig
) -> None:
    for task_name, metric_value in val_metrics.items():
        if config.calculate_val_win_prediction_only and task_name != "win_prediction":
            continue
        if config.log_wandb:
            wandb.log({f"val_{task_name}_metric": metric_value})


def train_model(
    run_name: str,
    config: TrainingConfig,
    continue_training: bool = False,
    load_path: Optional[str] = None,
):
    global model  # to be able to save the model on interrupt
    best_metric = float("inf")  # For loss minimization
    best_model_state = None
    # Initialize wandb
    if config.log_wandb:
        wandb.init(project="draftking", name=run_name, config=config.to_dict())

    num_champions, unknown_champion_id = get_num_champions()

    # Initialize the datasets with masking parameters
    train_dataset, test_dataset = (
        MatchDataset(
            mask_champions=config.mask_champions,
            unknown_champion_id=unknown_champion_id,
            train_or_test=split,
        )
        for split in ["train", "test"]
    )

    train_loader, test_loader = (
        DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=DATALOADER_WORKERS,
            collate_fn=collate_fn,
            prefetch_factor=PREFETCH_FACTOR,
            pin_memory=True,
        )
        for dataset in [train_dataset, test_dataset]
    )

    model = init_model(config, num_champions, continue_training, load_path)

    # Initialize loss functions for each task
    criterion = {}
    for task_name, task_def in TASKS.items():
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            criterion[task_name] = nn.BCEWithLogitsLoss()
        elif task_def.task_type == TaskType.REGRESSION:
            criterion[task_name] = nn.MSELoss()
        elif task_def.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            criterion[task_name] = nn.CrossEntropyLoss()

    # weight decay didn't change much when training for a short time at 0.001, but for longer trianing runs, 0.01 might be better
    fused = True if device.type == "cuda" else False
    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, config.weight_decay),
        lr=config.learning_rate,
        fused=fused,
    )

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        epoch_loss, epoch_steps = train_epoch(
            model, train_loader, optimizer, criterion, config, device, epoch
        )
        avg_loss = epoch_loss / epoch_steps * config.accumulation_steps

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {avg_loss:.4f}")
        if config.log_wandb:
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

        val_loss, val_metrics = validate(model, test_loader, criterion, config, device)

        if config.calculate_val_loss:
            if config.log_wandb:
                wandb.log({"avg_val_loss": val_loss})

            if val_loss < best_metric:
                best_metric = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, MODEL_PATH)
                print(f"New best model saved with validation loss: {best_metric:.4f}")

        log_validation_metrics(val_metrics, config)

        epoch_end_time = time.time()
        if config.log_wandb:
            wandb.log({"epoch_time": epoch_end_time - epoch_start_time})

    if config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train the MatchOutcomeTransformer model"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default="initial-setup",
        help="Name for the Wandb run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        default=False,
        help="Continue training from the last saved model",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load the model from (default: MODEL_PATH)",
    )
    args = parser.parse_args()

    # Initialize configuration
    config = TrainingConfig()
    if args.config:
        config.update_from_json(args.config)

    print("Training configuration:")
    print(config)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Set random seeds for reproducibility
    set_random_seeds()
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        train_model(
            args.run_name,
            config,
            continue_training=args.continue_training,
            load_path=args.load_path,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        cleanup()
        raise

    if args.profile:
        profiler.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)

        # Optionally, save profiling results to a file
        ps.dump_stats(DATA_DIR + "train_profile.prof")
        print("Profiling data saved to train_profile.prof")
