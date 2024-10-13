# scripts/train.py
import multiprocessing
import signal
import os
import pickle
import datetime
import time
import cProfile
import pstats
import io
from pstats import SortKey

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.data import DataLoader
import argparse


from utils.match_dataset import MatchDataset
from utils.model import MatchOutcomeModel
from utils import (
    get_best_device,
    ENCODERS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_BATCH_SIZE,
    DATA_DIR,
)
from utils.column_definitions import (
    COLUMNS,
    CATEGORICAL_COLUMNS,
    ColumnType,
)
from utils.task_definitions import TASKS, TaskType

# Enable cuDNN auto-tuner, source https://x.com/karpathy/status/1299921324333170689/photo/1
torch.backends.cudnn.benchmark = True

CALCULATE_VAL_LOSS = True
CALCULATE_VAL_WIN_PREDICTION_ONLY = True

device = get_best_device()
print(f"Using device: {device}")

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

MASK_CHAMPIONS = 0.1

# should use TensorFloat-32, which is faster that "highest" precision
torch.set_float32_matmul_precision("high")

LOG_WANDB = True

def get_optimizer_grouped_parameters(model, weight_decay):
    # Get all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Separate parameters into decay and no-decay groups
    # dim >= 2 are the weight matrices, dim < 2 are biases
    # decaying biases and normalization layers is not needed
    # source: https://youtu.be/l8pRSuU81PU?si=f_taru0joQ5LW19e&t=8861
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # Create optimizer groups
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Print statistics
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    return optim_groups

def save_model(model, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_timestamp_path = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pth"
    torch.save(model.state_dict(), model_timestamp_path)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {model_timestamp_path} and {MODEL_PATH}")
    return model_timestamp_path


def cleanup():
    if "model" in globals():
        save_model(model)
    if LOG_WANDB and wandb.run is not None:
        wandb.finish()


def signal_handler(signum, frame):
    print("Received interrupt signal. Saving model and exiting...")
    cleanup()
    exit(0)


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def collate_fn(batch):
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


def train_model(run_name: str):
    global model  # to be able to save the model on interrupt
    # Initialize wandb
    if LOG_WANDB:
        wandb.init(project="draftking", name=run_name)

    # Determine the maximum champion ID
    # max_champion_id = get_max_champion_id() # TODO: remove this. SUPER SLOW!!!
    max_champion_id = 950  # naafiri, hardcoded until get_max_champion_id is optimized
    unknown_champion_id = max_champion_id + 1
    num_champions = unknown_champion_id + 1  # Total number of embeddings

    # Initialize the datasets with masking parameters
    train_dataset = MatchDataset(
        mask_champions=MASK_CHAMPIONS,
        unknown_champion_id=unknown_champion_id,
        train_or_test="train",
    )
    test_dataset = MatchDataset(
        mask_champions=MASK_CHAMPIONS,
        unknown_champion_id=unknown_champion_id,
        train_or_test="test",
    )

    # Initialize the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
        prefetch_factor=PREFETCH_FACTOR,  # Prefetch next batch while current batch is being processed
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
        prefetch_factor=PREFETCH_FACTOR,  # Prefetch next batch while current batch is being processed
        pin_memory=True,
    )

    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }

    embed_dim = 64
    num_heads = 8
    num_transformer_layers = 2
    # Initialize the model
    model = MatchOutcomeModel(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        dropout=0.1,
    )

    # Create a dictionary with model parameters
    model_params = {
        "num_categories": num_categories,
        "num_champions": num_champions,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_transformer_layers": num_transformer_layers,
        "dropout": 0.1,
    }

    # Save model config
    with open(MODEL_CONFIG_PATH, "wb") as f:
        pickle.dump(model_params, f)

    model.to(device)
    if device != torch.device("mps"):
        print("Compiling model")
        model = torch.compile(model)
        print("Model compiled")

    if LOG_WANDB:
        wandb.watch(model, log_freq=1000)  # increased from 100

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
    weight_decay = 0.01
    fused = True if device.type == "cuda" else False
    optimizer = optim.AdamW(get_optimizer_grouped_parameters(model, weight_decay), lr=1e-3, fused=fused)
    max_grad_norm = 1.0

    # Before the training loop, create these tensors:
    task_weights = torch.tensor(
        [task_def.weight for task_def in TASKS.values()], device=device
    )
    task_names = list(TASKS.keys())

    accumulation_steps = 1
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()  # Moved outside for accumulation

        for batch_idx, (features, labels) in enumerate(train_loader):

            # Move all features to the device
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            optimizer.zero_grad()
            # only works with CUDA, will be automatically disabled for non-CUDA devices
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features)

                # Vectorized loss calculation
                losses = torch.stack(
                    [
                        criterion[task_name](outputs[task_name], labels[task_name])
                        for task_name in task_names
                    ]
                )

                # Weighted sum of losses
                total_loss = (losses * task_weights).sum()
                # Normalize the loss to account for accumulation steps
                total_loss = total_loss / accumulation_steps

                # For logging purposes
                loss_dict = {
                    task_name: loss.item()
                    for task_name, loss in zip(task_names, losses)
                }

            total_loss.backward()

            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_steps += 1

            # Perform optimization step every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                if ((batch_idx + 1) // accumulation_steps) % 20 == 0 and LOG_WANDB:
                    log_data = {
                        "epoch": epoch + 1,
                        "batch": (batch_idx + 1) // accumulation_steps,
                        "grad_norm": grad_norm,
                    }
                    log_data.update(
                        {f"train_loss_{k}": v for k, v in loss_dict.items()}
                    )
                    wandb.log(log_data)

        # Perform any remaining optimization step
        if (batch_idx + 1) % accumulation_steps != 0:
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = (
            epoch_loss / epoch_steps * accumulation_steps
        )  # multiplied to have an easier comparison with different number of accumulation steps

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        if LOG_WANDB:
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

        # Evaluation
        model.eval()
        metric_accumulators = {
            task_name: torch.zeros(2).to(device) for task_name in TASKS.keys()
        }
        num_samples = 0
        total_loss = 0.0
        total_steps = 0
        with torch.no_grad():
            for features, labels in test_loader:
                total_steps += 1
                # Move all features to the device
                features = {
                    k: v.to(device, non_blocking=True) for k, v in features.items()
                }
                labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(features)
                    # Vectorized loss calculation
                    if CALCULATE_VAL_LOSS:
                        losses = torch.stack(
                            [
                                criterion[task_name](
                                    outputs[task_name], labels[task_name]
                                )
                                for task_name in task_names
                            ]
                        )
                        # Weighted sum of losses
                        weighted_losses = losses * task_weights
                        total_loss += weighted_losses.sum()

                batch_size = next(iter(labels.values())).size(0)
                num_samples += batch_size

                for task_name, task_def in TASKS.items():
                    if (
                        CALCULATE_VAL_WIN_PREDICTION_ONLY
                        and task_name != "win_prediction"
                    ):
                        continue
                    task_output = outputs[task_name]
                    task_label = labels[task_name]

                    if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                        # Apply sigmoid to get probabilities(sigmoid was removed from model, because autocast requires BCEWithLogitsLoss)
                        probs = torch.sigmoid(task_output)
                        preds = (probs >= 0.5).float()
                        correct = (preds == task_label).float().sum()
                        metric_accumulators[task_name][0] += correct
                        metric_accumulators[task_name][1] += batch_size
                    elif task_def.task_type == TaskType.REGRESSION:
                        mse = nn.functional.mse_loss(
                            task_output, task_label, reduction="sum"
                        )
                        metric_accumulators[task_name][0] += mse
                        metric_accumulators[task_name][1] += batch_size

        if CALCULATE_VAL_LOSS:
            avg_loss = total_loss / total_steps
            if LOG_WANDB:
                wandb.log({"avg_val_loss": avg_loss})

        # Calculate final metrics
        metrics = {}
        for task_name, accumulator in metric_accumulators.items():
            if TASKS[task_name].task_type == TaskType.BINARY_CLASSIFICATION:
                metrics[task_name] = (accumulator[0] / accumulator[1]).item()
            elif TASKS[task_name].task_type == TaskType.REGRESSION:
                metrics[task_name] = (accumulator[0] / accumulator[1]).item()
        # Log evaluation metrics
        for task_name, metric_value in metrics.items():
            if CALCULATE_VAL_WIN_PREDICTION_ONLY and task_name != "win_prediction":
                continue
            if LOG_WANDB:
                wandb.log({f"val_{task_name}_metric": metric_value})

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch time: {epoch_time}")
        if LOG_WANDB:
            wandb.log({"epoch_time": epoch_time})

    if LOG_WANDB:
        wandb.finish()
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


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
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Set random seeds for reproducibility
    set_random_seeds()
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        train_model(args.run_name)
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
