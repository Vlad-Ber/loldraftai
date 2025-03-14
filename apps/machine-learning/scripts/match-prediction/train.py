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
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.data import DataLoader
import argparse

from torch.optim.lr_scheduler import OneCycleLR

from utils.match_prediction.match_dataset import MatchDataset
from utils.match_prediction.model import Model
from utils import DATA_DIR
from utils.match_prediction import (
    get_best_device,
    ENCODERS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_BATCH_SIZE,
    PREPARED_DATA_DIR,
    NUMERICAL_STATS_PATH,
)
from utils.match_prediction.column_definitions import (
    CATEGORICAL_COLUMNS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType, get_enabled_tasks
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.train import (
    get_optimizer_grouped_parameters,
    set_random_seeds,
    get_num_champions,
    collate_fn,
)
import psutil
from utils.rl.champions import Champion, ChampionClass

# Enable cuDNN auto-tuner, source https://x.com/karpathy/status/1299921324333170689/photo/1
torch.backends.cudnn.benchmark = True
# should use TensorFloat-32, which is faster that "highest" precision
torch.set_float32_matmul_precision("high")


device = get_best_device()

TRACK_SUBSET_METRICS = True  # Track validation metrics by patch, ELO, and champion ID


def get_dataloader_config():
    if device.type == "cuda":
        # Original CUDA config
        num_cpus = multiprocessing.cpu_count()
        return {
            "num_workers": max(1, min(num_cpus - 1, 8)),
            "prefetch_factor": 2,
            "pin_memory": True,
        }
    elif device.type == "mps":
        # M1/M2 Mac config
        return {
            "num_workers": 1,
            "prefetch_factor": 1,
            "pin_memory": True,
        }
    else:  # CPU
        # For 4 CPU machine, reserve 1 CPU for main process
        total_cpus = multiprocessing.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use 2 workers for 4 CPU machine, leaving 2 CPUs for main process and OS
        # Adjust prefetch_factor based on available memory
        return {
            "num_workers": min(total_cpus - 2, 2),  # Use 2 workers on 4 CPU machine
            "prefetch_factor": (
                2 if available_memory_gb > 8 else 1
            ),  # Lower if memory constrained
            "pin_memory": False,  # False for CPU training
        }


# Use in DataLoader initialization
dataloader_config = get_dataloader_config()


def save_model(model: Model, timestamp: Optional[str] = None) -> str:
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


def init_model(
    config: TrainingConfig,
    num_champions: int,
    continue_training: bool,
    load_path: Optional[str] = None,
):
    # Load label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    # Create mapping of champion IDs to their class
    champ_to_class = {str(champ.id): champ.champion_class for champ in Champion}
    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }

    # Initialize queue embeddings with same base vector + small noise
    if "queueId" in label_encoders:
        queue_encoder = label_encoders["queueId"]
        queue_base_vector = (
            torch.randn(config.embed_dim) * 0.1
        )  # Base vector for all queues
        queue_embeddings = torch.zeros(len(queue_encoder.classes_), config.embed_dim)

        print("\nQueue Embedding Initialization:")
        for queue_id in queue_encoder.classes_:
            idx = queue_encoder.transform([queue_id])[0]
            noise = torch.randn(config.embed_dim) * 0.01  # Very small noise
            queue_embeddings[idx] = queue_base_vector + noise
            print(f"Queue {queue_id}: initialized with base + noise")

    # Get the champion encoder
    champion_encoder = label_encoders["champion_ids"]

    # Create a mapping of champion IDs to display names
    champ_display_names = {str(champ.id): champ.display_name for champ in Champion}

    # Initialize embeddings with class-based bias
    class_counts = {class_type: 0 for class_type in ChampionClass}
    missing_classes = []
    missing_class_names = []

    class_means = {
        ChampionClass.MAGE: torch.randn(config.embed_dim) * 0.1,
        ChampionClass.TANK: torch.randn(config.embed_dim) * 0.1,
        ChampionClass.BRUISER: torch.randn(config.embed_dim) * 0.1,
        ChampionClass.ASSASSIN: torch.randn(config.embed_dim) * 0.1,
        ChampionClass.ADC: torch.randn(config.embed_dim) * 0.1,
        ChampionClass.ENCHANTER: torch.randn(config.embed_dim) * 0.1,
    }

    # Initialize base champion embeddings
    base_embeddings = torch.zeros(num_champions, config.embed_dim)
    for raw_id in champion_encoder.classes_:
        idx = champion_encoder.transform([raw_id])[0]
        if str(raw_id) == "UNKNOWN":
            base_embeddings[idx] = torch.zeros(config.embed_dim)
            continue
        champ_class = champ_to_class.get(str(raw_id), ChampionClass.UNIQUE)
        if champ_class != ChampionClass.UNIQUE:
            mean = class_means[champ_class]
            noise = torch.randn(config.embed_dim) * 0.01
            base_embeddings[idx] = mean + noise
        class_counts[champ_class] += 1
        if champ_class == ChampionClass.UNIQUE:
            missing_classes.append(raw_id)
            display_name = champ_display_names.get(str(raw_id), f"Champion {raw_id}")
            missing_class_names.append(display_name)

    # Log initialization statistics
    print("\nChampion Embedding Initialization Statistics:")
    print(f"Total champions: {num_champions}")
    for class_type, count in class_counts.items():
        print(f"{class_type.name}: {count} champions")
    if missing_class_names:
        print(
            f"\nWarning: {len(missing_class_names)} champions without class assignment:"
        )
        for name in sorted(missing_class_names):
            print(f"  - {name}")
    print()

    if config.log_wandb:
        wandb.log(
            {
                "init_total_champions": num_champions,
                **{
                    f"init_{class_type.name}_count": count
                    for class_type, count in class_counts.items()
                },
                "init_missing_classes_count": len(missing_classes),
            }
        )

    # Initialize model
    model = Model(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=config.embed_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    # Set the initialized queue embeddings
    if "queueId" in label_encoders:
        model.embeddings["queueId"].weight.data = queue_embeddings
        if config.log_wandb:
            # Log the maximum difference between queue embeddings
            max_diff = torch.max(torch.pdist(queue_embeddings)).item()
            wandb.log(
                {
                    "init_queue_embed_max_diff": max_diff,
                    "init_queue_embed_mean": queue_embeddings.mean().item(),
                    "init_queue_embed_std": queue_embeddings.std().item(),
                }
            )

    # Patch embeddings initialization (similar across patches)
    base_patch_vector = torch.randn(config.embed_dim) * 0.1
    patch_embeddings = torch.zeros(model.num_patches, config.embed_dim)
    for i in range(model.num_patches):
        noise = torch.randn(config.embed_dim) * 0.01
        patch_embeddings[i] = base_patch_vector + noise
    model.patch_embedding.weight.data = patch_embeddings

    # Champion+patch embeddings initialization (based on class means)
    champion_patch_embeddings = torch.zeros(
        num_champions * model.num_patches, config.embed_dim
    )
    for c in range(num_champions):
        base_vector = base_embeddings[c]
        for p in range(model.num_patches):
            idx = c * model.num_patches + p
            noise = torch.randn(config.embed_dim) * 0.01
            champion_patch_embeddings[idx] = base_vector + noise
    model.champion_patch_embedding.weight.data = champion_patch_embeddings

    # Log embedding statistics
    if config.log_wandb:
        patch_embed_mean = patch_embeddings.mean().item()
        patch_embed_std = patch_embeddings.std().item()

        # Handle single patch case
        if model.num_patches > 1:
            patch_embed_max_diff = torch.max(torch.pdist(patch_embeddings)).item()
        else:
            print(
                "Warning: Training with single patch, skipping patch distance calculations"
            )
            patch_embed_max_diff = 0.0

        champ_patch_embed_mean = champion_patch_embeddings.mean().item()
        champ_patch_embed_std = champion_patch_embeddings.std().item()

        # Handle single patch case for champion embeddings
        max_diff = 0.0
        if model.num_patches > 1:
            for c in range(num_champions):
                champ_embeds = champion_patch_embeddings[
                    c * model.num_patches : (c + 1) * model.num_patches
                ]
                diff = torch.max(torch.pdist(champ_embeds)).item()
                max_diff = max(max_diff, diff)
        wandb.log(
            {
                "init_patch_embed_mean": patch_embed_mean,
                "init_patch_embed_std": patch_embed_std,
                "init_patch_embed_max_diff": patch_embed_max_diff,
                "init_champ_patch_embed_mean": champ_patch_embed_mean,
                "init_champ_patch_embed_std": champ_patch_embed_std,
                "init_champ_patch_embed_max_diff": max_diff,
            }
        )

    if continue_training and load_path and Path(load_path).exists():
        print(f"Loading model from {load_path}")
        model.load_state_dict(torch.load(load_path, weights_only=True))

    model_params = {
        "num_categories": num_categories,
        "num_champions": num_champions,
        **config.to_dict(),
    }

    with open(MODEL_CONFIG_PATH, "wb") as f:
        pickle.dump(model_params, f)

    model.to(device)
    if device == torch.device("cuda"):
        print("Compiling model")
        model = torch.compile(model, backend="eager")
        print("Model compiled")

    if config.log_wandb:
        wandb.watch(model, log_freq=1000)

    return model


def apply_label_smoothing(labels: torch.Tensor, smoothing: float = 0.2) -> torch.Tensor:
    # Apply label smoothing
    return labels * (1 - smoothing) + 0.5 * smoothing


def train_epoch(
    model: Model,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[OneCycleLR],
    criterion: Dict[str, nn.Module],
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[float, int]:
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    enabled_tasks = get_enabled_tasks(config, epoch)
    task_names = list(enabled_tasks.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in enabled_tasks.values()], device=device
    )

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        # Apply label smoothing to binary classification labels
        for task_name, task_def in TASKS.items():
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                labels[task_name] = apply_label_smoothing(labels[task_name])

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(features)
            losses = torch.stack(
                [
                    criterion[task_name](outputs[task_name], labels[task_name])
                    for task_name in task_names
                ]
            )
            task_loss = (losses * task_weights).sum()

            # Patch regularization
            patch_reg_loss = torch.tensor(0.0, device=device)
            if model.num_patches > 1:
                for i in range(model.num_patches - 1):
                    diff = (
                        model.patch_embedding.weight[i]
                        - model.patch_embedding.weight[i + 1]
                    )
                    patch_reg_loss += torch.norm(diff, p=2)
                patch_reg_loss /= model.num_patches - 1

            # Champion+patch regularization
            champ_patch_reg_loss = torch.tensor(0.0, device=device)
            if model.num_patches > 1:
                for c in range(model.num_champions):
                    for p in range(model.num_patches - 1):
                        idx1 = c * model.num_patches + p
                        idx2 = c * model.num_patches + p + 1
                        diff = (
                            model.champion_patch_embedding.weight[idx1]
                            - model.champion_patch_embedding.weight[idx2]
                        )
                        champ_patch_reg_loss += torch.norm(diff, p=2)
                champ_patch_reg_loss /= model.num_champions * (model.num_patches - 1)

            # Combine losses with regularization weights
            total_loss = (
                task_loss
                + config.patch_reg_lambda * patch_reg_loss
                + config.champ_patch_reg_lambda * champ_patch_reg_loss
            ) / config.accumulation_steps

        total_loss.backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler is not None and batch_idx < len(train_loader) - 1:
                scheduler.step()
            optimizer.zero_grad()

            if (
                (batch_idx + 1) // config.accumulation_steps
            ) % 20 == 0 and config.log_wandb:
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler
                    else optimizer.param_groups[0]["lr"]
                )
                log_training_step(
                    epoch,
                    batch_idx,
                    grad_norm,
                    losses,
                    task_names,
                    config,
                    current_lr,
                    patch_reg_loss.item(),
                    champ_patch_reg_loss.item(),
                )

        epoch_loss += total_loss.item()
        epoch_steps += 1

    return epoch_loss, epoch_steps


def log_training_step(
    epoch: int,
    batch_idx: int,
    grad_norm: float,
    losses: torch.Tensor,
    task_names: List[str],
    config: TrainingConfig,
    current_lr: float,
    patch_reg_loss: float,
    champ_patch_reg_loss: float,
) -> None:
    if config.log_wandb:
        log_data = {
            "epoch": epoch + 1,
            "batch": (batch_idx + 1) // config.accumulation_steps,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "patch_reg_loss": patch_reg_loss,
            "champ_patch_reg_loss": champ_patch_reg_loss,
        }
        log_data.update(
            {f"train_loss_{k}": v.item() for k, v in zip(task_names, losses)}
        )
        wandb.log(log_data)


def validate(
    model: Model,
    test_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[Optional[float], Dict[str, float]]:
    model.eval()
    enabled_tasks = get_enabled_tasks(config, epoch)
    metric_accumulators = {
        task_name: torch.zeros(2).cpu() for task_name in enabled_tasks.keys()
    }

    # Load both patch mapping and numerical stats for denormalization
    try:
        with open(Path(PREPARED_DATA_DIR) / "patch_mapping.pkl", "rb") as f:
            patch_data = pickle.load(f)
            reverse_patch_mapping = {v: k for k, v in patch_data["mapping"].items()}

        with open(Path(NUMERICAL_STATS_PATH), "rb") as f:
            numerical_stats = pickle.load(f)
            patch_mean = numerical_stats["means"]["numerical_patch"]
            patch_std = numerical_stats["stds"]["numerical_patch"]
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load patch mapping or stats: {e}")
        reverse_patch_mapping = None
        patch_mean = None
        patch_std = None

    # Load encoders for metric names
    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
            queue_inverse_transform = encoders["queueId"].inverse_transform
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load queue encoders: {e}")
        queue_inverse_transform = lambda x: x  # fallback to original encoded values

    # New: Add subset accumulators for win prediction
    patch_metrics = {}  # {patch_val: [loss_sum, count]}
    elo_metrics = {}  # {elo_val: [loss_sum, count]}
    queue_metrics = {}  # {queue_id: [loss_sum, count]}

    # Create a separate criterion for per-sample losses
    per_sample_criterion = nn.BCEWithLogitsLoss(reduction="none")

    num_samples = 0
    total_loss = 0.0
    total_steps = 0
    task_names = list(enabled_tasks.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in enabled_tasks.values()], device=device
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

                    # New: Track win prediction loss by subsets
                    if TRACK_SUBSET_METRICS and "win_prediction" in enabled_tasks:
                        # Calculate all sample losses at once using the per-sample criterion
                        sample_losses = per_sample_criterion(
                            outputs["win_prediction"], labels["win_prediction"]
                        ).cpu()

                        # Get all patch and elo values
                        patch_values = features["numerical_patch"].cpu()
                        elo_values = features["numerical_elo"].cpu().int()

                        # Track by patch
                        unique_patches = torch.unique(patch_values)
                        for patch_val in unique_patches:
                            mask = patch_values == patch_val
                            if patch_val.item() not in patch_metrics:
                                patch_metrics[patch_val.item()] = [0.0, 0]
                            patch_metrics[patch_val.item()][0] += (
                                sample_losses[mask].sum().item()
                            )
                            patch_metrics[patch_val.item()][1] += mask.sum().item()

                        # Track by ELO
                        unique_elos = torch.unique(elo_values)
                        for elo_val in unique_elos:
                            mask = elo_values == elo_val
                            if elo_val.item() not in elo_metrics:
                                elo_metrics[elo_val.item()] = [0.0, 0]
                            elo_metrics[elo_val.item()][0] += (
                                sample_losses[mask].sum().item()
                            )
                            elo_metrics[elo_val.item()][1] += mask.sum().item()

                        # New: Track by queue type
                        queue_values = features["queueId"].cpu()
                        unique_queues = torch.unique(queue_values)
                        for queue_val in unique_queues:
                            mask = queue_values == queue_val
                            if queue_val.item() not in queue_metrics:
                                queue_metrics[queue_val.item()] = [0.0, 0]
                            queue_metrics[queue_val.item()][0] += (
                                sample_losses[mask].sum().item()
                            )
                            queue_metrics[queue_val.item()][1] += mask.sum().item()

            batch_size = next(iter(labels.values())).size(0)
            num_samples += batch_size
            total_steps += 1

            update_metric_accumulators(
                metric_accumulators, outputs, labels, batch_size, config
            )

    metrics = calculate_final_metrics(metric_accumulators)
    avg_loss = total_loss / total_steps if config.calculate_val_loss else None

    # New: Add subset metrics to the metrics dictionary
    if TRACK_SUBSET_METRICS:
        for patch_val, (loss_sum, count) in patch_metrics.items():
            if reverse_patch_mapping is not None and patch_mean is not None:
                # First denormalize the patch value
                denormalized_patch = (patch_val * patch_std) + patch_mean
                # Round to nearest integer since patch mapping uses integers
                denormalized_patch = round(denormalized_patch)

                # Convert to original format (e.g., "13.10")
                original_patch = reverse_patch_mapping.get(denormalized_patch)
                if original_patch is not None:
                    major = int(original_patch) // 50
                    minor = int(original_patch) % 50
                    patch_key = f"{major}.{minor:02d}"
                else:
                    patch_key = f"unknown_{patch_val:.2f}"
            else:
                patch_key = f"{patch_val:.2f}"

            metrics[f"win_prediction_patch_{patch_key}"] = loss_sum / count

        for elo_val, (loss_sum, count) in elo_metrics.items():
            metrics[f"win_prediction_elo_{elo_val}"] = loss_sum / count

        # Add queue metrics using original queueId values
        for queue_val, (loss_sum, count) in queue_metrics.items():
            original_queue_id = queue_inverse_transform([queue_val])[0]
            metrics[f"win_prediction_queue_{original_queue_id}"] = loss_sum / count

    return avg_loss, metrics


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
            correct = (preds == task_label).float().sum().cpu()
            metric_accumulators[task_name][0] += correct
            metric_accumulators[task_name][1] += batch_size
        elif task_def.task_type == TaskType.REGRESSION:
            mse = nn.functional.mse_loss(task_output, task_label, reduction="sum").cpu()
            metric_accumulators[task_name][0] += mse
            metric_accumulators[task_name][1] += batch_size


def calculate_final_metrics(
    metric_accumulators: Dict[str, torch.Tensor],
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
    if config.log_wandb:
        log_data = {}
        for task_name, metric_value in val_metrics.items():
            if (
                config.calculate_val_win_prediction_only
                and task_name != "win_prediction"
            ):
                continue
            log_data[f"val_{task_name}_metric"] = metric_value
        wandb.log(log_data)


def train_model(
    run_name: str | None,
    config: TrainingConfig,
    continue_training: bool = False,
    load_path: Optional[str] = None,
):
    global model  # to be able to save the model on interrupt
    best_metric = float("inf")  # For loss minimization
    best_model_state = None

    num_champions, unknown_champion_id = get_num_champions()

    # Initialize the datasets with masking parameters and dataset fraction
    datasets = []
    for split in ["train", "test", "test_masked"]:
        masking_function = (
            config.get_masking_function() if split in ["train", "test_masked"] else None
        )
        train_or_test = "test" if split.startswith("test") else "train"

        dataset = MatchDataset(
            masking_function=masking_function,
            unknown_champion_id=unknown_champion_id,
            train_or_test=train_or_test,
            dataset_fraction=config.dataset_fraction,
        )
        datasets.append(dataset)

    train_dataset, test_dataset, test_masked_dataset = datasets

    # Initialize wandb
    if config.log_wandb:
        config_dict = config.to_dict()
        config_dict["num_samples"] = len(
            train_dataset
        )  # TODO: this could take some time, maybe we should have a file with this stat and claculate it only once?
        wandb.init(project="draftking", name=run_name, config=config_dict)

    train_loader, test_loader, test_masked_loader = (
        DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            **dataloader_config,
            collate_fn=collate_fn,
            persistent_workers=device.type == "cuda",
        )
        for dataset in [train_dataset, test_dataset, test_masked_dataset]
    )

    model = init_model(config, num_champions, continue_training, load_path)

    # Initialize loss functions for each task
    criterion = {}
    enabled_tasks = get_enabled_tasks(config, epoch=0)  # Get only enabled tasks
    for task_name, task_def in enabled_tasks.items():
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            criterion[task_name] = nn.BCEWithLogitsLoss()
        elif task_def.task_type == TaskType.REGRESSION:
            criterion[task_name] = nn.MSELoss()
        elif task_def.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            criterion[task_name] = nn.CrossEntropyLoss()

    fused = True if device.type == "cuda" else False
    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, config.weight_decay),
        lr=config.learning_rate,
        fused=fused,
    )

    # Initialize scheduler
    if config.use_one_cycle_lr:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
            anneal_strategy="cos",  # cosine annealing
        )

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        epoch_loss, epoch_steps = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler if config.use_one_cycle_lr else None,  # Pass scheduler
            criterion,
            config,
            device,
            epoch,
        )
        avg_loss = epoch_loss / epoch_steps * config.accumulation_steps

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {avg_loss:.4f}")
        if config.log_wandb:
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

        # Only run validation on specified intervals
        if (epoch + 1) % config.validation_interval == 0:
            val_loss, val_metrics = validate(
                model, test_loader, criterion, config, device, epoch
            )
            val_masked_loss, val_masked_metrics = validate(
                model, test_masked_loader, criterion, config, device, epoch
            )

            if config.calculate_val_loss:
                if config.log_wandb:
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_masked_loss": val_masked_loss,
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                            **{
                                f"val_masked_{k}": v
                                for k, v in val_masked_metrics.items()
                            },
                        }
                    )

                if val_loss < best_metric:
                    best_metric = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    torch.save(best_model_state, MODEL_PATH)
                    print(
                        f"New best model saved with validation loss: {best_metric:.4f}"
                    )

                # save with timestamp anyways
                save_model(model)

            log_validation_metrics(val_metrics, config)

        epoch_end_time = time.time()
        if config.log_wandb:
            wandb.log({"epoch_time": epoch_end_time - epoch_start_time})

    if config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train the match-prediction model")
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default=None,
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
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0)",
    )

    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Run validation every N epochs",
    )
    parser.add_argument(
        "--disable-aux-tasks",
        action="store_true",
        help="Disable auxiliary tasks and train only on win prediction",
    )
    args = parser.parse_args()

    # Initialize configuration
    config = TrainingConfig()
    if args.config:
        config.update_from_json(args.config)

    # Update config with command line arguments
    config.dataset_fraction = args.dataset_fraction
    config.validation_interval = args.validation_interval
    config.aux_tasks_enabled = not args.disable_aux_tasks

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
