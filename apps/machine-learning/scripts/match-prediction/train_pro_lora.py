#!/usr/bin/env python3

import os
import argparse
import torch
import wandb
import pickle
import pandas as pd
import numpy as np
import json
import ast
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from utils.match_prediction import (
    get_best_device,
    ENCODERS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    RAW_PRO_GAMES_DIR,
    NUMERICAL_STATS_PATH,
    PREPARED_DATA_DIR,
    PATCH_MAPPING_PATH,
)
from utils.match_prediction.model import Model
from utils.match_prediction.train_utils import (
    get_optimizer_grouped_parameters,
    set_random_seeds,
    get_num_champions,
)
from utils.match_prediction.column_definitions import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    POSITIONS,
)


class FineTuningConfig:
    """Configuration class for fine-tuning"""

    def __init__(self):
        # Fine-tuning hyperparameters - edit these directly instead of using command line flags
        self.num_epochs = 60
        self.learning_rate = 5e-6  # Lower learning rate for fine-tuning
        self.weight_decay = 0.05  # Much stronger regularization
        self.dropout = 0.3  # Higher dropout to prevent overfitting
        self.batch_size = 16
        self.val_split = 0.2
        self.hidden_dims = [1536, 768, 384, 192]
        self.embed_dim = 256
        self.max_grad_norm = 1.0
        self.log_wandb = True
        self.log_batch_interval = 5  # How often to log batch progress
        self.save_checkpoints = False

        # New unfreezing parameters
        # NOTE: it didn't change much, but it's best practice so we can maybe keep it for now
        self.progressive_unfreezing = True  # Enable progressive unfreezing
        self.epochs_per_unfreeze = 10  # Number of epochs before unfreezing next layer
        self.initial_frozen_layers = (
            6  # Number of layers to start frozen (2 linear + 2 batchnorm)
        )
        self.unfreeze_embeddings = False  # Whether to eventually unfreeze embeddings

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def to_dict(self):
        return {key: value for key, value in vars(self).items()}


class ProMatchDataset(Dataset):
    """Dataset for professional matches fine-tuning"""

    def __init__(
        self,
        pro_games_df: pd.DataFrame,
        unknown_champion_id: int,
        label_encoders: Dict[str, Any],
        patch_mapping: Dict[float, int],
        train_or_test: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.unknown_champion_id = unknown_champion_id
        self.label_encoders = label_encoders
        self.patch_mapping = patch_mapping

        # Split data into train/test
        np.random.seed(seed)
        indices = np.random.permutation(len(pro_games_df))
        split_idx = int(len(indices) * (1 - val_split))

        if train_or_test == "train":
            self.df = pro_games_df.iloc[indices[:split_idx]].reset_index(drop=True)
        else:
            self.df = pro_games_df.iloc[indices[split_idx:]].reset_index(drop=True)

        print(f"Created {train_or_test} dataset with {len(self.df)} pro games")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        row = self.df.iloc[idx]

        # Extract features
        features = {}

        # Process champion IDs
        try:
            champion_ids = self._get_champion_ids(row)
            encoded_champs = self.label_encoders["champion_ids"].transform(champion_ids)
            features["champion_ids"] = torch.tensor(encoded_champs, dtype=torch.long)
        except Exception as e:
            print(f"ERROR processing champion IDs for row {idx}: {e}")
            print(f"Row data: {row}")
            raise

        # Process patch
        major = row["gameVersionMajorPatch"]
        minor = row["gameVersionMinorPatch"]
        raw_patch = major * 50 + minor
        numerical_patch = self.patch_mapping.get(float(raw_patch), 0)
        features["numerical_patch"] = torch.tensor(numerical_patch, dtype=torch.long)

        # Process queue ID - always use queue 420 (ranked solo/duo) for pro games
        queue_420_idx = self.label_encoders["queueId"].transform([420])[0]
        features["queueId"] = torch.tensor(queue_420_idx, dtype=torch.long)

        # Add numerical_elo = 0 for pro games (highest skill level)
        features["numerical_elo"] = torch.tensor(0.0, dtype=torch.float32)

        # Extract labels
        labels = {
            "win_prediction": torch.tensor(row["team_100_win"], dtype=torch.float32)
        }

        return features, labels

    def _get_champion_ids(self, row) -> List[int]:
        """Extract champion IDs from a row of the DataFrame"""
        # First check if champion_ids exists and has data
        if "champion_ids" in row:
            champ_ids = row["champion_ids"]

            # If already a list, return it
            if isinstance(champ_ids, list):
                return champ_ids

            # If it's a string, try parsing it
            elif isinstance(champ_ids, str):
                try:
                    # Try to parse JSON
                    return json.loads(champ_ids.replace("'", '"'))
                except:
                    # If it looks like a list in string form but isn't valid JSON
                    if champ_ids.startswith("[") and champ_ids.endswith("]"):
                        try:
                            # Try using ast.literal_eval which is safer than eval
                            return ast.literal_eval(champ_ids)
                        except:
                            pass

            # Additional handling for pandas Series or other types
            else:
                try:
                    # Try converting to a Python list
                    return list(champ_ids)
                except:
                    pass

        # Try individual columns as fallback
        champion_ids = []
        for team in [100, 200]:
            for position in POSITIONS:
                col_name = f"team_{team}_{position}_championId"
                if col_name in row:
                    champion_ids.append(row[col_name])

        if len(champion_ids) == 10:
            return champion_ids

        # Last resort - try to extract from the repr string
        try:
            import re

            champ_str = str(row["champion_ids"])
            matches = re.findall(r"\d+", champ_str)
            if len(matches) == 10:
                return [int(x) for x in matches]
        except:
            pass

        raise ValueError(f"Could not extract champion IDs from row: {row}")


def filter_pro_games(pro_games_df, patch_mapping):
    """
    Filter professional games for compatibility with model training data

    Args:
        pro_games_df: DataFrame with professional games
        patch_mapping: Patch mapping dictionary from the original model

    Returns:
        Filtered DataFrame, a list of patches, and counts of filtered games
    """
    original_count = len(pro_games_df)

    # Convert patch mapping keys to version strings for better readability
    patches = sorted(
        f"{int(float(patch)) // 50}.{int(float(patch)) % 50:02d}"
        for patch in patch_mapping.keys()
    )

    # Filter out games with no/invalid champion IDs
    valid_champs_mask = pro_games_df.apply(has_valid_champion_ids, axis=1)
    missing_champs_count = (~valid_champs_mask).sum()

    if missing_champs_count > 0:
        print(
            f"Filtering out {missing_champs_count} games with missing or invalid champion IDs"
        )
        pro_games_df = pro_games_df[valid_champs_mask].reset_index(drop=True)

    # Filter pro games to include only compatible patches
    patch_filtered_count = 0
    compatible_games = []

    for _, row in pro_games_df.iterrows():
        patch_str = f"{int(row['gameVersionMajorPatch'])}.{int(row['gameVersionMinorPatch']):02d}"
        if patch_str in patches:
            compatible_games.append(True)
        else:
            compatible_games.append(False)
            patch_filtered_count += 1

    pro_games_df = pro_games_df[compatible_games].reset_index(drop=True)

    return (
        pro_games_df,
        patches,
        {
            "original_count": original_count,
            "missing_champion_ids": missing_champs_count,
            "incompatible_patches": patch_filtered_count,
            "remaining_count": len(pro_games_df),
        },
    )


def has_valid_champion_ids(row):
    """Check if valid champion IDs can be extracted from a row"""
    try:
        # First check if champion_ids exists and has data
        if "champion_ids" in row:
            champ_ids = row["champion_ids"]

            # If already a list, check if it has 10 elements
            if isinstance(champ_ids, list):
                return len(champ_ids) == 10

            # If it's a string, try parsing it
            elif isinstance(champ_ids, str):
                try:
                    # Try to parse JSON
                    parsed = json.loads(champ_ids.replace("'", '"'))
                    return isinstance(parsed, list) and len(parsed) == 10
                except:
                    # If it looks like a list in string form but isn't valid JSON
                    if champ_ids.startswith("[") and champ_ids.endswith("]"):
                        try:
                            # Try using ast.literal_eval which is safer than eval
                            parsed = ast.literal_eval(champ_ids)
                            return isinstance(parsed, list) and len(parsed) == 10
                        except:
                            pass

            # Additional handling for pandas Series or other types
            else:
                try:
                    # Try converting to a Python list
                    parsed = list(champ_ids)
                    return len(parsed) == 10
                except:
                    pass

        # Try individual columns as fallback
        champion_ids = []
        for team in [100, 200]:
            for position in POSITIONS:
                col_name = f"team_{team}_{position}_championId"
                if col_name in row:
                    champion_ids.append(row[col_name])

        return len(champion_ids) == 10

    except:
        return False


def pro_collate_fn(batch):
    """
    Custom collate function for the ProMatchDataset
    """
    features_batch = {}
    labels_batch = {}

    # Process each item in the batch
    for features, labels in batch:
        # Collect features
        for k, v in features.items():
            if k not in features_batch:
                features_batch[k] = []
            features_batch[k].append(v)

        # Collect labels
        for k, v in labels.items():
            if k not in labels_batch:
                labels_batch[k] = []
            labels_batch[k].append(v)

    # Convert lists to tensors
    features_batch = {
        k: torch.stack(v) if v[0].dim() > 0 else torch.tensor(v)
        for k, v in features_batch.items()
    }
    labels_batch = {
        k: torch.stack(v) if v[0].dim() > 0 else torch.tensor(v)
        for k, v in labels_batch.items()
    }

    return features_batch, labels_batch


def create_dataloaders(
    pro_games_df,
    unknown_champion_id,
    label_encoders,
    numerical_stats,
    patch_mapping,
    config,
):
    """Create train and validation dataloaders for fine-tuning"""
    # Create datasets
    train_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        unknown_champion_id=unknown_champion_id,
        label_encoders=label_encoders,
        patch_mapping=patch_mapping,
        train_or_test="train",
        val_split=config.val_split,
    )

    val_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        unknown_champion_id=unknown_champion_id,
        label_encoders=label_encoders,
        patch_mapping=patch_mapping,
        train_or_test="test",
        val_split=config.val_split,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pro_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pro_collate_fn,
    )

    return train_loader, val_loader


# Custom LoRA implementation without relying on PEFT library
class LoRALinear(nn.Module):
    """Custom LoRA implementation for linear layers"""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
    ):
        """
        Initialize LoRA layer by adding low-rank adapters to a base linear layer

        Args:
            base_layer: The original linear layer
            rank: Rank of the low-rank matrices (r)
            alpha: Scaling factor for the LoRA output
            dropout: Dropout probability for the LoRA layers
        """
        super().__init__()

        # Store original layer properties
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Keep the original weights frozen
        self.weight = nn.Parameter(base_layer.weight.detach(), requires_grad=False)

        # Handle bias
        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach(), requires_grad=False)
        else:
            self.bias = None

        # LoRA components
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
        self.lora_dropout = nn.Dropout(p=dropout)
        self.scaling = alpha / rank

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Make sure LoRA parameters are registered as module parameters
        self.to(base_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make sure everything is on the same device
        device = x.device

        # Original layer output
        orig_output = F.linear(x, self.weight, self.bias)

        # LoRA path - ensure all tensors are on the same device
        lora_x = self.lora_dropout(x)
        lora_output = (
            (lora_x @ self.lora_A.t().to(device))
            @ self.lora_B.t().to(device)
            * self.scaling
        )

        # Combine outputs
        return orig_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights with the original weights"""
        device = self.weight.device
        merged_layer = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )

        # Merge weights
        with torch.no_grad():
            # Make sure all tensors are on the same device for multiplication
            merged_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
            merged_layer.weight.copy_(merged_weight)

            if self.bias is not None:
                merged_layer.bias.copy_(self.bias)

        # Move merged layer to the same device as original weights
        merged_layer.to(device)

        return merged_layer


class LoRAConfig:
    """Configuration class for LoRA fine-tuning"""

    def __init__(self):
        # Basic fine-tuning hyperparameters
        self.num_epochs = 30  # Reduced from 60
        self.learning_rate = 1e-4  # Reduced from 5e-4
        self.weight_decay = 0.1  # Increased from 0.01
        self.batch_size = 16
        self.val_split = 0.2
        self.log_wandb = True
        self.log_batch_interval = 5
        self.save_checkpoints = False
        self.max_grad_norm = 0.5  # Reduced from 1.0

        # LoRA-specific hyperparameters
        self.lora_r = 4  # Reduced from 8
        self.lora_alpha = 8  # Reduced from 16
        self.lora_dropout = 0.2  # Increased from 0.1

        # Early stopping parameters
        self.early_stopping = True
        self.patience = 5
        self.min_delta = 0.001

        # Target modules as list of tuples
        self.target_modules = [
            ("mlp", 0),  # First linear layer in MLP
            ("mlp", 4),  # Second linear layer
            ("mlp", 8),  # Third linear layer
            ("mlp", 12),  # Fourth linear layer
            ("output_layers", "win_prediction"),  # Output layer for win prediction
        ]

        # Whether to merge LoRA weights into base model on save
        self.merge_weights = True

        # Device to use (auto-detected)
        self.device = "auto"  # Will be set to actual device later

        # Data augmentation
        self.use_data_augmentation = True
        self.augmentation_noise_std = 0.01

        # Mixup augmentation
        self.use_mixup = True
        self.mixup_alpha = 0.2

        # Label smoothing
        self.label_smoothing = 0.1

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def to_dict(self):
        return {key: value for key, value in vars(self).items()}


def apply_lora_to_model(model: Model, config: LoRAConfig) -> Model:
    """
    Apply LoRA adapters to the model's linear layers

    Args:
        model: The base model
        config: LoRA configuration

    Returns:
        Model with LoRA adapters applied to specified layers
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # Apply LoRA to MLP layers
    for target_type, target_idx in config.target_modules:
        if target_type == "mlp" and isinstance(target_idx, int):
            # Get the layer
            original_layer = model.mlp[target_idx]

            # Check if it's a linear layer
            if isinstance(original_layer, nn.Linear):
                # Replace with LoRA version
                lora_layer = LoRALinear(
                    original_layer,
                    rank=config.lora_r,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                )
                model.mlp[target_idx] = lora_layer
                print(
                    f"Applied LoRA to mlp[{target_idx}]: {original_layer.__class__.__name__}"
                )

        elif target_type == "output_layers" and isinstance(target_idx, str):
            # Get the output layer
            if target_idx in model.output_layers:
                original_layer = model.output_layers[target_idx]

                # Check if it's a linear layer
                if isinstance(original_layer, nn.Linear):
                    # Replace with LoRA version
                    lora_layer = LoRALinear(
                        original_layer,
                        rank=config.lora_r,
                        alpha=config.lora_alpha,
                        dropout=config.lora_dropout,
                    )
                    model.output_layers[target_idx] = lora_layer
                    print(
                        f"Applied LoRA to output_layers['{target_idx}']: {original_layer.__class__.__name__}"
                    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})"
    )

    return model


def merge_lora_weights(model: Model) -> Model:
    """
    Merge LoRA weights back into the base model

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with merged weights
    """
    # Create a new model instance to hold the merged weights
    merged_model = copy.deepcopy(model)

    # Merge weights in MLP
    for i, layer in enumerate(model.mlp):
        if isinstance(layer, LoRALinear):
            merged_model.mlp[i] = layer.merge_weights()

    # Merge weights in output layers
    for name, layer in model.output_layers.items():
        if isinstance(layer, LoRALinear):
            merged_model.output_layers[name] = layer.merge_weights()

    return merged_model


def fine_tune_model_lora(
    pretrained_model_path: str,
    pro_games_df: pd.DataFrame,
    config: LoRAConfig,
    output_model_path: str,
    run_name: Optional[str] = None,
) -> float:
    """Fine-tune a pre-trained model on professional game data using LoRA"""
    # Get device - Mac will use MPS if available, otherwise CPU
    if config.device == "auto":
        device = get_best_device()
    else:
        device = torch.device(config.device)

    print(f"Using device: {device}")
    config.device = str(device)  # Update config with actual device

    # Load resources
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    with open(MODEL_CONFIG_PATH, "rb") as f:
        model_config = pickle.load(f)

    # Load numerical stats for normalization
    with open(NUMERICAL_STATS_PATH, "rb") as f:
        numerical_stats = pickle.load(f)

    # Load patch mapping
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]

    # Get number of champions
    num_champions, unknown_champion_id = get_num_champions()

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        pro_games_df,
        unknown_champion_id,
        label_encoders,
        numerical_stats,
        patch_mapping,
        config,
    )

    # Initialize the base model
    model = Model(
        num_categories=model_config["num_categories"],
        num_champions=model_config["num_champions"],
        embed_dim=model_config.get(
            "embed_dim", 256
        ),  # Use original embed_dim or default
        hidden_dims=model_config.get(
            "hidden_dims", [1536, 768, 384, 192]
        ),  # Use original hidden_dims or default
        dropout=0.1,  # Lower dropout for LoRA
    )

    # Load pre-trained weights
    print(f"Loading pre-trained model from {pretrained_model_path}")
    state_dict = torch.load(pretrained_model_path, map_location=device)

    # Remove '_orig_mod.' prefix from state dict keys if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        fixed_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(fixed_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Important: Move model to device BEFORE applying LoRA
    model.to(device)

    # Apply LoRA adapters
    model = apply_lora_to_model(model, config)

    # Initialize wandb
    if config.log_wandb:
        wandb.init(
            project="draftking-pro-lora",
            name=run_name or f"lora-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                **config.to_dict(),
                "pretrained_model": pretrained_model_path,
                "pro_games_count": len(pro_games_df),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "total_params": sum(p.numel() for p in model.parameters()),
            },
        )
        wandb.watch(model, log_freq=100)

    # Define loss function for win prediction
    criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer - only train parameters that require gradients
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to device
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs["win_prediction"], labels["win_prediction"])
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
            )

            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # Calculate accuracy
            probs = torch.sigmoid(outputs["win_prediction"])
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels["win_prediction"]).sum().item()
            train_total += labels["win_prediction"].size(0)

            # Log batch progress
            if batch_idx % config.log_batch_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config.num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / train_steps
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                # Move data to device
                features = {k: v.to(device) for k, v in features.items()}
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(features)
                loss = criterion(outputs["win_prediction"], labels["win_prediction"])

                val_loss += loss.item()
                val_steps += 1

                # Calculate accuracy
                probs = torch.sigmoid(outputs["win_prediction"])
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels["win_prediction"]).sum().item()
                val_total += labels["win_prediction"].size(0)

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float("inf")
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model with validation loss: {best_val_loss:.4f}")

        epoch_time = time.time() - epoch_start

        # Console output
        print(
            f"Epoch {epoch+1}/{config.num_epochs} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}\n"
            f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )

        # Log metrics to wandb
        if config.log_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch_time": epoch_time,
                }
            )

    # Save the best model
    if best_model_state is not None:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

        # Load best state
        model.load_state_dict(best_model_state)

        if config.merge_weights:
            # Merge LoRA weights with base model
            merged_model = merge_lora_weights(model)
            torch.save(merged_model.state_dict(), output_model_path)
            print(f"Merged LoRA model saved to {output_model_path}")
        else:
            # Save model with LoRA adapters (separate implementation)
            lora_path = output_model_path.replace(".pth", "_with_lora.pth")
            torch.save(model.state_dict(), lora_path)
            print(f"Model with LoRA adapters saved to {lora_path}")

    if config.log_wandb:
        wandb.finish()

    return best_val_loss


def main():
    """Main function to run LoRA fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Fine-tune model on professional game data using LoRA"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pre-trained model to fine-tune",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the fine-tuned model (default: model_path with _pro_lora suffix)",
    )

    parser.add_argument(
        "--pro-data-path",
        type=str,
        default=os.path.join(RAW_PRO_GAMES_DIR, "pro_games.parquet"),
        help="Path to the professional games parquet file",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the wandb run",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (auto, cpu, cuda, mps)",
    )

    args = parser.parse_args()

    # Set up output path if not specified
    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path)
        name_parts = model_name.rsplit(".", 1)
        args.output_path = os.path.join(
            model_dir,
            f"{name_parts[0]}_pro_lora.{name_parts[1] if len(name_parts) > 1 else 'pth'}",
        )

    # Load pro game data
    pro_games_df = pd.read_parquet(args.pro_data_path)
    print(f"Loaded {len(pro_games_df)} professional games from {args.pro_data_path}")

    # Load patch mapping to filter for compatible patches
    patch_mapping_path = Path(PREPARED_DATA_DIR) / "patch_mapping.pkl"
    with open(patch_mapping_path, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]

    # Filter games
    pro_games_df, patches, filter_stats = filter_pro_games(pro_games_df, patch_mapping)

    print(f"Original model was trained on {len(patches)} patches: {', '.join(patches)}")
    print(
        f"Filtered out {filter_stats['missing_champion_ids']} games with missing or invalid champion IDs"
    )
    print(
        f"Filtered out {filter_stats['incompatible_patches']} games with incompatible patches"
    )
    print(f"Remaining pro games for fine-tuning: {filter_stats['remaining_count']}")

    if len(pro_games_df) == 0:
        print("Error: No compatible pro games found. Cannot proceed with fine-tuning.")
        return

    # Create configuration for LoRA fine-tuning
    config = LoRAConfig()

    # Set device from command line argument
    config.device = args.device

    print(f"LoRA fine-tuning configuration:")
    print(config)

    # Set random seeds for reproducibility
    set_random_seeds()

    # Start fine-tuning with LoRA
    best_val_loss = fine_tune_model_lora(
        pretrained_model_path=args.model_path,
        pro_games_df=pro_games_df,
        config=config,
        output_model_path=args.output_path,
        run_name=args.run_name,
    )

    print(
        f"LoRA fine-tuning complete. Best model saved to {args.output_path} with validation loss {best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
