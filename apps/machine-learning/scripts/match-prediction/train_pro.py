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
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

from utils.match_prediction import (
    get_best_device,
    ENCODERS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    RAW_PRO_GAMES_DIR,
    NUMERICAL_STATS_PATH,
    PREPARED_DATA_DIR,
)
from utils.match_prediction.model import Model
from utils.match_prediction.train import (
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
        self.num_epochs = 20
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.dropout = 0.1
        self.batch_size = 16
        self.val_split = 0.2
        self.hidden_dims = [768, 384, 192, 96]
        self.embed_dim = 128
        self.max_grad_norm = 1.0
        self.log_wandb = True
        self.log_batch_interval = 5  # How often to log batch progress
        self.save_checkpoints = True

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
        numerical_stats: Dict[str, Dict[str, float]],
        train_or_test: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.unknown_champion_id = unknown_champion_id
        self.label_encoders = label_encoders
        self.numerical_stats = numerical_stats

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

            # Verify we have 10 champions
            if len(champion_ids) != 10:
                print(
                    f"WARNING: Row {idx} has {len(champion_ids)} champions instead of 10!"
                )
                print(f"Row data: {row}")
                raise ValueError(f"Expected 10 champions, got {len(champion_ids)}")

            # Convert champion IDs to encoded format
            encoded_champs = self.label_encoders["champion_ids"].transform(champion_ids)
            features["champion_ids"] = torch.tensor(encoded_champs, dtype=torch.long)
        except Exception as e:
            print(f"ERROR processing champion IDs for row {idx}: {e}")
            print(f"Row data: {row}")
            raise

        # Process numerical features
        for col in NUMERICAL_COLUMNS:
            if col == "numerical_elo":
                # Pro games are highest ELO
                val = 0
            elif col == "numerical_patch":
                # Calculate numerical patch
                major = row["gameVersionMajorPatch"]
                minor = row["gameVersionMinorPatch"]
                val = major * 50 + minor
            else:
                continue

            # Normalize using pre-computed stats
            mean = self.numerical_stats["means"].get(col, 0.0)
            std = self.numerical_stats["stds"].get(col, 1.0)

            if std > 0:
                normalized_val = (val - mean) / std
            else:
                normalized_val = val - mean

            features[col] = torch.tensor(normalized_val, dtype=torch.float32)

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
    pro_games_df, unknown_champion_id, label_encoders, numerical_stats, config
):
    """Create train and validation dataloaders for fine-tuning"""
    # Create datasets
    train_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        unknown_champion_id=unknown_champion_id,
        label_encoders=label_encoders,
        numerical_stats=numerical_stats,
        train_or_test="train",
        val_split=config.val_split,
    )

    val_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        unknown_champion_id=unknown_champion_id,
        label_encoders=label_encoders,
        numerical_stats=numerical_stats,
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


def fine_tune_model(
    pretrained_model_path: str,
    pro_games_df: pd.DataFrame,
    config: FineTuningConfig,
    output_model_path: str,
    run_name: Optional[str] = None,
):
    """Fine-tune a pre-trained model on professional game data"""
    device = get_best_device()
    print(f"Using device: {device}")

    # Load resources
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    with open(MODEL_CONFIG_PATH, "rb") as f:
        model_config = pickle.load(f)

    # Load numerical stats for normalization
    with open(NUMERICAL_STATS_PATH, "rb") as f:
        numerical_stats = pickle.load(f)

    # Get number of champions
    num_champions, unknown_champion_id = get_num_champions()

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        pro_games_df, unknown_champion_id, label_encoders, numerical_stats, config
    )

    # Initialize the model
    model = Model(
        num_categories=model_config["num_categories"],
        num_champions=model_config["num_champions"],
        embed_dim=config.embed_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    # Load pre-trained weights
    print(f"Loading pre-trained model from {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.to(device)

    # Initialize wandb
    if config.log_wandb:
        wandb.init(
            project="draftking-pro-finetune",
            name=run_name or f"finetune-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                **config.to_dict(),
                "pretrained_model": pretrained_model_path,
                "pro_games_count": len(pro_games_df),
            },
        )
        wandb.watch(model, log_freq=100)

    # Define loss function for win prediction only
    criterion = {"win_prediction": nn.BCEWithLogitsLoss()}

    # Configure optimizer
    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, config.weight_decay),
        lr=config.learning_rate,
    )

    # Fine-tune the model
    best_val_metric = 0.0  # Track best accuracy
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
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion["win_prediction"](
                outputs["win_prediction"], labels["win_prediction"]
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # Calculate accuracy
            probs = torch.sigmoid(outputs["win_prediction"])
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels["win_prediction"]).sum().item()
            train_total += labels["win_prediction"].size(0)

            # Log batch progress - minimal console output
            if batch_idx % config.log_batch_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config.num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
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
                features = {k: v.to(device) for k, v in features.items()}
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(features)
                loss = criterion["win_prediction"](
                    outputs["win_prediction"], labels["win_prediction"]
                )

                val_loss += loss.item()
                val_steps += 1

                # Calculate accuracy
                probs = torch.sigmoid(outputs["win_prediction"])
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels["win_prediction"]).sum().item()
                val_total += labels["win_prediction"].size(0)

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float("inf")
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        epoch_time = time.time() - epoch_start

        # Minimal console output - let wandb handle the metrics
        print(f"Epoch {epoch+1}/{config.num_epochs} completed in {epoch_time:.2f}s")

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

        # Save best model based on validation accuracy
        if val_accuracy > best_val_metric:
            best_val_metric = val_accuracy
            best_model_state = model.state_dict().copy()

            # Save intermediate best model
            if config.save_checkpoints:
                intermediate_path = (
                    f"{output_model_path.rsplit('.', 1)[0]}_epoch{epoch+1}.pth"
                )
                torch.save(best_model_state, intermediate_path)
                print(f"New best model with validation accuracy: {best_val_metric:.4f}")

    # Save the final best model
    if best_model_state is not None:
        torch.save(best_model_state, output_model_path)
        print(
            f"Best model saved to {output_model_path} with accuracy {best_val_metric:.4f}"
        )
    else:
        # Save the final model if no best model was found
        torch.save(model.state_dict(), output_model_path)
        print(f"Final model saved to {output_model_path}")

    if config.log_wandb:
        wandb.finish()

    return best_val_metric


def main():
    """Main function to run fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Fine-tune model on professional game data"
    )

    # Keep only essential path-related arguments
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
        help="Path to save the fine-tuned model (default: model_path with _pro_finetuned suffix)",
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

    args = parser.parse_args()

    # Set up output path if not specified
    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path)
        name_parts = model_name.rsplit(".", 1)
        args.output_path = os.path.join(
            model_dir,
            f"{name_parts[0]}_pro_finetuned.{name_parts[1] if len(name_parts) > 1 else 'pth'}",
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

    # Create configuration for fine-tuning - edit this object directly in the code
    config = FineTuningConfig()

    print(f"Fine-tuning configuration:")
    print(config)

    # Set random seeds for reproducibility
    set_random_seeds()

    # Start fine-tuning
    best_accuracy = fine_tune_model(
        pretrained_model_path=args.model_path,
        pro_games_df=pro_games_df,
        config=config,
        output_model_path=args.output_path,
        run_name=args.run_name,
    )

    print(
        f"Fine-tuning complete. Best model saved to {args.output_path} with accuracy {best_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
