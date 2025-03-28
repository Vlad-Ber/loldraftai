# scripts/match-prediction/validation.py
# TODO: needs to be updated for new model architecture
import argparse
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from typing import Dict, List, Callable, Any, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.match_prediction import (
    get_best_device,
    MODEL_PATH,
    TRAIN_BATCH_SIZE,
    CHAMPION_ID_ENCODER_PATH,
    load_model_state_dict,
)
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.match_dataset import MatchDataset
from utils.match_prediction.model import Model
from utils.match_prediction.train_utils import collate_fn
from utils.match_prediction.task_definitions import TaskDefinition, TaskType
from utils import DATA_DIR, champion_play_rates_path


def get_elo_subgroup(feature: torch.Tensor) -> str:
    """
    Determine the ELO subgroup for a sample based on its ELO feature.

    Parameters:
        feature (torch.Tensor): ELO feature for the sample (scalar tensor).

    Returns:
        str: Subgroup name.
    """
    # feature: scalar tensor
    elo = feature.item()
    subgroup_name = f"elo_{elo}"
    return subgroup_name


with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
    champion_id_encoder: LabelEncoder = pickle.load(f)["mapping"]


def get_patch_subgroup(feature: torch.Tensor) -> str:
    """
    Assign a subgroup based on the patch number.
    """
    patch_number = feature.item()
    subgroup_name = f"patch_{patch_number}"
    return subgroup_name


def get_playrate_subgroup_batch(
    champion_ids_batch: torch.Tensor,  # Shape: [batch_size, 10]
    play_rates: Dict[str, Dict[str, Dict[str, float]]],
    patches_batch: torch.Tensor,  # Shape: [batch_size]
) -> List[str]:
    """
    Vectorized version that processes entire batch at once.
    """
    batch_size = champion_ids_batch.shape[0]

    play_rate_buckets = [
        (0.000001, "almost_never"),
        (0.00005, "ultra_rare"),
        (0.0001, "very_rare"),
        (0.0005, "rare"),
        (0.001, "quite_rare"),
        (0.005, "moderately_common"),
        (0.01, "common"),
        (1.0, "very_common"),
    ]
    role_names = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

    # Convert champion IDs for entire batch at once
    champion_ids_flat = champion_ids_batch.view(-1)  # Shape: [batch_size * 10]
    champion_names = champion_id_encoder.inverse_transform(
        champion_ids_flat.cpu().numpy()
    )

    # Initialize play rate tensor
    play_rates_tensor = torch.ones(
        batch_size, 10, dtype=torch.float32
    )  # Shape: [batch_size, 10]

    patches = patches_batch.cpu().numpy()
    for b in range(batch_size):
        patch = str(patches[b])
        for pos in range(10):
            champ_name = champion_names[b * 10 + pos]
            role = role_names[pos % 5]
            play_rates_tensor[b, pos] = (
                play_rates.get(patch, {}).get(champ_name, {}).get(role, 0.0)
            )

    # Ensure minimum play rate and find minimum per sample
    play_rates_tensor = torch.maximum(play_rates_tensor, torch.tensor(1e-7))
    min_play_rates, _ = play_rates_tensor.min(dim=1)  # Shape: [batch_size]

    # Vectorized bucket assignment
    result = []
    for min_rate in min_play_rates.cpu().numpy():
        for threshold, bucket_name in play_rate_buckets:
            if min_rate <= threshold:
                result.append(f"playrate_{bucket_name}")
                break
        else:
            result.append("playrate_unknown")

    return result


def validate_with_subgroups(
    model: Model,
    test_loader: DataLoader,
    device: torch.device,
    play_rates: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Validate the model and perform subgroup analysis.

    Parameters:
        model (Model): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the computations on.
        play_rates (Dict[Tuple[int, int], float]): Play rates for champion-role combinations.

    Returns:
        pd.DataFrame: DataFrame containing the subgroup analysis results.
    """
    model.eval()

    subgroup_total_loss: Dict[str, float] = defaultdict(float)
    subgroup_counts: Dict[str, int] = defaultdict(int)
    total_samples = 0

    TASKS = {
        "win_prediction": TaskDefinition(
            name="win_prediction",
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=1,
        ),
    }

    task_names = list(TASKS.keys())
    task_weights = torch.tensor(
        [TASKS[name].weight for name in task_names], device=device
    )  # Shape: [num_tasks]

    criterions = {}
    # could refactor code in common with train.py
    for task_name, task_def in TASKS.items():
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            criterions[task_name] = nn.BCEWithLogitsLoss(reduction="none")
        elif task_def.task_type == TaskType.REGRESSION:
            criterions[task_name] = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for features_batch, labels_batch in tqdm(test_loader, desc="Validating"):
            # Move features and labels to device
            features_batch = {k: v.to(device) for k, v in features_batch.items()}
            labels_batch = {k: labels_batch[k].to(device) for k in TASKS.keys()}

            batch_size = labels_batch["win_prediction"].size(0)
            total_samples += batch_size

            # Compute model outputs
            outputs_batch = model(features_batch)  # Outputs are dict of tensors

            # Compute losses for entire batch at once
            batch_losses = []
            for task_name in task_names:
                output = outputs_batch[task_name].view(
                    batch_size, -1
                )  # [batch_size, feat_dim]
                label = labels_batch[task_name].view(
                    batch_size, -1
                )  # [batch_size, feat_dim]
                task_loss = criterions[task_name](output, label)  # [batch_size]
                batch_losses.append(task_loss)

            # Stack and compute total loss for all samples at once
            batch_losses_tensor = torch.stack(
                batch_losses, dim=1
            )  # [batch_size, num_tasks]
            total_losses = (batch_losses_tensor * task_weights).sum(
                dim=1
            )  # [batch_size]

            # Vectorized subgroup assignment
            elo_subgroups = [f"elo_{elo}" for elo in features_batch["elo"].tolist()]
            patch_subgroups = [
                f"patch_{patch}" for patch in features_batch["patch"].tolist()
            ]
            playrate_subgroups = get_playrate_subgroup_batch(
                features_batch["champion_ids"], play_rates, features_batch["patch"]
            )

            # Process all subgroups at once using numpy operations
            all_subgroups = elo_subgroups + patch_subgroups + playrate_subgroups
            all_losses = total_losses.repeat_interleave(
                3
            )  # Repeat each loss 3 times for the 3 subgroup types

            for sg, loss in zip(all_subgroups, all_losses):
                subgroup_total_loss[sg] += loss.item()
                subgroup_counts[sg] += 1

    # Prepare results
    data = {
        "mean_loss": {
            k: subgroup_total_loss[k] / subgroup_counts[k] for k in subgroup_total_loss
        },
        "count": subgroup_counts,
        "percentage": {
            k: 100.0 * subgroup_counts[k] / total_samples for k in subgroup_counts
        },
    }
    df = pd.DataFrame(data)

    return df


def main():
    """
    Main function to perform validation and subgroup analysis.
    """
    parser = argparse.ArgumentParser(description="Validate the match-prediction model")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use only a small subset of the dataset for quick iteration",
    )
    args = parser.parse_args()

    device = get_best_device()

    # Initialize model
    config = TrainingConfig()
    model = Model(config=config, dropout=0, hidden_dims=config.hidden_dims)
    model = load_model_state_dict(model, device=device, path=MODEL_PATH)
    model.eval()

    # Initialize test dataset and loader
    test_dataset = MatchDataset(
        train_or_test="test", dataset_fraction=0.01 if args.small else 1.0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    with open(champion_play_rates_path, "r") as f:
        play_rates = json.load(f)

    # Run validation and subgroup analysis
    results_df = validate_with_subgroups(model, test_loader, device, play_rates)

    # Print and save results
    print("\nValidation Results:")
    print(results_df)

    # Save results to CSV
    results_df.to_csv(DATA_DIR + "/validation_results.csv")
    print("\nSaved validation results to 'validation_results.csv'")


if __name__ == "__main__":
    main()
