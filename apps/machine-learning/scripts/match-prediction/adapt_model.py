import torch
import torch.nn as nn
import argparse
import pickle
from utils.match_prediction import (
    MODEL_PATH,
    PATCH_MAPPING_PATH,
    CHAMPION_ID_ENCODER_PATH,
)
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.model import Model


def load_model(config: TrainingConfig):
    """Initialize a new model with the current configuration."""
    return Model(config)


def save_model(model: Model, model_path: str):
    """Save the updated model to the specified path."""
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def slide_embeddings(model: Model, old_num_patches: int):
    """Slide embeddings when the number of patches remains the same."""
    with torch.no_grad():
        # Ensure we don’t exceed the old model’s embeddings
        weights_to_shift = min(old_num_patches, model.num_patches)
        # Shift patch embeddings: 1->0, 2->1, ..., (N-1)->(N-2), (N-1) initialized with (N-2)
        model.patch_embedding.weight[: weights_to_shift - 1] = (
            model.patch_embedding.weight[1:weights_to_shift].clone()
        )
        if weights_to_shift > 1:
            model.patch_embedding.weight[weights_to_shift - 1] = (
                model.patch_embedding.weight[weights_to_shift - 2].clone()
            )

        # Shift champion_patch embeddings for each champion
        num_champions = model.num_champions
        num_patches = model.num_patches
        for c in range(num_champions):
            start_idx = c * num_patches
            old_start_idx = c * old_num_patches
            # Shift within the old number of patches
            if (
                old_start_idx + weights_to_shift
                <= model.champion_patch_embedding.weight.size(0)
            ):
                model.champion_patch_embedding.weight[
                    start_idx : start_idx + weights_to_shift - 1
                ] = model.champion_patch_embedding.weight[
                    start_idx + 1 : start_idx + weights_to_shift
                ].clone()
                if weights_to_shift > 1:
                    model.champion_patch_embedding.weight[
                        start_idx + weights_to_shift - 1
                    ] = model.champion_patch_embedding.weight[
                        start_idx + weights_to_shift - 2
                    ].clone()


def add_new_patch(model: Model, old_num_patches: int):
    """Initialize new patch embeddings when the number of patches increases."""
    with torch.no_grad():
        # Initialize new patch embeddings with the last old patch embedding
        if old_num_patches > 0:
            last_patch_weight = model.patch_embedding.weight[
                old_num_patches - 1
            ].clone()
            for i in range(old_num_patches, model.num_patches):
                model.patch_embedding.weight[i] = last_patch_weight

        # Initialize new champion_patch embeddings
        num_champions = model.num_champions
        for c in range(num_champions):
            old_start_idx = c * old_num_patches
            new_start_idx = c * model.num_patches
            if old_num_patches > 0:
                last_old_idx = min(
                    old_start_idx + old_num_patches - 1,
                    model.champion_patch_embedding.weight.size(0) - 1,
                )
                last_patch_weight = model.champion_patch_embedding.weight[
                    last_old_idx
                ].clone()
                for i in range(old_num_patches, model.num_patches):
                    idx = new_start_idx + i
                    if idx < model.champion_patch_embedding.weight.size(0):
                        model.champion_patch_embedding.weight[idx] = last_patch_weight


def main():
    parser = argparse.ArgumentParser(description="Adapt model for online learning")
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["slide", "add"],
        required=True,
        help="Action to perform: 'slide' for sliding embeddings, 'add' for adding a new patch",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the updated model (default: overwrite input model)",
    )
    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig()

    # Load current patch mapping to determine new_num_patches
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]
    new_num_patches = len(patch_mapping)

    # Load pre-trained state_dict
    state_dict = torch.load(args.model_path)

    # Determine old_num_patches from state_dict
    old_patch_weight = state_dict["patch_embedding.weight"]
    old_num_patches = old_patch_weight.size(0)

    # Initialize the model with current configuration (reflects current patch mapping)
    model = load_model(config)

    # Load state_dict with strict=False to handle potential size mismatches
    model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded pre-trained state_dict with {old_num_patches} patches into model with {model.num_patches} patches."
    )

    if args.action == "slide":
        if new_num_patches != old_num_patches:
            print(
                f"Warning: 'slide' action with new_num_patches ({new_num_patches}) != old_num_patches ({old_num_patches}). Proceeding with shift."
            )
        slide_embeddings(model, old_num_patches)
        print("Embeddings slid successfully.")
    elif args.action == "add":
        if new_num_patches <= old_num_patches:
            print(
                f"Warning: 'add' action but new_num_patches ({new_num_patches}) <= old_num_patches ({old_num_patches}). Proceeding with initialization."
            )
        add_new_patch(model, old_num_patches)
        print("New patch embeddings initialized successfully.")

    # Save the updated model
    output_path = args.output_path if args.output_path else args.model_path
    save_model(model, output_path)


if __name__ == "__main__":
    main()
