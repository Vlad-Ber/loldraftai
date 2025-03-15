import torch
import torch.nn as nn
import pickle
from utils.match_prediction import NUMERICAL_STATS_PATH, PATCH_MAPPING_PATH
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


class Model(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        embed_dim=64,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_champions = num_champions

        # Load patch mapping and stats
        with open(PATCH_MAPPING_PATH, "rb") as f:
            self.patch_mapping = pickle.load(f)["mapping"]
        with open(NUMERICAL_STATS_PATH, "rb") as f:
            numerical_stats = pickle.load(f)
            self.patch_mean = numerical_stats["means"]["numerical_patch"]
            self.patch_std = numerical_stats["stds"]["numerical_patch"]

        # Number of unique patches
        self.num_patches = len(self.patch_mapping)
        self.patch_values = torch.tensor(
            list(self.patch_mapping.keys()), dtype=torch.float32
        )

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for col in CATEGORICAL_COLUMNS:
            self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Patch embedding (general meta changes)
        self.patch_embedding = nn.Embedding(self.num_patches, embed_dim)

        # Champion+patch embeddings (champion-specific changes)
        self.champion_patch_embedding = nn.Embedding(
            num_champions * self.num_patches, embed_dim
        )

        # Project numerical features (e.g., elo)
        self.numerical_projection = (
            nn.Linear(1, embed_dim) if "numerical_elo" in NUMERICAL_COLUMNS else None
        )

        # Total input dimension for MLP
        num_categorical = len(CATEGORICAL_COLUMNS)
        num_champions_in_game = len(POSITIONS) * 2  # 10 champions
        num_numerical_projections = 1 if "numerical_elo" in NUMERICAL_COLUMNS else 0
        total_embed_features = (
            num_categorical + num_champions_in_game + num_numerical_projections + 1
        )  # +1 for patch_embed
        mlp_input_dim = total_embed_features * embed_dim

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions_in_game}")
        print(f"- Numerical features projection: {num_numerical_projections}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # MLP
        layers = []
        prev_dim = mlp_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            dropout_rate = dropout if i < len(hidden_dims) - 1 else dropout * 0.5
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Output layers
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)

    def map_numerical_to_patch_id(self, numerical_patch):
        """Convert normalized numerical_patch to a patch index."""
        raw_patch = numerical_patch * self.patch_std + self.patch_mean
        distances = torch.abs(
            self.patch_values.to(raw_patch.device) - raw_patch.unsqueeze(-1)
        )
        patch_indices = torch.argmin(distances, dim=-1)
        patch_indices = torch.clamp(patch_indices, 0, self.num_patches - 1)
        return patch_indices

    def forward(self, features):
        batch_size = features["champion_ids"].size(0)
        embeddings_list = []

        # Categorical features
        for col in CATEGORICAL_COLUMNS:
            embed = self.embeddings[col](features[col])
            embeddings_list.append(embed)

        # Patch embedding (general meta)
        numerical_patch = features["numerical_patch"]  # Normalized
        patch_indices = self.map_numerical_to_patch_id(numerical_patch)  # (batch_size,)
        patch_embed = self.patch_embedding(patch_indices)  # (batch_size, embed_dim)
        embeddings_list.append(patch_embed)

        # Champion+patch embeddings
        champion_ids = features["champion_ids"]  # (batch_size, 10)
        patch_indices_expanded = patch_indices.unsqueeze(1).expand(
            -1, 10
        )  # (batch_size, 10)
        combined_indices = (
            champion_ids * self.num_patches + patch_indices_expanded
        )  # (batch_size, 10)
        champ_patch_embeds = self.champion_patch_embedding(
            combined_indices
        )  # (batch_size, 10, embed_dim)
        champion_features = champ_patch_embeds.view(
            batch_size, -1
        )  # (batch_size, 10*embed_dim)
        embeddings_list.append(champion_features)

        # Numerical features (e.g., elo)
        if self.numerical_projection and "numerical_elo" in NUMERICAL_COLUMNS:
            numerical_elo = features["numerical_elo"].unsqueeze(-1)  # (batch_size, 1)
            numerical_embed = self.numerical_projection(numerical_elo)
            embeddings_list.append(numerical_embed)

        # Concatenate all features
        combined_features = torch.cat(embeddings_list, dim=1)

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs
