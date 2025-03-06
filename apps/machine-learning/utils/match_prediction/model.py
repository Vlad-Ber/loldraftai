import torch
import torch.nn as nn
import pickle
import math

from utils.match_prediction import ENCODERS_PATH
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from utils.match_prediction.config import TrainingConfig


class Model(nn.Module):
    def __init__(
        self,
        num_categories,  # Dict of {col: num_categories} for categorical features
        num_champions,  # Total number of unique champions
        embed_dim=64,  # Embedding dimension
        hidden_dims=[256, 128, 64],  # MLP hidden layer sizes
        dropout=0.2,  # Dropout rate
    ):
        super(Model, self).__init__()
        self.embed_dim = embed_dim

        # Embeddings for categorical features (if any)
        self.embeddings = nn.ModuleDict()
        for col in CATEGORICAL_COLUMNS:
            self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Champion embeddings
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)

        # Separate projections for elo and patch
        self.elo_projection = nn.Linear(1, embed_dim)
        self.patch_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, embed_dim)
        )

        # Adjustment layer for patch-champion interaction
        self.adjustment_layer = nn.Linear(embed_dim, num_champions)

        # Calculate MLP input dimension
        num_categorical = len(CATEGORICAL_COLUMNS)
        num_champions_positions = 10  # 5 champions per team
        mlp_input_dim = (
            num_champions_positions * embed_dim
        ) + 2 * embed_dim  # champions, elo, patch
        if num_categorical > 0:
            mlp_input_dim += num_categorical * embed_dim

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions_positions}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # MLP layers
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

        # Output layers for each task
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)

    def forward(self, features):
        batch_size = features["champion_ids"].size(0)
        embeddings_list = []

        # Process categorical features
        for col in CATEGORICAL_COLUMNS:
            embed = self.embeddings[col](features[col])
            embeddings_list.append(embed)

        # Process patch and compute adjustments
        patch = features["numerical_patch"].unsqueeze(-1)  # (batch_size, 1)
        patch_embed = self.patch_mlp(patch)  # (batch_size, embed_dim)
        adjustment = self.adjustment_layer(patch_embed)  # (batch_size, num_champions)

        # Process champion features with patch adjustments
        champion_ids = features["champion_ids"]  # (batch_size, 10)
        adjustments_per_champion = adjustment.gather(
            1, champion_ids
        )  # (batch_size, 10)
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # (batch_size, 10, embed_dim)
        adjusted_champion_embeds = champion_embeds * (
            1 + adjustments_per_champion.unsqueeze(-1)
        )
        champion_features = adjusted_champion_embeds.view(
            batch_size, -1
        )  # (batch_size, 10*embed_dim)
        embeddings_list.append(champion_features)

        # Process elo
        elo = features["numerical_elo"].unsqueeze(-1)  # (batch_size, 1)
        elo_embed = self.elo_projection(elo)  # (batch_size, embed_dim)
        embeddings_list.append(elo_embed)

        # Append patch embedding
        embeddings_list.append(patch_embed)

        # Concatenate all features
        combined_features = torch.cat(embeddings_list, dim=1)

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs for each task
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs
