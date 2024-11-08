# Utils/match_prediction/model.py

import torch
import torch.nn as nn
import pickle

from utils.match_prediction import ENCODERS_PATH
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


class MatchOutcomeModel(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        embed_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.1,
    ):
        super(MatchOutcomeModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_positions = len(POSITIONS) * 2  # Assuming two teams

        # Define context features (all features except 'champion_ids')
        self.context_categorical_features = CATEGORICAL_COLUMNS
        self.context_numerical_features = NUMERICAL_COLUMNS

        # Embeddings for categorical context features
        self.embeddings = nn.ModuleDict()
        for col in self.context_categorical_features:
            self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Embedding for champions
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)

        # Positional embedding for roles
        self.position_embedding = nn.Embedding(self.num_positions, embed_dim)

        # Transformer Encoder for champion embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Projection layer for numerical context features
        self.numerical_context_proj = nn.Linear(
            len(self.context_numerical_features), embed_dim
        )

        self.role_percentage_proj = nn.Linear(5, embed_dim)  # 5 roles

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(
                embed_dim, 128, bias=False
            ),  # bias false because before batchnorm
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Create output layers for each task
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                self.output_layers[task_name] = nn.Linear(
                    128, 1
                )  # Remove Sigmoid, needed for BCEWithLogitsLoss(which is needed for autocast)
            elif task_def.task_type == TaskType.REGRESSION:
                self.output_layers[task_name] = nn.Linear(128, 1)

    def forward(self, features):
        # Compute context vector
        context_vector = self.compute_context_vector(features)

        # Process champion embeddings
        champion_features = self.process_champion_embeddings(features, context_vector)

        # Pass through fully connected layers
        x = self.fc(champion_features)  # Shared representation

        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs

    def compute_context_vector(self, features):
        """
        Computes the context vector by embedding categorical and numerical context features.
        """
        # Embed categorical context features and sum them
        context_embeds = []
        for col in self.context_categorical_features:
            embed = self.embeddings[col](features[col])  # [batch_size, embed_dim]
            context_embeds.append(embed)

        # Project numerical context features into embedding space
        if self.context_numerical_features:
            numerical_context_features = torch.stack(
                [features[col] for col in self.context_numerical_features], dim=1
            )  # [batch_size, num_context_numerical_features]
            numerical_context_embed = self.numerical_context_proj(
                numerical_context_features
            )  # [batch_size, embed_dim]
            context_embeds.append(numerical_context_embed)

        # Sum all context embeddings to create the context vector
        context_vector = torch.stack(context_embeds, dim=0).sum(
            dim=0
        )  # [batch_size, embed_dim]

        return context_vector

    def process_champion_embeddings(self, features, context_vector):
        """
        Processes champion embeddings by adding position embeddings and context vector,
        then passing through a transformer encoder and pooling.
        """
        batch_size, num_champions = features["champion_ids"].size()
        device = features["champion_ids"].device

        # Embed champions
        champion_embeds = self.champion_embedding(
            features["champion_ids"]
        )  # [batch_size, num_champions, embed_dim]

        # Process champion role percentages
        role_percentages = features[
            "champion_role_percentages"
        ]  # [batch_size, num_champions, 5]
        role_percentage_embeds = self.role_percentage_proj(
            role_percentages
        )  # [batch_size, num_champions, embed_dim]

        # Embed positions
        position_indices = (
            torch.arange(num_champions, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )  # [batch_size, num_champions]
        position_embeds = self.position_embedding(
            position_indices
        )  # [batch_size, num_champions, embed_dim]

        # Expand context vector to match champion embeddings
        context_vector_expanded = context_vector.unsqueeze(
            1
        )  # [batch_size, 1, embed_dim]

        # Sum champion, position, and context embeddings
        champion_inputs = (
            champion_embeds
            + position_embeds
            + context_vector_expanded
            + role_percentage_embeds
        )  # [batch_size, num_champions, embed_dim]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            champion_inputs
        )  # [batch_size, num_champions, embed_dim]

        # Pooling: mean over the sequence dimension
        champion_features = transformer_output.mean(dim=1)  # [batch_size, embed_dim]

        return champion_features


import torch
import torch.nn as nn
import pickle

from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


import torch
import torch.nn as nn
import pickle

from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


class SimpleMatchModel(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        embed_dim=64,
        hidden_dims=[256, 128],
        dropout=0.1,
    ):
        super(SimpleMatchModel, self).__init__()

        # Store dimensions for debugging
        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for col in CATEGORICAL_COLUMNS:
            self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Champion embeddings
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)

        # Project numerical features
        self.numerical_projection = (
            nn.Linear(len(NUMERICAL_COLUMNS), embed_dim) if NUMERICAL_COLUMNS else None
        )

        # Project role percentages
        self.role_projection = nn.Linear(5, embed_dim)  # 5 roles

        # Calculate total input dimension for MLP
        num_categorical = len(CATEGORICAL_COLUMNS)  # Categorical features
        num_champions = len(POSITIONS) * 2  # Champions from both teams
        num_numerical_projections = (
            1 if NUMERICAL_COLUMNS else 0
        )  # Projected numerical features
        total_embed_features = (
            num_categorical + num_champions + num_numerical_projections
        )
        mlp_input_dim = total_embed_features * embed_dim

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions}")
        print(f"- Numerical features projection: {num_numerical_projections}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # MLP layers
        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
            embed = self.embeddings[col](features[col])  # [batch_size, embed_dim]
            embeddings_list.append(embed)

        # Process champion features
        champion_ids = features["champion_ids"]  # [batch_size, num_champions]
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # [batch_size, num_champions, embed_dim]

        # Combine champion and role embeddings(empty for now) #TODO: remove
        champion_features = champion_embeds

        # Reshape champion features to [batch_size, num_champions * embed_dim]
        champion_features = champion_features.view(batch_size, -1)
        embeddings_list.append(champion_features)

        # Process numerical features if they exist
        if self.numerical_projection is not None and NUMERICAL_COLUMNS:
            numerical_features = torch.stack(
                [features[col] for col in NUMERICAL_COLUMNS], dim=1
            )
            numerical_embed = self.numerical_projection(
                numerical_features
            )  # [batch_size, embed_dim]
            embeddings_list.append(numerical_embed)

        # Concatenate all features
        combined_features = torch.cat(embeddings_list, dim=1)

        # Debug print dimensions
        if combined_features.shape[1] != self.mlp[0].weight.shape[1]:
            print(f"\nDimension mismatch!")
            print(f"Combined features shape: {combined_features.shape}")
            print(f"First MLP layer input dim: {self.mlp[0].weight.shape[1]}")
            print(f"First MLP layer weight shape: {self.mlp[0].weight.shape}")
            for i, embedding in enumerate(embeddings_list):
                print(f"Embedding {i} shape: {embedding.shape}")

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs for each task
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs


if __name__ == "__main__":
    # Example usage
    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    num_champions = 200

    embed_dim = 64
    num_heads = 8
    num_transformer_layers = 2

    model = SimpleMatchModel(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=embed_dim,
    )
    # Print model architecture
    print(model)

    # Calculate and print model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"\nModel Size: {size_all_mb:.3f} MB")

    # Print sizes of individual layers
    print("\nLayer Sizes:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            layer_size = (
                sum(p.nelement() * p.element_size() for p in module.parameters())
                / 1024**2
            )
            print(f"{name}: {layer_size:.3f} MB")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")
