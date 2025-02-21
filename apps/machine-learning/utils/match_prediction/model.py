# utils/match_prediction/model.py

import torch
import torch.nn as nn
from typing import Dict, List
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


#######################################
# Model Definition
#######################################
class SimpleMatchModel(nn.Module):
    """
    A modular Transformer-based model that treats each champion and numerical column as tokens in a sequence.

    **Updated Architecture**:
    1) Champion IDs -> Embedding -> [batch_size, #champs, embed_dim]
    2) Numerical columns -> Linear projections -> [batch_size, 1, embed_dim] each
       -> Concatenated into [batch_size, #numerical_cols, embed_dim]
    3) CLS token (learnable) added to the sequence -> [batch_size, 1, embed_dim]
    4) Concatenate CLS + champion tokens + numeric tokens -> [batch_size, total_tokens + 1, embed_dim]
    5) Optional token-type embeddings (disabled by default)
    6) Positional embeddings for all tokens (including CLS)
    7) Transformer encoder (deeper with smaller embeddings)
    8) Extract CLS token output instead of mean pooling -> [batch_size, embed_dim]
    9) MLP
    10) Task heads
    """

    def __init__(
        self,
        num_champions: int,
        embed_dim: int = 64,  # Smaller embeddings for efficiency
        hidden_dims: List[int] = [128, 64],  # Adjusted MLP dimensions
        dropout: float = 0.1,
        num_attention_heads: int = 4,  # Adjusted for smaller embed_dim
        num_transformer_layers: int = 5,  # 6,  # Deeper model
        dim_feedforward: int = 128,  # Reduced for efficiency
        use_token_types: bool = False,  # Disabled by default for simplicity
        tasks: Dict[str, Dict] = None,
    ):
        """
        **Parameters**:
        - `num_champions`: Number of unique champion IDs.
        - `embed_dim`: Dimensionality of each token embedding (default: 64).
        - `hidden_dims`: List of hidden layer sizes for the MLP (default: [128, 64]).
        - `dropout`: Dropout rate in Transformer and MLP (default: 0.1).
        - `num_attention_heads`: Number of attention heads in Transformer (default: 4).
        - `num_transformer_layers`: Number of Transformer layers (default: 3).
        - `dim_feedforward`: Feedforward dimension in Transformer (default: 128).
        - `use_token_types`: Whether to use token-type embeddings (default: False).
        - `tasks`: Dict of task_name -> { "task_type": TaskType } for output heads.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_token_types = use_token_types
        self.tasks = tasks if tasks is not None else TASKS

        ### Champion Embeddings
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)

        ### Numerical Feature Projections
        # Each numerical column gets its own linear projection: [batch_size, 1] -> [batch_size, embed_dim]
        self.numerical_embeddings = nn.ModuleDict()
        for col in NUMERICAL_COLUMNS:
            self.numerical_embeddings[col] = nn.Linear(1, embed_dim)

        ### CLS Token
        # Learnable parameter for the CLS token: [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        ### Token-Type Embeddings (Optional)
        if self.use_token_types:
            # 0: CLS, 1: champion tokens, 2: numerical tokens
            self.token_type_embedding = nn.Embedding(3, embed_dim)
        else:
            self.token_type_embedding = None

        ### Positional Embeddings
        # Total tokens = 1 (CLS) + #champion_ids (e.g., 10) + #numerical_columns
        self.num_champion_tokens = 10  # Adjust based on your data
        self.num_numerical_tokens = len(NUMERICAL_COLUMNS)
        self.total_tokens = 1 + self.num_champion_tokens + self.num_numerical_tokens
        self.pos_embedding = nn.Parameter(torch.randn(self.total_tokens, embed_dim))

        ### Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        ### MLP (applied to CLS token output)
        layers = []
        prev_dim = embed_dim
        for hd in hidden_dims:
            layers.append(
                nn.Linear(prev_dim, hd, bias=True)
            )  # Added bias back since we removed norm
            layers.append(nn.LayerNorm(hd))  # Changed from BatchNorm1d to LayerNorm
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        self.mlp = nn.Sequential(*layers)

        ### Output Heads (per task)
        self.output_layers = nn.ModuleDict()
        for task_name, task_info in self.tasks.items():
            task_type = task_info.task_type
            if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)

        ### Initialization Summary
        print("SimpleMatchModel initialized with:")
        print(f" - #Champion tokens: {self.num_champion_tokens}")
        print(
            f" - #Numerical tokens: {self.num_numerical_tokens} ({NUMERICAL_COLUMNS})"
        )
        print(f" - CLS token: True")
        print(f" - embed_dim: {embed_dim}, hidden_dims: {hidden_dims}")
        print(f" - num_transformer_layers: {num_transformer_layers}")
        print(f" - num_attention_heads: {num_attention_heads}")
        print(f" - use_token_types: {use_token_types}")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        **Input**:
        - `features["champion_ids"]`: LongTensor [batch_size, num_champion_tokens]
        - `features[col]` for each col in NUMERICAL_COLUMNS: FloatTensor [batch_size]

        **Output**:
        - Dict of task_name -> logits [batch_size]
        """
        batch_size = features["champion_ids"].shape[0]

        ### 1) Champion Tokens
        # [batch_size, num_champion_tokens, embed_dim]
        champ_embeds = self.champion_embedding(features["champion_ids"])

        ### 2) Numerical Tokens
        numeric_tokens = []
        for col in NUMERICAL_COLUMNS:
            vals = features[col].unsqueeze(-1)  # [batch_size, 1]
            embedded = self.numerical_embeddings[col](vals)  # [batch_size, embed_dim]
            embedded = embedded.unsqueeze(1)  # [batch_size, 1, embed_dim]
            numeric_tokens.append(embedded)

        ### 3) CLS Token
        # Expand CLS token to batch size: [batch_size, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, 1, self.embed_dim)

        ### 4) Concatenate CLS + Champion + Numeric Tokens
        # [batch_size, total_tokens + 1, embed_dim]
        tokens = torch.cat([cls_tokens, champ_embeds] + numeric_tokens, dim=1)

        ### 5) Token-Type Embeddings (Optional)
        if self.use_token_types:
            # Token-type IDs: 0 for CLS, 1 for champions, 2 for numerical
            token_type_ids = (
                [0] + [1] * self.num_champion_tokens + [2] * self.num_numerical_tokens
            )
            token_type_ids = (
                torch.tensor(token_type_ids, device=tokens.device)
                .unsqueeze(0)
                .expand(batch_size, self.total_tokens)
            )
            type_embeds = self.token_type_embedding(
                token_type_ids
            )  # [batch_size, total_tokens, embed_dim]
            tokens = tokens + type_embeds

        ### 6) Positional Embeddings
        # [batch_size, total_tokens + 1, embed_dim]
        pos_emb = self.pos_embedding.unsqueeze(0).expand(
            batch_size, self.total_tokens, self.embed_dim
        )
        tokens = tokens + pos_emb

        ### 7) Transformer Encoder
        # [batch_size, total_tokens + 1, embed_dim]
        encoded = self.transformer_encoder(tokens)

        ### 8) Extract CLS Token Output
        # [batch_size, embed_dim]
        pooled = encoded[:, 0, :]

        ### 9) MLP
        # [batch_size, hidden_dims[-1]]
        x = self.mlp(pooled)

        ### 10) Output Heads
        outputs = {}
        for task_name, layer in self.output_layers.items():
            logits = layer(x).squeeze(-1)  # [batch_size]
            outputs[task_name] = logits

        return outputs


#######################################
# Example Usage
#######################################
if __name__ == "__main__":
    # Example: 200 champions
    model = SimpleMatchModel(
        num_champions=200,
        embed_dim=64,  # Smaller embeddings
        hidden_dims=[128, 64],  # Adjusted MLP
        dropout=0.1,
        num_attention_heads=4,  # Suitable for embed_dim=64
        num_transformer_layers=3,  # Deeper model
        dim_feedforward=128,  # Reduced for efficiency
        use_token_types=False,  # Disabled by default
        tasks=TASKS,
    )

    # Dummy batch
    batch_size = 8
    feats = {
        "champion_ids": torch.randint(0, 200, (batch_size, 10)),
        "numerical_patch": torch.rand(batch_size),  # e.g., patch number
        "numerical_elo": torch.randint(800, 3000, (batch_size,)).float(),
    }

    # Forward pass
    out = model(feats)
    for k, v in out.items():
        print(f"{k}: {v.shape}")

    # Total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")


