# utils/match_prediction/train.py
import torch.nn as nn
import torch
import os
import pickle
from typing import Any, Dict, List, Tuple

from utils.match_prediction import ENCODERS_PATH
from utils.match_prediction.column_definitions import (
    COLUMNS,
    ColumnType,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


def get_optimizer_grouped_parameters(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    # Get all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Separate parameters into three groups:
    # 1. log variance parameters (no decay) (source that explains it https://chatgpt.com/c/679686b5-7634-8008-bed6-398cc9ccdecc)
    # 2. weight matrices (with decay)
    # 3. biases and normalization layer    
    # decaying biases and normalization layers is not needed
    # source: https://youtu.be/l8pRSuU81PU?si=f_taru0joQ5LW19e&t=8861s (no decay)
    log_var_params = [p for n, p in param_dict.items() if "log_vars" in n]
    non_log_var_params = [
        p for n, p in param_dict.items() if "log_vars" not in n and p.dim() >= 2
    ]
    bias_and_norm_params = [
        p for n, p in param_dict.items() if "log_vars" not in n and p.dim() < 2
    ]

    # Create optimizer groups
    optim_groups = [
        {"params": non_log_var_params, "weight_decay": weight_decay},
        {"params": bias_and_norm_params, "weight_decay": 0.0},
        {"params": log_var_params, "weight_decay": 0.0},
    ]

    # Print statistics
    num_decay_params = sum(p.numel() for p in non_log_var_params)
    num_nodecay_params = sum(p.numel() for p in bias_and_norm_params)
    num_logvar_params = sum(p.numel() for p in log_var_params)
    print(
        f"num decayed parameter tensors: {len(non_log_var_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(bias_and_norm_params)}, with {num_nodecay_params:,} parameters"
    )
    print(
        f"num log variance parameter tensors: {len(log_var_params)}, with {num_logvar_params:,} parameters"
    )

    return optim_groups


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_num_champions() -> tuple[int, int]:
    """Get the total number of champion embeddings needed and the ID for unknown champions.

    Returns:
        tuple[int, int]: (num_champions, unknown_champion_id)
    """
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    champion_encoder = label_encoders["champion_ids"]
    # Get the unknown champion ID directly from the encoder
    unknown_champion_id = champion_encoder.transform(["UNKNOWN"])[0]
    # Total number of embeddings is simply the number of classes in the encoder
    num_champions = len(champion_encoder.classes_)

    return num_champions, unknown_champion_id


# TODO: here are 2? why? delete one?
def collate_fn(
    batch: List[Dict[str, Any]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
