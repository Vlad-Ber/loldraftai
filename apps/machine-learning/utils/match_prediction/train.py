import torch.nn as nn
import torch
import os
import pickle
from utils.match_prediction import ENCODERS_PATH


def get_optimizer_grouped_parameters(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    # Get all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Separate parameters into decay and no-decay groups
    # dim >= 2 are the weight matrices, dim < 2 are biases
    # decaying biases and normalization layers is not needed
    # source: https://youtu.be/l8pRSuU81PU?si=f_taru0joQ5LW19e&t=8861
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    # Create optimizer groups
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Print statistics
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )

    return optim_groups


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_num_champions():
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    champion_encoder = label_encoders["champion_ids"]
    max_champion_id = max(
        int(champ_id) for champ_id in champion_encoder.classes_ if champ_id != "UNKNOWN"
    )
    unknown_champion_id = max_champion_id + 1
    num_champions = unknown_champion_id + 1  # Total number of embeddings

    return num_champions, unknown_champion_id
