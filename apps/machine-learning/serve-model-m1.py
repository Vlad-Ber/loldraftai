# serve-model-m1.py
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Literal, Dict, Optional
import uvicorn
import pickle
import time
import os
from pathlib import Path

from utils.match_prediction.column_definitions import COLUMNS, ColumnType
from utils.match_prediction import (
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    PREPARED_DATA_DIR,
)

# Set up Apple Silicon MPS acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration")
else:
    device = torch.device("cpu")
    print("MPS not available, falling back to CPU")

# Initialize FastAPI with minimal configuration
app = FastAPI(title="LoL Draft Model Server - M1 Optimized")

# Load resources
print("Loading resources...")

# Load label encoders
with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

# Load numerical stats for normalization
with open(NUMERICAL_STATS_PATH, "rb") as f:
    numerical_stats = pickle.load(f)

# Get patch mapping
with open(Path(PREPARED_DATA_DIR) / "patch_mapping.pkl", "rb") as f:
    patch_mapping = pickle.load(f)["mapping"]

# Load model config
with open(MODEL_CONFIG_PATH, "rb") as f:
    model_config = pickle.load(f)

# Load PyTorch model
from utils.match_prediction.model import Model
from utils.match_prediction.column_definitions import CATEGORICAL_COLUMNS

# Create model with same architecture
model = Model(
    num_categories=model_config["num_categories"],
    num_champions=model_config["num_champions"],
    embed_dim=model_config["embed_dim"],
    hidden_dims=model_config["hidden_dims"],
    dropout=0.0,  # No dropout needed for inference
)

# Load model weights and move to device
print(f"Loading PyTorch model from {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

# Enable float16 precision when possible (faster on Apple Silicon)
use_float16 = False
if use_float16 and device.type == "mps":
    print("Enabling float16 precision for faster inference")

# For authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    expected_key = os.getenv("API_KEY")
    # If API_KEY is empty, skip verification
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class APIInput(BaseModel):
    champion_ids: List[int | Literal["UNKNOWN"]]
    numerical_elo: int
    patch: Optional[str] = None


class WinratePrediction(BaseModel):
    win_probability: float


def preprocess_batch(batch_inputs):
    """
    Efficiently preprocess a batch of inputs for the model
    """
    batch_size = len(batch_inputs)

    # Process champion IDs
    all_champion_ids = []
    for input_data in batch_inputs:
        # Handle "UNKNOWN" champions properly
        ids = [id if id != "UNKNOWN" else "UNKNOWN" for id in input_data.champion_ids]
        all_champion_ids.append(ids)

    # Convert to encoded values
    encoded_champions = []
    for champion_ids in all_champion_ids:
        # Transform directly with label encoder handling "UNKNOWN"
        encoded = label_encoders["champion_ids"].transform(champion_ids)
        encoded_champions.append(encoded)

    # Create tensor on device
    champion_ids_tensor = torch.tensor(
        encoded_champions, dtype=torch.long, device=device
    )

    # Process numerical ELO
    elo_mean = numerical_stats["means"].get("numerical_elo", 0.0)
    elo_std = numerical_stats["stds"].get("numerical_elo", 1.0)

    elos = []
    for input_data in batch_inputs:
        if elo_std == 0:
            normalized_elo = 0.0
        else:
            normalized_elo = (input_data.numerical_elo - elo_mean) / elo_std
        elos.append(normalized_elo)

    elo_tensor = torch.tensor(elos, dtype=torch.float32, device=device)

    # Process patch
    patch_mean = numerical_stats["means"].get("numerical_patch", 0.0)
    patch_std = numerical_stats["stds"].get("numerical_patch", 1.0)

    patches = []
    for input_data in batch_inputs:
        if not input_data.patch:
            # Use the latest patch if none provided
            latest_patch = max(patch_mapping.keys())
            numerical_patch = int(latest_patch)
        else:
            major, minor = input_data.patch.split(".")
            raw_patch = int(major) * 50 + int(minor)
            numerical_patch = patch_mapping.get(float(raw_patch), 0)

        if patch_std == 0:
            normalized_patch = 0.0
        else:
            normalized_patch = (numerical_patch - patch_mean) / patch_std
        patches.append(normalized_patch)

    patch_tensor = torch.tensor(patches, dtype=torch.float32, device=device)

    # Return as dictionary for model input
    return {
        "champion_ids": champion_ids_tensor,
        "numerical_elo": elo_tensor,
        "numerical_patch": patch_tensor,
    }


@app.post("/predict-batch")
async def predict_batch(inputs: List[APIInput], api_key: str = Depends(verify_api_key)):
    """
    Optimized batch prediction endpoint
    """
    batch_size = len(inputs)
    start_time = time.time()

    # Preprocess inputs
    preprocess_start = time.time()
    model_inputs = preprocess_batch(inputs)
    preprocess_time = time.time() - preprocess_start

    # Run model inference
    inference_start = time.time()
    with torch.no_grad():
        # Use float16 precision if enabled and available
        if use_float16 and device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(model_inputs)
        else:
            outputs = model(model_inputs)

        # Get win predictions
        win_predictions = outputs["win_prediction"]
        win_probabilities = torch.sigmoid(win_predictions).cpu().numpy()

    inference_time = time.time() - inference_start
    total_time = time.time() - start_time

    # Return predictions
    return [
        WinratePrediction(win_probability=float(prob)) for prob in win_probabilities
    ]


if __name__ == "__main__":
    print(f"Starting M1-optimized model server on http://0.0.0.0:8000")
    uvicorn.run("serve-model-m1:app", host="0.0.0.0", port=8000, reload=False)
