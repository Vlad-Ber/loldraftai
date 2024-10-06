import os
import pickle
import torch
import numpy as np
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from utils.model import MatchOutcomeModel
from utils.column_definitions import COLUMNS, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, ColumnType
from utils import MODEL_PATH, ENCODERS_PATH, NUMERICAL_STATS_PATH

# Load label encoders
with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

# Load numerical stats for normalization
with open(NUMERICAL_STATS_PATH, "rb") as f:
    numerical_stats = pickle.load(f)

# Define input schema
class ModelInput(BaseModel):
    region: str
    averageTier: str
    averageDivision: str
    champion_ids: List[int]
    gameVersionMajorPatch: float
    gameVersionMinorPatch: float

app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class MatchOutcomePredictor:
    def __init__(self):
        # Load the model
        self.model = MatchOutcomeModel(
            num_categories={col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS},
            num_champions=max(max(encoder.classes_) for encoder in label_encoders.values()) + 1,
            embed_dim=32,
            dropout=0.1
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.eval()

    def preprocess_input(self, input_data: ModelInput) -> Dict[str, torch.Tensor]:
        processed_input = {}
        for col, col_def in COLUMNS.items():
            if col_def.column_type == ColumnType.CATEGORICAL:
                value = getattr(input_data, col)
                if value not in label_encoders[col].classes_:
                    value = "UNKNOWN"
                processed_input[col] = torch.tensor([label_encoders[col].transform([value])[0]], dtype=torch.long)
            elif col_def.column_type == ColumnType.NUMERICAL:
                value = getattr(input_data, col)
                mean, std = numerical_stats['means'][col], numerical_stats['stds'][col]
                processed_input[col] = torch.tensor([(value - mean) / std], dtype=torch.float)
            elif col_def.column_type == ColumnType.LIST:
                processed_input[col] = torch.tensor(getattr(input_data, col), dtype=torch.long)
        return processed_input

    @app.post("/predict")
    async def predict(self, input_data: ModelInput) -> Dict[str, float]:
        processed_input = self.preprocess_input(input_data)
        with torch.no_grad():
            output = self.model(processed_input)
        win_probability = output['win_prediction'].item()
        return {"win_probability": win_probability}

deployment = MatchOutcomePredictor.bind()

if __name__ == "__main__":
    serve.run(deployment)
