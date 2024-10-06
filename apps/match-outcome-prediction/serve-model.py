import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import torch
import uvicorn
import pickle
from utils.model import MatchOutcomeModel
from utils.column_definitions import COLUMNS, ColumnType
from utils import (
    MODEL_PATH,
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    MODEL_CONFIG_PATH,
    get_best_device,
)


# Define input schema
class ModelInput(BaseModel):
    region: str
    averageTier: str
    averageDivision: str
    champion_ids: List[int]
    gameVersionMajorPatch: float
    gameVersionMinorPatch: float


app = FastAPI()

# Load label encoders
with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

# Load numerical stats for normalization
with open(NUMERICAL_STATS_PATH, "rb") as f:
    numerical_stats = pickle.load(f)

with open(MODEL_CONFIG_PATH, "rb") as f:
    model_params = pickle.load(f)

# Load the model
device = get_best_device()
print(f"Using device: {device}")

model = MatchOutcomeModel(
    num_categories=model_params["num_categories"],
    num_champions=model_params["num_champions"],
    embed_dim=model_params["embed_dim"],
    dropout=model_params["dropout"],
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Create an asyncio.Queue to hold incoming requests
request_queue = asyncio.Queue()

# Max batch size and batch wait time
MAX_BATCH_SIZE = 32
BATCH_WAIT_TIME = 0.05  # in seconds


def preprocess_input(input_data: ModelInput) -> Dict[str, torch.Tensor]:
    processed_input = {}
    for col, col_def in COLUMNS.items():
        if col_def.column_type == ColumnType.CATEGORICAL:
            value = getattr(input_data, col)
            if value not in label_encoders[col].classes_:
                value = "UNKNOWN"
            processed_input[col] = torch.tensor(
                [label_encoders[col].transform([value])[0]],
                dtype=torch.long,
                device=device,
            )
        elif col_def.column_type == ColumnType.NUMERICAL:
            value = getattr(input_data, col)
            mean = numerical_stats["means"].get(col, 0.0)
            std = numerical_stats["stds"].get(col, 1.0)
            if std == 0:
                normalized_value = 0.0  # Avoid division by zero
            else:
                normalized_value = (value - mean) / std
            processed_input[col] = torch.tensor(
                [normalized_value], dtype=torch.float, device=device
            )
        elif col_def.column_type == ColumnType.LIST:
            processed_input[col] = torch.tensor(
                [getattr(input_data, col)], dtype=torch.long, device=device
            )
    return processed_input


@app.post("/predict")
async def predict(input_data: ModelInput):
    # Create a future to hold the response
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    # Put the request into the queue
    await request_queue.put({"input": input_data, "future": future})
    # Wait for the result
    result = await future
    return result


# Define the background task
async def model_inference_worker():
    while True:
        requests = []
        # Wait until there is at least one item in the queue
        request = await request_queue.get()
        requests.append(request)
        start_time = asyncio.get_event_loop().time()
        # Wait for more items in the queue up to MAX_BATCH_SIZE or BATCH_WAIT_TIME
        while len(requests) < MAX_BATCH_SIZE:
            try:
                # Wait for the remaining time
                timeout = BATCH_WAIT_TIME - (
                    asyncio.get_event_loop().time() - start_time
                )
                if timeout <= 0:
                    break
                request = await asyncio.wait_for(request_queue.get(), timeout)
                requests.append(request)
            except asyncio.TimeoutError:
                break
        # Process the batch
        inputs = [r["input"] for r in requests]
        processed_inputs = [preprocess_input(i) for i in inputs]
        batched_input = {
            key: torch.cat([inp[key] for inp in processed_inputs], dim=0)
            for key in processed_inputs[0].keys()
        }

        with torch.no_grad():
            output = model(batched_input)

        win_probabilities = output["win_prediction"].cpu().numpy().tolist()
        # Set the results for each request
        for i, r in enumerate(requests):
            r["future"].set_result({"win_probability": win_probabilities[i]})


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_inference_worker())


# For running with uvicorn
if __name__ == "__main__":
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000)
