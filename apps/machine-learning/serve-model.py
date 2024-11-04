import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
import uvicorn
import pickle
from utils.match_prediction.model import MatchOutcomeModel, SimpleMatchModel
from utils.match_prediction.column_definitions import COLUMNS, ColumnType
from utils.match_prediction import (
    MODEL_PATH,
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    MODEL_CONFIG_PATH,
    CHAMPION_FEATURES_PATH,
    POSITIONS,
    get_best_device,
)


class APIInput(BaseModel):
    champion_ids: List[int]
    numerical_elo: int
    numerical_patch: int


class ModelInput(APIInput):
    champion_role_percentages: List[List[float]]


def api_input_to_model_input(api_input: APIInput) -> ModelInput:
    try:
        # Encode champion IDs
        encoded_champion_ids = label_encoders["champion_ids"].transform(
            api_input.champion_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid champion IDs")

    champion_role_percentages = [
        [champion_features.get(ch_id, {}).get(role, 0) for role in POSITIONS]
        for ch_id in api_input.champion_ids  # Note: We still use original IDs for champion features
    ]
    # Create a dictionary from the API input, excluding 'champion_ids'
    input_dict = api_input.model_dump(exclude={"champion_ids"})

    return ModelInput(
        **input_dict,
        champion_ids=encoded_champion_ids.tolist(),  # Use encoded champion IDs
        champion_role_percentages=champion_role_percentages,
    )


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

model = SimpleMatchModel(
    num_categories=model_params["num_categories"],
    num_champions=model_params["num_champions"],
    embed_dim=model_params["embed_dim"],
    dropout=model_params["dropout"],
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

with open(CHAMPION_FEATURES_PATH, "rb") as f:
    champion_features = pickle.load(f)


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
            if col == "champion_ids":
                processed_input[col] = torch.tensor(
                    [getattr(input_data, col)], dtype=torch.long, device=device
                )
            else:
                processed_input[col] = torch.tensor(
                    [getattr(input_data, col)], dtype=torch.float, device=device
                )

    return processed_input


@app.post("/predict")
async def predict(api_input: APIInput):
    model_input = api_input_to_model_input(api_input)

    # Create a future to hold the response
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    # Put the request into the queue
    await request_queue.put({"input": model_input, "future": future})
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

        win_probabilities = (
            torch.sigmoid(output["win_prediction"]).cpu().numpy().tolist()
        )
        # Set the results for each request
        for i, r in enumerate(requests):
            r["future"].set_result({"win_probability": win_probabilities[i]})


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_inference_worker())


# For running with uvicorn
if __name__ == "__main__":
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000)
