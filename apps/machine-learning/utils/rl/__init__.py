import numpy as np
import requests
import os
from utils import DATA_DIR

ROLE_CHAMPIONS_PATH = os.path.join(DATA_DIR, "role_champions.json")


def fetch_blue_side_winrate_prediction(
    champion_ids: np.ndarray,
    numerical_elo: int = 2,
    numerical_patch: int = 14 * 50 + 18,
):
    """
    Fetch the blue side winrate prediction for a given set of champion IDs.
    """
    input_data = {
        # Champion IDs are blue picks from top to bot, then red picks from top to bot
        "champion_ids": champion_ids.tolist(),  # list of 10 champion IDs
        "numerical_elo": numerical_elo,
        "numerical_patch": numerical_patch,
    }

    response = requests.post("http://localhost:8000/predict", json=input_data)
    winrate_prediction = response.json()[
        "win_probability"
    ]  # blue side winrate prediction

    return winrate_prediction
