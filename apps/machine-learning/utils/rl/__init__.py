import numpy as np
import requests


def fetch_blue_side_winrate_prediction(champion_ids: np.ndarray):
    """
    Fetch the blue side winrate prediction for a given set of champion IDs.
    """
    input_data = {
        # "region": "EUW1",
        # "averageTier": "DIAMOND",
        # "averageDivision": "II",
        # Champion IDs are blue picks from top to bot, then red picks from top to bot
        "champion_ids": champion_ids.tolist(),  # list of 10 champion IDs
        # "gameVersionMajorPatch": 14,
        # "gameVersionMinorPatch": 18,
        "numerical_elo": 2,
        "numerical_patch": 14 * 50 + 18,
    }

    response = requests.post("http://localhost:8000/predict", json=input_data)
    winrate_prediction = response.json()[
        "win_probability"
    ]  # blue side winrate prediction

    return winrate_prediction
