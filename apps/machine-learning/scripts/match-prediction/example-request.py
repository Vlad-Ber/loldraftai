import requests

input_data = {
    "region": "EUW1",
    "averageTier": "GRANDMASTER",
    "averageDivision": "I",
    "champion_ids": [36, 5, 61, 147, 235, 163, 427, 910, 21, 111],
    "gameVersionMajorPatch": 14,
    "gameVersionMinorPatch": 18,
}

response = requests.post("http://localhost:8000/predict", json=input_data)
print(response.json())
