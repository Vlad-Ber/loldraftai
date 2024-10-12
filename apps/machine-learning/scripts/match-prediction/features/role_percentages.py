import os
import pickle
from collections import defaultdict
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import NORMALIZED_DATA_DIR, CHAMPION_FEATURES_PATH


def calculate_champion_role_percentages():
    champion_role_counts = defaultdict(lambda: defaultdict(int))
    total_champion_counts = defaultdict(int)

    # Process both train and test directories
    for file in tqdm(
        os.listdir(NORMALIZED_DATA_DIR), desc="Processing champion role percentages"
    ):
        if file.endswith(".parquet"):
            file_path = os.path.join(NORMALIZED_DATA_DIR, file)
            table = pq.read_table(file_path)
            df = table.to_pandas()

            for _, row in df.iterrows():
                champion_ids = row["champion_ids"]
                for i, champion_id in enumerate(champion_ids):
                    role = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][i % 5]
                    champion_role_counts[champion_id][role] += 1
                    total_champion_counts[champion_id] += 1

    # Calculate percentages
    champion_role_percentages = {}
    for champion_id, role_counts in champion_role_counts.items():
        total = total_champion_counts[champion_id]
        percentages = {role: count / total for role, count in role_counts.items()}

        # Normalize percentages to [-0.5, 0.5] range
        normalized_percentages = {
            role: (percentage - 0.5) for role, percentage in percentages.items()
        }

        champion_role_percentages[champion_id] = normalized_percentages

    # Save the results
    with open(CHAMPION_FEATURES_PATH, "wb") as f:
        pickle.dump(champion_role_percentages, f)

    print(f"Champion role percentages saved to {CHAMPION_FEATURES_PATH}")


if __name__ == "__main__":
    calculate_champion_role_percentages()
