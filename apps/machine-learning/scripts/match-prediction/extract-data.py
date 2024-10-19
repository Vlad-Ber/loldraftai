# scripts/match-prediction/extract-data.py
import os
import argparse
import datetime
import shutil
import pandas as pd
from tqdm import tqdm
import glob
from utils import DATA_DIR
from utils.match_prediction import (
    RAW_DATA_DIR,
    DATA_EXTRACTION_BATCH_SIZE,
)
from utils.match_prediction.database import Match, get_session
from utils.match_prediction.column_definitions import extract_raw_features
from utils.match_prediction.task_definitions import TASKS

LAST_EXTRACTION_TIMESTAMP_FILE = os.path.join(
    DATA_DIR, "last_extract_data_timestamp.txt"
)


def get_last_batch_number():
    existing_files = glob.glob(os.path.join(RAW_DATA_DIR, "raw_data_batch_*.parquet"))
    if not existing_files:
        return -1
    batch_numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_files]
    return max(batch_numbers)


def get_last_extraction_timestamp():
    if os.path.exists(LAST_EXTRACTION_TIMESTAMP_FILE):
        with open(LAST_EXTRACTION_TIMESTAMP_FILE, "r") as f:
            return datetime.datetime.fromisoformat(f.read().strip())
    return None


def save_extraction_timestamp(timestamp):
    with open(LAST_EXTRACTION_TIMESTAMP_FILE, "w") as f:
        f.write(timestamp.isoformat())


def batch_query(
    session, batch_size=DATA_EXTRACTION_BATCH_SIZE, continue_extraction=False
):
    """
    Generator that yields batches of matches from the database.
    """
    last_id = None
    last_timestamp = get_last_extraction_timestamp() if continue_extraction else None

    while True:
        query = session.query(Match).filter(
            Match.processed == True, Match.processingErrored == False
        )
        if last_id:
            query = query.filter(Match.id > last_id)
        if last_timestamp:
            query = query.filter(Match.updatedAt > last_timestamp)
        query = query.order_by(Match.id).limit(batch_size)
        matches = query.all()
        if not matches:
            break
        last_id = matches[-1].id
        yield matches

    if continue_extraction:
        save_extraction_timestamp(datetime.datetime.now(datetime.UTC))


def extract_raw_data(match: Match):
    features = extract_raw_features(match)

    if len(features["champion_ids"]) != 10:
        return None  # Skip this sample if it doesn't meet expectations

    # Extract labels for all tasks
    for task_name, task_def in TASKS.items():
        task_label = task_def.extractor(match)
        if task_label is None:
            return None  # Skip sample if any label is missing
        features[task_name] = task_label

    return features


def extract_and_save_raw_batches(continue_extraction=True):
    """
    Extract and save batches of raw data from the database.
    """
    session = get_session()

    if not continue_extraction:
        print("Starting from scratch, cleaning up previous data")
        # Clean up previous data only if starting from scratch
        shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        batch_num = 0
    else:
        # Find the last batch number if continuing
        batch_num = get_last_batch_number() + 1
        print(f"Continuing from previous extraction with batch {batch_num}")

    for matches in tqdm(
        batch_query(session, continue_extraction=continue_extraction),
        desc="Processing and saving batches",
    ):
        data = []
        for match in matches:
            # Extract raw features
            features = extract_raw_data(match)
            if features:
                data.append(features)

        if data:
            df = pd.DataFrame(data)

            # Save to Parquet
            file_name = f"raw_data_batch_{batch_num:05d}.parquet"
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            df.to_parquet(file_path, index=False)
            batch_num += 1
        else:
            print(f"Skipping batch, no valid samples")

    session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw data from the database and save to Parquet files"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all data, not just new data since last extraction",
    )
    args = parser.parse_args()

    continue_extraction = not args.all

    extract_and_save_raw_batches(continue_extraction)
    print("Raw data extraction and saving completed.")


if __name__ == "__main__":
    main()
