# scripts/match-prediction/extract-data.py
import os
import pickle
import argparse
import enum
import datetime
import shutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sqlalchemy import distinct
from tqdm import tqdm
from utils import (
    RAW_DATA_DIR,
    DATA_DIR,
    ENCODERS_PATH,
    DATA_EXTRACTION_BATCH_SIZE,
)
from utils.database import Match, get_session
from utils.column_definitions import (
    COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    extract_raw_features,
    ColumnType,
)
from utils.task_definitions import TASKS

LAST_EXTRACTION_TIMESTAMP_FILE = os.path.join(DATA_DIR, "last_extract_data_timestamp.txt")

def get_last_extraction_timestamp():
    if os.path.exists(LAST_EXTRACTION_TIMESTAMP_FILE):
        with open(LAST_EXTRACTION_TIMESTAMP_FILE, 'r') as f:
            return datetime.fromisoformat(f.read().strip())
    return None

def save_extraction_timestamp(timestamp):
    with open(LAST_EXTRACTION_TIMESTAMP_FILE, 'w') as f:
        f.write(timestamp.isoformat())


TEST_SIZE = 0.2  # 20% for testing


def batch_query(session, batch_size=DATA_EXTRACTION_BATCH_SIZE, continue_extraction=False):
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
        save_extraction_timestamp(datetime.utcnow())


def create_label_encoders(session):
    """
    Create and fit label encoders for all categorical columns.
    """
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        print(f"Creating label encoder for {col}")
        query = session.query(distinct(getattr(Match, col))).filter(
            Match.processed == True,
            Match.processingErrored == False,
            getattr(Match, col) != None,  # Exclude NULL values
        )
        unique_values = [
            value.value if isinstance(value, enum.Enum) else value
            for (value,) in query.all()
            if value is not None
        ]
        unique_values = sorted(
            set(unique_values)
        )  # Ensure uniqueness and consistent ordering

        # Add a special 'UNKNOWN' value to handle unseen categories
        unique_values.append("UNKNOWN")

        print(f"Unique values for {col}: {unique_values}")

        encoder = LabelEncoder()
        encoder.fit(unique_values)
        label_encoders[col] = encoder

    return label_encoders


def extract_and_save_batches(continue_extraction=True):
    """
    Extract and save batches of data from the database.
    """
    session = get_session()

    # Clean up previous data
    shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Create and fit label encoders
    label_encoders = create_label_encoders(session)

    # Save label encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"Saved label encoders to {ENCODERS_PATH}")

    # Process and save data batches
    batch_num = 0
    for matches in tqdm(batch_query(session, continue_extraction), desc="Processing and saving batches"):
        data = []
        for match in matches:
            # Extract features
            features = extract_features(match, label_encoders)
            if features:
                data.append(features)
        if (
            data and len(data) >= DATA_EXTRACTION_BATCH_SIZE
        ):  # skip last batch, if not full
            df = pd.DataFrame(data)
            # Shuffle the data before splitting
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Split into train/test
            df_train, df_test = train_test_split(
                df, test_size=TEST_SIZE, random_state=42
            )

            # Save to Parquet
            train_file = os.path.join(RAW_DATA_DIR, f"train_{batch_num}.parquet")
            test_file = os.path.join(RAW_DATA_DIR, f"test_{batch_num}.parquet")
            df_train.to_parquet(train_file, index=False)
            df_test.to_parquet(test_file, index=False)
            batch_num += 1
        if len(data) < DATA_EXTRACTION_BATCH_SIZE:
            print(f"Skipping batch, only {len(data)} samples")

    session.close()


# TODO: verify if encoders make sense, and should they not be in the normalisation step? YES THEY SHOULD
def extract_features(match: Match, label_encoders: dict):
    features = extract_raw_features(match)

    if len(features["champion_ids"]) != 10:
        return None  # Skip this sample if it doesn't meet expectations

    # Apply label encoding for categorical features
    for col, def_ in COLUMNS.items():
        if def_.column_type == ColumnType.CATEGORICAL:
            features[col] = label_encoders[col].transform([features[col]])[0]

    for col in NUMERICAL_COLUMNS:
        if features[col] == -1:
            return None  # Skip samples with missing numerical data

    # Extract labels for all tasks
    for task_name, task_def in TASKS.items():
        task_label = task_def.extractor(match)
        if task_label is None:
            return None  # Skip sample if any label is missing
        features[task_name] = task_label

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from the database and save to Parquet files"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all data, not just new data since last extraction",
    )
    args = parser.parse_args()

    continue_extraction = not args.all

    extract_and_save_batches(continue_extraction)
    print("Data extraction and saving completed.")


if __name__ == "__main__":
    main()
