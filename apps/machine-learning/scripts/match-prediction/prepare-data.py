# scripts/match-prediction/prepare-data.py
import os
import glob
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.match_prediction import (
    RAW_DATA_DIR,
    PREPARED_DATA_DIR,
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    TASK_STATS_PATH,
)
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


def load_data(file_path):
    return pd.read_parquet(file_path)


def save_data(df, file_path):
    df.to_parquet(file_path, index=False)


def create_encoders(data_files):
    encoders = {}
    for col in CATEGORICAL_COLUMNS + ["champion_ids"]:
        print(f"Creating encoder for {col}")
        unique_values = set()
        for file_path in tqdm(data_files, desc=f"Processing {col}"):
            df = load_data(file_path)
            if col == "champion_ids":
                unique_values.update(df[col].explode().unique())
            else:
                unique_values.update(df[col].unique())

        unique_values = sorted(unique_values)
        unique_values.append("UNKNOWN")  # Add UNKNOWN for unseen categories

        encoder = LabelEncoder()
        encoder.fit(unique_values)
        encoders[col] = encoder

    return encoders


def compute_stats(data_files):
    numerical_sums = {col: 0.0 for col in NUMERICAL_COLUMNS}
    numerical_sumsq = {col: 0.0 for col in NUMERICAL_COLUMNS}
    task_sums = {
        task: 0.0
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    }
    task_sumsq = {
        task: 0.0
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    }
    total_count = 0

    for file_path in tqdm(data_files, desc="Computing stats"):
        df = load_data(file_path)
        for col in NUMERICAL_COLUMNS:
            numerical_sums[col] += df[col].sum()
            numerical_sumsq[col] += (df[col] ** 2).sum()
        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION:
                task_sums[task] += df[task].sum()
                task_sumsq[task] += (df[task] ** 2).sum()
        total_count += len(df)

    numerical_means = {
        col: numerical_sums[col] / total_count for col in NUMERICAL_COLUMNS
    }
    numerical_stds = {
        col: np.sqrt((numerical_sumsq[col] / total_count) - (numerical_means[col] ** 2))
        for col in NUMERICAL_COLUMNS
    }
    task_means = {task: task_sums[task] / total_count for task in task_sums}
    task_stds = {
        task: np.sqrt((task_sumsq[task] / total_count) - (task_means[task] ** 2))
        for task in task_sums
    }

    return numerical_means, numerical_stds, task_means, task_stds


def prepare_data(
    input_files,
    output_dir,
    encoders,
    numerical_means,
    numerical_stds,
    task_means,
    task_stds,
):
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    for file_index, file_path in enumerate(tqdm(input_files, desc="Preparing data")):
        df = load_data(file_path)

        # Encode categorical columns
        for col in CATEGORICAL_COLUMNS:
            df[col] = encoders[col].transform(df[col])

        # Encode champion_ids
        df["champion_ids"] = df["champion_ids"].apply(
            lambda x: encoders["champion_ids"].transform(x)
        )

        # Normalize numerical columns
        for col in NUMERICAL_COLUMNS:
            if numerical_stds[col] != 0:
                df[col] = (df[col] - numerical_means[col]) / numerical_stds[col]
            else:
                df[col] = (
                    df[col] - numerical_means[col]
                )  # Center the data if std is zero

        # Normalize regression tasks
        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION:
                if task_stds[task] != 0:
                    df[task] = (df[task] - task_means[task]) / task_stds[task]
                else:
                    df[task] = (
                        df[task] - task_means[task]
                    )  # Center the data if std is zero

        # Split into train and test
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        # Save prepared data
        save_data(
            df_train, os.path.join(output_dir, "train", f"train_{file_index}.parquet")
        )
        save_data(
            df_test, os.path.join(output_dir, "test", f"test_{file_index}.parquet")
        )

        # Clear memory
        del df, df_train, df_test


def main():
    parser = argparse.ArgumentParser(description="Prepare data for machine learning")
    parser.add_argument(
        "--input-dir", default=RAW_DATA_DIR, help="Input directory containing raw data"
    )
    parser.add_argument(
        "--output-dir",
        default=PREPARED_DATA_DIR,
        help="Output directory for prepared data",
    )
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))

    print("Creating encoders...")
    encoders = create_encoders(input_files)

    print("Computing statistics...")
    numerical_means, numerical_stds, task_means, task_stds = compute_stats(input_files)

    print("Saving encoders and statistics...")
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    with open(NUMERICAL_STATS_PATH, "wb") as f:
        pickle.dump({"means": numerical_means, "stds": numerical_stds}, f)
    with open(TASK_STATS_PATH, "wb") as f:
        pickle.dump({"means": task_means, "stds": task_stds}, f)

    print("Preparing data...")
    prepare_data(
        input_files,
        args.output_dir,
        encoders,
        numerical_means,
        numerical_stds,
        task_means,
        task_stds,
    )

    print("Data preparation completed.")


if __name__ == "__main__":
    main()
