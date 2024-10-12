import os
import glob
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import (
    RAW_DATA_DIR,
    NORMALIZED_DATA_DIR,
    NUMERICAL_STATS_PATH,
    TASK_STATS_PATH,
)
from utils.column_definitions import NUMERICAL_COLUMNS
from utils.task_definitions import TASKS, TaskType


def load_data(file_path):
    return pd.read_parquet(file_path)


def save_data(df, file_path):
    df.to_parquet(file_path, index=False)


def compute_stats(data_dir):
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

    for file_path in tqdm(
        glob.glob(os.path.join(data_dir, "*.parquet")), desc="Computing stats"
    ):
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


def normalize_data(
    input_dir, output_dir, numerical_means, numerical_stds, task_means, task_stds
):
    os.makedirs(output_dir, exist_ok=True)

    for file_path in tqdm(
        glob.glob(os.path.join(input_dir, "*.parquet")), desc="Normalizing data"
    ):
        df = load_data(file_path)

        for col in NUMERICAL_COLUMNS:
            if numerical_stds[col] != 0:
                df[col] = (df[col] - numerical_means[col]) / numerical_stds[col]
            else:
                df[col] = (
                    df[col] - numerical_means[col]
                )  # Center the data if std is zero

        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION:
                if task_stds[task] != 0:
                    df[task] = (df[task] - task_means[task]) / task_stds[task]
                else:
                    df[task] = (
                        df[task] - task_means[task]
                    )  # Center the data if std is zero

        output_file = os.path.join(output_dir, os.path.basename(file_path))
        save_data(df, output_file)


def main():
    print("Computing statistics...")
    numerical_means, numerical_stds, task_means, task_stds = compute_stats(RAW_DATA_DIR)

    print("Saving normalization parameters...")
    with open(NUMERICAL_STATS_PATH, "wb") as f:
        pickle.dump({"means": numerical_means, "stds": numerical_stds}, f)

    with open(TASK_STATS_PATH, "wb") as f:
        pickle.dump({"means": task_means, "stds": task_stds}, f)

    print("Normalizing data...")
    normalize_data(
        RAW_DATA_DIR,
        NORMALIZED_DATA_DIR,
        numerical_means,
        numerical_stds,
        task_means,
        task_stds,
    )

    print("Data normalization completed.")


if __name__ == "__main__":
    main()
