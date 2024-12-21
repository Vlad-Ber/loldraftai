# utils/match_prediction/match_dataset.py
import os
import glob
import random
import pickle
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from typing import Callable, Optional, List

from utils.match_prediction import (
    PREPARED_DATA_DIR,
    CHAMPION_FEATURES_PATH,
    PARQUET_READER_BATCH_SIZE,
    POSITIONS,
)
from utils.match_prediction.column_definitions import COLUMNS, ColumnType
from utils.match_prediction.task_definitions import TASKS, TaskType


class MatchDataset(IterableDataset):
    def __init__(
        self,
        transform=None,
        masking_function: Optional[Callable[[], int]] = None,
        unknown_champion_id=None,
        train_or_test="train",
        small_dataset=False,
    ):
        self.data_files = sorted(
            glob.glob(
                os.path.join(
                    PREPARED_DATA_DIR, train_or_test, f"{train_or_test}*.parquet"
                )
            )
        )
        self.transform = transform
        self.small_dataset = small_dataset
        self.train_or_test = train_or_test

        # If small_dataset is True, only use 5% of the files
        if small_dataset:
            num_files = max(1, int(len(self.data_files) * 0.05))
            self.data_files = self.data_files[:num_files]

        self.total_samples = self._count_total_samples()
        self.masking_function = masking_function
        self.unknown_champion_id = unknown_champion_id

        # Shuffle the data files
        random.seed(42)  # For reproducibility
        random.shuffle(self.data_files)

    def _count_total_samples(self):
        count_path = os.path.join(PREPARED_DATA_DIR, "sample_counts.pkl")
        try:
            if os.path.exists(count_path):
                with open(count_path, "rb") as f:
                    counts = pickle.load(f)
                    count = counts.get(self.train_or_test)
                    if count is not None:
                        if self.small_dataset:
                            count = int(count * 0.05)
                        return count
        except Exception as e:
            print(f"Warning: Error reading sample counts file: {e}")

        # Fall back to counting if file doesn't exist or is invalid
        print(f"Counting samples in {self.train_or_test} dataset...")
        total = 0
        for file_path in self.data_files:
            parquet_file = pq.ParquetFile(file_path)
            total += parquet_file.metadata.num_rows

        # Try to save the count for future use
        try:
            counts = {}
            if os.path.exists(count_path):
                with open(count_path, "rb") as f:
                    counts = pickle.load(f)
            counts[self.train_or_test] = total
            with open(count_path, "wb") as f:
                pickle.dump(counts, f)
        except Exception as e:
            print(f"Warning: Could not save sample counts: {e}")

        if self.small_dataset:
            total = int(total * 0.05)
        return total

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start, iter_end = 0, len(self.data_files)
        else:
            per_worker = int(np.ceil(len(self.data_files) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_files))

        for file_path in self.data_files[iter_start:iter_end]:
            # Use context manager to ensure file is closed
            with pq.ParquetFile(file_path) as parquet_file:
                for batch in parquet_file.iter_batches(
                    batch_size=PARQUET_READER_BATCH_SIZE
                ):
                    df_chunk = batch.to_pandas()
                    df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

                    # Process the chunk and yield individual samples
                    samples = self._get_samples(df_chunk)
                    for sample in samples:
                        yield sample

    def _get_samples(self, df_chunk):
        for col, col_def in COLUMNS.items():
            if col_def.column_type == ColumnType.CATEGORICAL:
                df_chunk[col] = df_chunk[col].astype(int)
            elif col_def.column_type == ColumnType.NUMERICAL:
                df_chunk[col] = df_chunk[col].astype(float)
            elif col_def.column_type == ColumnType.LIST and col == "champion_ids":
                df_chunk[col] = df_chunk[col].apply(
                    lambda x: [int(ch_id) for ch_id in x]
                )
                if (
                    self.masking_function is not None
                    and self.unknown_champion_id is not None
                ):
                    df_chunk[col] = df_chunk[col].apply(
                        lambda x: self._mask_champions(x, self.masking_function())
                    )
                # Add champion role percentages
                df_chunk[col] = df_chunk[col].apply(
                    lambda x: torch.tensor(x, dtype=torch.long)
                )

        for task_name, task_def in TASKS.items():
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                df_chunk[task_name] = df_chunk[task_name].astype(float)
            elif task_def.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                df_chunk[task_name] = df_chunk[task_name].astype(int)
            # No need to normalize regression tasks as they are already normalized

        samples = df_chunk.to_dict("records")
        if self.transform:
            samples = [self.transform(sample) for sample in samples]

        return samples

    def _mask_champions(self, champion_list: List[int], num_to_mask: int) -> List[int]:
        """Masks a specific number of champions in the list"""
        mask_indices = np.random.choice(
            len(champion_list), size=num_to_mask, replace=False
        )
        return [
            self.unknown_champion_id if i in mask_indices else ch_id
            for i, ch_id in enumerate(champion_list)
        ]
