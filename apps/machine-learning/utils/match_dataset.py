import os
import glob
import random
import pickle
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

from utils import NORMALIZED_DATA_DIR, CHAMPION_FEATURES_PATH, PARQUET_READER_BATCH_SIZE
from utils.column_definitions import COLUMNS, ColumnType
from utils.task_definitions import TASKS, TaskType


class MatchDataset(IterableDataset):
    def __init__(
        self,
        transform=None,
        mask_champions=0.0,
        unknown_champion_id=None,
        train_or_test="train",
    ):
        self.data_files = sorted(
            glob.glob(os.path.join(NORMALIZED_DATA_DIR, f"{train_or_test}*.parquet"))
        )
        self.transform = transform
        self.total_samples = self._count_total_samples()
        self.mask_champions = mask_champions
        self.unknown_champion_id = unknown_champion_id

        # Shuffle the data files
        random.seed(42)  # For reproducibility
        random.shuffle(self.data_files)

        # Load champion features
        with open(CHAMPION_FEATURES_PATH, "rb") as f:
            self.champion_features = pickle.load(f)

    def _count_total_samples(self):
        total = 0
        for file_path in self.data_files:
            parquet_file = pq.ParquetFile(file_path)
            total += parquet_file.metadata.num_rows
        return total

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start, iter_end = 0, len(self.data_files)
        else:
            per_worker = int(len(self.data_files) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for file_path in self.data_files[iter_start:iter_end]:
            # Use memory mapping to read the Parquet file
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_READER_BATCH_SIZE
            ):
                df_chunk = batch.to_pandas()
                df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

                # Process the entire chunk at once
                samples = self._get_samples(df_chunk)
                yield from samples

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
                if self.mask_champions > 0 and self.unknown_champion_id is not None:
                    mask = (
                        np.random.random(df_chunk[col].apply(len).sum())
                        > self.mask_champions
                    )
                    df_chunk[col] = df_chunk[col].apply(
                        lambda x: [
                            ch_id if m else self.unknown_champion_id
                            for ch_id, m in zip(x, mask[: len(x)])
                        ]
                    )
                # Add champion role percentages
                df_chunk["champion_role_percentages"] = df_chunk[col].apply(
                    lambda x: [
                        [
                            self.champion_features.get(ch_id, {}).get(role, 0)
                            for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
                        ]
                        for ch_id in x
                    ]
                )
                df_chunk["champion_role_percentages"] = df_chunk[
                    "champion_role_percentages"
                ].apply(lambda x: torch.tensor(x, dtype=torch.float))
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
