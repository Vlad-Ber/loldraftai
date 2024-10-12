# utils/__init__.py
import os
import torch


def get_best_device():
    """
    Get the best device for training the model.(cuda or mps if possible)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


DATABASE_URL = os.getenv("DATABASE_URL")

# Get the directory of the current file (__init__.py)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the 'machine-learning' directory
machine_learning_dir = os.path.dirname(current_file_dir)

DATA_DIR = os.path.join(machine_learning_dir, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, "normalized_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "match_outcome_model.pth")
MODEL_CONFIG_PATH = os.path.join(DATA_DIR, "model_config.pkl")
NUMERICAL_STATS_PATH = os.path.join(DATA_DIR, "numerical_feature_stats.pkl")
TASK_STATS_PATH = os.path.join(DATA_DIR, "task_stats.pkl")

CHAMPION_FEATURES_PATH = os.path.join(DATA_DIR, "champion_features.pkl")

DEVICE = get_best_device()

# Batch sizes
if DEVICE.type == "mps":
    TRAIN_BATCH_SIZE = 2048  # 2048 #512  # Used during training
else:
    # cuda/runpod config
    TRAIN_BATCH_SIZE = 4096  # Used during training

# Data batch size could probably be higher
DATA_EXTRACTION_BATCH_SIZE = 512 * 4  # Used during data extraction from the database
PARQUET_READER_BATCH_SIZE = 512 * 4  # Used when reading data from Parquet files
