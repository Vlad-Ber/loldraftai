import os

# Get the directory of the current file (__init__.py)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to the 'machine-learning' directory
machine_learning_dir = os.path.dirname(current_file_dir)
champion_play_rates_path = os.path.join(
    machine_learning_dir, "../../packages/ui/src/lib/config/champion_play_rates.json"
)

DATA_DIR = os.path.join(machine_learning_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
