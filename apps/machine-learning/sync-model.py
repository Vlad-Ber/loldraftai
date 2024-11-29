import os
import argparse
from pathlib import Path
from utils.match_prediction import (
    MODEL_PATH,
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    TASK_STATS_PATH,
    MODEL_CONFIG_PATH,
    CHAMPION_FEATURES_PATH,
    PREPARED_DATA_DIR,
)

# Default settings
DEFAULT_VM_USER = "azureuser"
DEFAULT_VM_IP = "40.67.128.19"
DEFAULT_KEY_PATH = os.path.expanduser(
    "~/Documents/LeagueDraftv2/DataCollectionVM/DataCollectionVM_key.pem"
)
DEFAULT_REMOTE_BASE = "/home/azureuser/draftking-monorepo"

# Get the project root directory (2 levels up from the script)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MACHINE_LEARNING_DIR = Path(__file__).parent


def create_sync_script(
    vm_user: str = DEFAULT_VM_USER,
    vm_ip: str = DEFAULT_VM_IP,
    key_path: str = DEFAULT_KEY_PATH,
) -> str:
    # Files to sync
    files_to_sync = [
        MODEL_PATH,
        ENCODERS_PATH,
        NUMERICAL_STATS_PATH,
        TASK_STATS_PATH,
        MODEL_CONFIG_PATH,
        CHAMPION_FEATURES_PATH,
        Path(PREPARED_DATA_DIR) / "patch_mapping.pkl",
    ]

    # Common SSH options
    ssh_opts = f"-i {key_path}"

    # Create rsync commands
    commands = []
    for file_path in files_to_sync:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist")
            continue

        # Convert to Path object for easier manipulation
        file_path = Path(file_path)

        # Get the relative path from the project root
        rel_path = file_path.relative_to(PROJECT_ROOT)

        # Construct the remote path
        remote_path = f"{DEFAULT_REMOTE_BASE}/{rel_path}"
        remote_dir = str(Path(remote_path).parent)

        # Create remote directory if it doesn't exist
        commands.append(f"ssh {ssh_opts} {vm_user}@{vm_ip} 'mkdir -p {remote_dir}'")

        # Copy the file
        commands.append(f"scp {ssh_opts} {file_path} {vm_user}@{vm_ip}:{remote_path}")

    # Create server restart command
    restart_command = f"ssh {ssh_opts} {vm_user}@{vm_ip} 'cd {DEFAULT_REMOTE_BASE}/apps/machine-learning && git pull && sudo systemctl restart model-server'"

    return "\n".join(commands + [restart_command])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model sync commands")
    parser.add_argument("--vm-user", default=DEFAULT_VM_USER, help="VM username")
    parser.add_argument("--vm-ip", default=DEFAULT_VM_IP, help="VM IP address")
    parser.add_argument("--key-path", default=DEFAULT_KEY_PATH, help="Path to SSH key")

    args = parser.parse_args()

    script = create_sync_script(args.vm_user, args.vm_ip, args.key_path)
    print("\nRun these commands to sync and restart the model server:")
    print("----------------------------------------")
    print(script)
    print("----------------------------------------")
