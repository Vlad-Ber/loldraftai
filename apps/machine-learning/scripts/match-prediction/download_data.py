import os
import argparse
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from dotenv import load_dotenv
from utils import DATA_DIR

load_dotenv()

CONTAINER_NAME = "league-matches"
PROCESSED_DATA_PREFIX = "processed"


def download_new_parquet_files(output_dir: str) -> None:
    """
    Download new parquet files from Azure that don't exist locally.

    Args:
        output_dir: Directory where to save the downloaded files
    """
    # Validate Azure connection string
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_CONNECTION_STRING environment variable is not set")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Azure client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Get list of existing local files
    local_files = set(os.listdir(output_dir))

    # List all processed files in Azure
    azure_blobs = container_client.list_blobs(name_starts_with=PROCESSED_DATA_PREFIX)
    azure_files = [blob for blob in azure_blobs]

    # Filter files that need to be downloaded
    files_to_download = [
        blob for blob in azure_files if os.path.basename(blob.name) not in local_files
    ]

    print(f"Files already downloaded: {len(azure_files) - len(files_to_download)}")

    if not files_to_download:
        print("No new files to download")
        return

    # Download new files with progress bar
    for blob in tqdm(files_to_download, desc="Downloading files"):
        file_name = os.path.basename(blob.name)
        output_path = os.path.join(output_dir, file_name)

        blob_client = container_client.get_blob_client(blob.name)
        with open(output_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

    print(f"Downloaded {len(files_to_download)} new files")


def main():
    parser = argparse.ArgumentParser(
        description="Download new parquet files from Azure"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DATA_DIR, "raw_azure"),
        help="Output directory for downloaded files (default: data/raw_azure)",
    )
    args = parser.parse_args()

    download_new_parquet_files(args.output_dir)


if __name__ == "__main__":
    main()
