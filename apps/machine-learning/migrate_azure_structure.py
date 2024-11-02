import os
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

CONTAINER_NAME = "league-matches"
PROCESSED_DATA_PREFIX = "processed"
RAW_DATA_PREFIX = "raw"


def migrate_files():
    # Validate Azure connection string
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_CONNECTION_STRING environment variable is not set")

    # Initialize Azure client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Get current year/month for the new structure
    now = datetime.now()
    year = now.year
    month = f"{now.month:02d}"

    # List all files in both prefixes
    for prefix in [PROCESSED_DATA_PREFIX, RAW_DATA_PREFIX]:
        print(f"\nMigrating files in {prefix}/...")
        blobs = container_client.list_blobs(name_starts_with=prefix)

        for blob in tqdm(list(blobs)):
            # Get original filename
            filename = os.path.basename(blob.name)

            # Create new path
            new_path = f"{prefix}/{year}/{month}/{filename}"

            if new_path != blob.name:
                # Copy to new location
                source_blob = container_client.get_blob_client(blob.name)
                new_blob = container_client.get_blob_client(new_path)

                new_blob.start_copy_from_url(source_blob.url)

                # Delete old blob after successful copy
                source_blob.delete_blob()

                print(f"Moved {blob.name} -> {new_path}")


if __name__ == "__main__":
    migrate_files()
