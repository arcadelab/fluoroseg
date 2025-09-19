import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def download(url: str, destination_path: str | Path) -> None:
    # Convert the destination_path to a Path object if it's not already
    destination = Path(destination_path)

    # Check if the file already exists
    if not destination.exists():
        logger.info(f"File not found, downloading from {url}...")
        try:
            # Send a GET request to download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Write the content to a file
            with destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"File downloaded and saved to {destination}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}")
    else:
        logger.info(f"File already exists at {destination}.")
