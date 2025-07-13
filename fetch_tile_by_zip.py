# Usage: python fetch_tile_by_zip.py

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta

import requests
from pystac_client import Client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)





def find_sentinel2_tile(lat, lon):
    """Finds the best Sentinel-2 tile for a given latitude and longitude."""
    logging.info(f"Searching for Sentinel-2 tile near (lat={lat}, lon={lon})...")
    try:
        api = Client.open("https://earth-search.aws.element84.com/v1")
        search = api.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=[
                (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            query={"eo:cloud_cover": {"lt": 20}},  # Relaxed cloud cover
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        )

        item = next(search.items(), None)
        if item:
            logging.info(f"Found best item: {item.id} with {item.properties['eo:cloud_cover']}% cloud cover.")
            return item
        
        logging.warning("No tile found with the primary filter. Retrying with a broader search...")
        search = api.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=[
                (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ"), # Go back a full year
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
        )
        item = next(search.items(), None)
        if item:
            logging.info(f"Found fallback item: {item.id}")
            return item

    except Exception as e:
        logging.error(f"Error searching for tile: {e}")
    return None


def download_tile(url, save_path):
    """Downloads a tile from a URL."""
    logging.info(f"Downloading tile from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192 * 10): # Larger chunk for bigger files
                f.write(chunk)
        logging.info(f"Tile saved to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading tile: {e}")
        return None


def main(zip_code, lat, lon):
    """Main function to fetch a tile for a given ZIP code."""
    output_dir = f"{zip_code}_tile"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output will be saved to '{output_dir}/'")

    item = find_sentinel2_tile(lat, lon)
    if not item:
        logging.error("Could not find a suitable tile. Exiting.")
        return

    tile_url = item.assets["visual"].href
    
    # Extract info directly from the item ID
    # Example ID: S2C_18TYM_20250620_0_L2A
    parts = item.id.split('_')
    date = parts[2]
    tile_id = parts[1]

    filename = f"{date}_{tile_id}.tif"
    output_path = os.path.join(output_dir, filename)

    download_tile(tile_url, output_path)


if __name__ == "__main__":
    # Coordinates for 06066 (Vernon, CT)
    ZIP_CODE = "06066"
    LATITUDE = 41.836
    LONGITUDE = -72.460
    
    main(ZIP_CODE, LATITUDE, LONGITUDE)
