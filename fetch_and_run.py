# Usage example: python fetch_and_run.py --save-embedding --lat 41.490479 --lon -73.057288

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import rasterio
import requests
import torch
from PIL import Image
from pystac_client import Client
from pyproj import Proj, transform
from rasterio.windows import Window
from transformers import AutoImageProcessor, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def extract_date_from_path(url):
    """Extracts the acquisition date from a Sentinel-2 URL."""
    # Match YYYY/M/D and _YYYYMMDD_ pattern
    match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/.*?_(\d{8})', url)
    if match:
        return match.group(4)  # Returns YYYYMMDD
    
    # Fallback for other URL structures, like S2B_18TWL_20250708_0_L2A
    match = re.search(r'_(\d{8})_', url)
    if match:
        return match.group(1)

    return None


def find_sentinel2_image_url(lat, lon):
    """Finds a Sentinel-2 image URL for a given latitude and longitude."""
    logging.info(f"Searching for Sentinel-2 image near (lat={lat}, lon={lon})...")
    try:
        api = Client.open("https://earth-search.aws.element84.com/v1")
        search = api.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=[
                (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            query={"eo:cloud_cover": {"lt": 20}},
        )

        for item in search.items():
            from shapely.geometry import Point, shape

            point = Point(lon, lat)
            if shape(item.geometry).contains(point):
                logging.info(f"Found containing item: {item.id}")
                if "visual" in item.assets:
                    return item.assets["visual"].href
                elif "thumbnail" in item.assets:
                    return item.assets["thumbnail"].href

        logging.warning(
            "Could not find a tile that directly contains the point. Falling back to the first result."
        )
        item = next(search.items(), None)
        if item and "visual" in item.assets:
            return item.assets["visual"].href
    except Exception as e:
        logging.error(f"Error searching for image: {e}")
    return None


def download_image(url, save_path="temp_image.tif"):
    """Downloads an image from a URL."""
    logging.info(f"Downloading image from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Image saved to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image: {e}")
        return None


def crop_center(image_path, lat, lon, output_path, crop_size=2048):
    """Crops a GeoTIFF image around a central point."""
    logging.info(f"Cropping image to {crop_size}x{crop_size} around (lat={lat}, lon={lon})")
    try:
        with rasterio.open(image_path) as src:
            src_crs = "EPSG:4326"  # WGS 84
            dst_crs = src.crs
            x, y = transform(Proj(init=src_crs), Proj(init=dst_crs), lon, lat)
            row, col = src.index(x, y)
            window = Window(
                col - crop_size // 2, row - crop_size // 2, crop_size, crop_size
            )
            data = src.read((1, 2, 3), window=window)

            if data.size == 0:
                logging.error("Cropped window is empty.")
                return None

            img_data = np.moveaxis(data, 0, -1).astype(np.uint8)
            img = Image.fromarray(img_data)
            img.save(output_path, "PNG")
            logging.info(f"Cropped image saved to {output_path}")
            return output_path
    except Exception as e:
        logging.error(f"Error cropping image: {e}")
        return None


def run_model_on_image(image_path):
    """Runs the model on an image."""
    logging.info("Loading model and processor...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")

    logging.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    logging.info("Running inference...")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding, embedding.shape


def main(save_embedding, lat, lon, output_dir):
    """Main function to fetch data, run the model, and save outputs."""
    image_url = find_sentinel2_image_url(lat, lon)
    if not image_url:
        logging.error("Could not find a suitable image. Exiting.")
        return

    acquisition_date = extract_date_from_path(image_url)
    if not acquisition_date:
        logging.warning("Could not extract date from URL. Using generic filenames.")
        acquisition_date = "unknown_date"

    # Define date-based filenames
    base_name = f"{acquisition_date}_lat{lat:.4f}_lon{lon:.4f}"
    
    # Prepend output directory if provided
    if output_dir:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.join(output_dir, base_name)

    cropped_image_path = f"{base_name}.png"
    metadata_path = f"{base_name}_metadata.json"
    embedding_path = f"{base_name}_embedding.npy"

    temp_image_path = download_image(image_url)
    if not temp_image_path:
        logging.error("Failed to download image. Exiting.")
        return

    final_image_path = crop_center(temp_image_path, lat, lon, cropped_image_path)
    if not final_image_path:
        logging.error("Failed to crop image. Exiting.")
        return

    embedding, embedding_shape = run_model_on_image(final_image_path)
    logging.info(f"Embedding vector shape: {embedding_shape}")

    metadata = {
        "image_source": image_url,
        "acquisition_date": acquisition_date,
        "latitude": lat,
        "longitude": lon,
        "embedding_shape": list(embedding_shape),
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to {metadata_path}")

    if save_embedding:
        np.save(embedding_path, embedding.numpy())
        logging.info(f"Embedding saved to {embedding_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch a Sentinel-2 tile, run it through a model, and save the output."
    )
    parser.add_argument(
        "--save-embedding",
        action="store_true",
        help="If set, saves the embedding vector as a .npy file.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=41.490479,
        help="Latitude for the location.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-73.057288,
        help="Longitude for the location.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the output files.",
    )
    args = parser.parse_args()

    main(args.save_embedding, args.lat, args.lon, args.output_dir)
