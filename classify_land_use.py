# Usage:
# CLI: python classify_land_use.py --zip 06770
# Python: from classify_land_use import predict_land_use; result = predict_land_use(zip_code="06770")

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import rasterio
import requests
import torch
from PIL import Image
from geopy.geocoders import Nominatim
from pystac_client import Client
from pyproj import Proj, transform
from rasterio.windows import Window
import joblib
from sklearn.svm import SVC # Placeholder for classifier
from transformers import AutoImageProcessor, AutoModel

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
# Suppress less important warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


# --- Core Functions ---

def get_coords_from_zip(zip_code):
    """Converts a ZIP code to latitude and longitude."""
    logging.info(f"Geocoding ZIP code: {zip_code}...")
    try:
        geolocator = Nominatim(user_agent="land_use_classifier")
        location = geolocator.geocode(f"{zip_code}, USA")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        logging.error(f"Error geocoding ZIP code {zip_code}: {e}")
    return None, None

def find_sentinel2_tile(lat, lon):
    """Finds the best Sentinel-2 tile for a given location."""
    logging.info(f"Searching for Sentinel-2 tile near (lat={lat}, lon={lon})...")
    try:
        api = Client.open("https://earth-search.aws.element84.com/v1")
        search = api.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=[
                (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            query={"eo:cloud_cover": {"lt": 25}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        )
        item = next(search.items(), None)
        if item:
            logging.info(f"Found item: {item.id} with {item.properties.get('eo:cloud_cover', 'N/A')}% cloud cover.")
            return item
    except Exception as e:
        logging.error(f"Error searching for tile: {e}")
    return None

def download_and_crop(item, lat, lon, output_path, crop_size=1024):
    """Downloads and crops a GeoTIFF centered on the given coordinates."""
    tile_url = item.assets["visual"].href
    temp_image_path = "temp_tile.tif"

    logging.info(f"Downloading tile from {tile_url}...")
    try:
        response = requests.get(tile_url, stream=True)
        response.raise_for_status()
        with open(temp_image_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192 * 10):
                f.write(chunk)

        logging.info(f"Cropping image to {crop_size}x{crop_size}...")
        with rasterio.open(temp_image_path) as src:
            src_crs = "EPSG:4326"
            dst_crs = src.crs
            x, y = transform(Proj(init=src_crs), Proj(init=dst_crs), lon, lat, always_xy=True)
            row, col = src.index(x, y)
            window = Window(col - crop_size // 2, row - crop_size // 2, crop_size, crop_size)
            data = src.read((1, 2, 3), window=window)

            if data.size == 0:
                logging.error("Cropped window is empty.")
                return None

            img_data = np.moveaxis(data, 0, -1).astype(np.uint8)
            Image.fromarray(img_data).save(output_path, "PNG")
            logging.info(f"Cropped image saved to {output_path}")
            return output_path
    except Exception as e:
        logging.error(f"Error in download/crop process: {e}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    return None

def get_embedding(image_path):
    """Generates a vector embedding for an image."""
    logging.info("Generating embedding...")
    try:
        # Using a public, CPU-friendly model
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            # Using the CLS token for a global representation
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
    return None

def classify_embedding(embedding):
    """
    Classifies the embedding into a land use category using the trained model.
    """
    logging.info("Classifying land use...")
    try:
        classifier = joblib.load("land_use_classifier.pkl")
        prediction = classifier.predict(embedding)
        confidence = classifier.predict_proba(embedding).max()
        logging.info(f"Predicted: {prediction[0]} (Confidence: {confidence:.2f})")
        return prediction[0], confidence
    except FileNotFoundError:
        logging.error("Classifier model not found. Please run train_classifier.py first.")
        return "error", 0.0
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return "error", 0.0

def predict_land_use(lat=None, lon=None, zip_code=None):
    """
    Main prediction pipeline for use as a callable function.
    """
    if zip_code and not (lat and lon):
        lat, lon = get_coords_from_zip(zip_code)
        if not lat:
            return {"error": f"Could not geocode ZIP code {zip_code}"}
    elif not (lat and lon):
        return {"error": "Latitude/longitude or a ZIP code must be provided."}

    item = find_sentinel2_tile(lat, lon)
    if not item:
        return {"error": "Could not find a suitable satellite tile."}

    # Define filenames
    date = item.datetime.strftime('%Y%m%d')
    base_name = f"{date}_lat{lat:.4f}_lon{lon:.4f}"
    image_path = f"{base_name}.png"
    
    final_image_path = download_and_crop(item, lat, lon, image_path)
    if not final_image_path:
        return {"error": "Failed to download or crop the image."}

    embedding = get_embedding(final_image_path)
    if embedding is None:
        return {"error": "Failed to generate embedding."}

    prediction, confidence = classify_embedding(embedding)

    # Prepare output
    result = {
        "zip_code": zip_code,
        "latitude": lat,
        "longitude": lon,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "embedding_shape": list(embedding.shape),
        "tile_url": item.assets["visual"].href,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save outputs
    np.save(f"{base_name}_embedding.npy", embedding)
    with open(f"{base_name}_metadata.json", "w") as f:
        json.dump(result, f, indent=4)
    
    logging.info(f"Successfully processed location. Results saved to '{base_name}_metadata.json'")
    return result

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify land use from satellite imagery for a given location."
    )
    parser.add_argument("--zip", help="ZIP code of the target location (e.g., 06770).")
    parser.add_argument("--lat", type=float, help="Latitude of the target location.")
    parser.add_argument("--lon", type=float, help="Longitude of the target location.")
    
    args = parser.parse_args()

    if not args.zip and not (args.lat and args.lon):
        # Default to Naugatuck, CT if no arguments are provided
        logging.warning("No location provided. Defaulting to Naugatuck, CT (06770).")
        args.zip = "06770"

    result = predict_land_use(lat=args.lat, lon=args.lon, zip_code=args.zip)
    
    print("\n--- Classification Result ---")
    print(json.dumps(result, indent=4))
    print("---------------------------\n")
