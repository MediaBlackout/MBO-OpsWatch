# Project: Sentinel-2 Tile Fetch and AI Embedding

This project contains a Python script (`fetch_and_run.py`) designed to fetch a satellite image for a specific geographic location, process it, and run it through a computer vision model to generate a feature embedding.

## Core Functionality

1.  **Data Source**: The script uses the public [STAC (SpatioTemporal Asset Catalog)](https://stacspec.org/) API for [Sentinel-2 data on AWS](https://registry.opendata.aws/sentinel-2/). It does not require any private accounts or API keys.
2.  **Location Input**: Users can specify a latitude and longitude via command-line arguments (`--lat` and `--lon`). The default is set to a location in Naugatuck, CT.
3.  **Image Fetching**: It searches for a recent, low-cloud-cover Sentinel-2 tile that contains the specified coordinates.
4.  **Image Cropping**: Instead of using the entire satellite tile (which can be very large), the script downloads the high-resolution GeoTIFF and crops a 2048x2048 pixel area centered exactly on the requested coordinates. This provides a detailed, focused image. The cropped image is saved as `cropped_image.png`.
5.  **AI Model Inference**: The cropped image is then run through the `facebook/dinov2-base` model, a powerful, publicly available vision model from Hugging Face.
6.  **Output**:
    `output_metadata.json`: A JSON file with details about the image source, coordinates, and embedding shape.
    *   `cropped_image.png`: The high-resolution, cropped image of the location.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the script**:
    ```bash
    # Run with default location (Naugatuck, CT) and save the embedding
    python fetch_and_run.py --save-embedding

    # Run for a different location
    python fetch_and_run.py --lat 40.7128 --lon -74.0060
    ```

## Key Libraries

-   `pystac-client`: To search for Sentinel-2 data.
-   `rasterio`, `pyproj`, `shapely`: For geospatial calculations and cropping the GeoTIFF image.
-   `Pillow`: For image manipulation.
-   `transformers`, `torch`: For running the AI model.
-   `numpy`: For handling the embedding vector.
-   `boto3`: For AWS integration (credentials managed via `.env`).
