import os
import argparse
import io
import requests
from PIL import Image
import ee

# Define Rondônia Ariquemes/Jamari study region bounding box: [lon_min, lat_min, lon_max, lat_max]
DEFAULT_BBOX = [-63.0, -10.5, -62.8, -10.3]

def initialize_gee():
    """Initializes Google Earth Engine API, authenticating if necessary."""
    try:
        ee.Initialize()
        print("Successfully initialized Google Earth Engine.")
    except Exception as e:
        print("Earth Engine initialization failed. Attempting authentication...")
        try:
            ee.Authenticate()
            ee.Initialize()
            print("Successfully authenticated and initialized Google Earth Engine.")
        except Exception as auth_err:
            raise RuntimeError(
                "Could not authenticate with Earth Engine. Please run 'earthengine authenticate' manually."
            ) from auth_err

def download_gee_image(image_ee: ee.Image, region: ee.Geometry, dimensions: int, filepath: str):
    """Downloads an Earth Engine image via getThumbURL and saves it as a PNG."""
    url = image_ee.getThumbURL({
        'region': region,
        'dimensions': dimensions,
        'format': 'png'
    })
    print(f"Downloading from GEE: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img.save(filepath)
        print(f"Saved: {filepath}")
    else:
        raise RuntimeError(f"GEE download failed with status code {response.status_code}: {response.text}")

def get_s2_composite(year: int, region: ee.Geometry) -> ee.Image:
    """Creates a cloud-free median Sentinel-2 RGB composite for the given year."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    # Filter Harmonized Sentinel-2 SR collection
    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )
    
    count = s2_col.size().getInfo()
    print(f"Found {count} Sentinel-2 scenes for year {year}")
    
    if count == 0:
        # Fallback: widen cloudy threshold if no scenes found
        print("Widening search: cloud threshold -> 25%")
        s2_col = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
        )
        
    composite = s2_col.median()
    # Select RGB bands and stretch to [0, 3000] reflectance values
    rgb = composite.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    return rgb

def get_hansen_loss_mask(year1: int, year2: int, region: ee.Geometry) -> ee.Image:
    """Creates a binary forest loss mask between year1 and year2 from Hansen Global Forest Change."""
    # Using 2023 version covering 2000-2023
    hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    lossyear = hansen.select('lossyear')
    
    # Hansen years are coded as 1-23 corresponding to 2001-2023
    y1_code = year1 - 2000
    y2_code = year2 - 2000
    
    loss_mask = lossyear.gte(y1_code).And(lossyear.lte(y2_code))
    # Visualize as white on black background
    visualized = loss_mask.visualize(min=0, max=1, palette=['black', 'white'])
    return visualized

def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 and Hansen forest loss validation data using GEE.")
    parser.add_argument("--year1", type=int, default=2018, help="Start year of study (e.g. 2018)")
    parser.add_argument("--year2", type=int, default=2022, help="End year of study (e.g. 2022)")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, 
                        help="Bounding box coordinates [lon_min, lat_min, lon_max, lat_max]")
    parser.add_argument("--dimensions", type=int, default=1024, help="Dimension of output PNG image")
    parser.add_argument("--output_dir", type=str, default="data/demo", help="Output directory to save images")
    args = parser.parse_args()
    
    # Initialize GEE
    initialize_gee()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define geometry region
    region = ee.Geometry.Rectangle(args.bbox)
    
    # Download Sentinel-2 Year 1
    print(f"\n--- Processing Year A: {args.year1} ---")
    s2_y1 = get_s2_composite(args.year1, region)
    y1_path = os.path.join(args.output_dir, f"sentinel2_{args.year1}.png")
    download_gee_image(s2_y1, region, args.dimensions, y1_path)
    
    # Download Sentinel-2 Year 2
    print(f"\n--- Processing Year B: {args.year2} ---")
    s2_y2 = get_s2_composite(args.year2, region)
    y2_path = os.path.join(args.output_dir, f"sentinel2_{args.year2}.png")
    download_gee_image(s2_y2, region, args.dimensions, y2_path)
    
    # Download Hansen Loss Validation Mask
    print(f"\n--- Processing Hansen Loss Mask: {args.year1} to {args.year2} ---")
    hansen_mask = get_hansen_loss_mask(args.year1, args.year2, region)
    mask_path = os.path.join(args.output_dir, "hansen_validation_mask.png")
    download_gee_image(hansen_mask, region, args.dimensions, mask_path)
    
    print("\nAll downloads completed successfully!")

if __name__ == "__main__":
    main()
