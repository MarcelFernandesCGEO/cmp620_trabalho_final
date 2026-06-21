"""
Degrade a high-resolution GT (e.g. 35 cm) to the SR output resolution (~2.39 m).

Applies a Gaussian blur (sigma derived from the resolution ratio) before
downsampling to avoid aliasing, then reprojects to EPSG:3857.

The output GT_degradado is the reference used in all metric computations.

Usage:
    python degrade_gt.py \
        --input  dados/gt/gt_original/gt.tif \
        --output dados/gt/gt_degradado/gt_2.5m.tif \
        [--target_res 2.389]
"""

import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.ndimage import gaussian_filter
from pathlib import Path


# SR output resolution: infer_geotiff.py produces TARGET_RES / 4 = 9.555 / 4 ≈ 2.389 m
DEFAULT_TARGET_RES = 9.555 / 4  # 2.389 m


def degrade(input_path, output_path, target_res=DEFAULT_TARGET_RES):
    with rasterio.open(input_path) as src:
        native_res = (src.res[0] + src.res[1]) / 2
        print(f"Input resolution: {native_res:.4f} m  →  target: {target_res:.4f} m")

        # Gaussian sigma = half the downscale factor (prevents aliasing)
        scale_factor = target_res / native_res
        sigma = scale_factor / 2.0
        print(f"Downscale factor: {scale_factor:.2f}x  |  Gaussian sigma: {sigma:.2f} px")

        data = src.read()  # [C, H, W]
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        src_bounds = src.bounds

    # Blur in float to avoid uint8 clipping artifacts
    blurred = gaussian_filter(data.astype(np.float32), sigma=[0, sigma, sigma])
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    dst_crs = rasterio.crs.CRS.from_epsg(3857)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds,
        resolution=(target_res, target_res)
    )
    print(f"Output grid: {dst_width}x{dst_height} at {target_res:.4f}m")

    n_bands = blurred.shape[0]
    dst_data = np.zeros((n_bands, dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=blurred,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": dst_width,
        "height": dst_height,
        "count": n_bands,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(dst_data)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Degrade GT to SR output resolution")
    parser.add_argument("--input", required=True, help="High-res GT GeoTIFF (e.g. 35 cm)")
    parser.add_argument("--output", required=True, help="Output degraded GT GeoTIFF")
    parser.add_argument("--target_res", type=float, default=DEFAULT_TARGET_RES,
                        help=f"Target resolution in meters (default: {DEFAULT_TARGET_RES:.3f})")
    args = parser.parse_args()

    degrade(args.input, args.output, args.target_res)


if __name__ == "__main__":
    main()
