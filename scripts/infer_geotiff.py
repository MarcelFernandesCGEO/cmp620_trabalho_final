"""
Super-Resolution Inference with GeoTIFF output.

Takes multiple GeoTIFFs (same region, different dates/sensors),
runs the Satlas ESRGAN model, and produces a georeferenced GeoTIFF output.
Supports Sentinel-2, PlanetScope and CBERS imagery (uses only RGB bands).

Supports optional tile overlap + weighted blending to eliminate tile-edge artifacts.

Usage:
    python infer_geotiff.py \
        --images /path/to/images/*.tif \
        --weights /path/to/esrgan_8S2.pth \
        --output /path/to/output_sr.tif \
        --overlap 0.5 --blend linear

Environment:
    SATLAS_REPO: path to allenai/satlas-super-resolution (default: auto-detected)
"""

import argparse
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from pathlib import Path

SATLAS_REPO = Path(os.environ.get(
    "SATLAS_REPO",
    str(Path.home() / "Desktop" / "allenai" / "satlas-super-resolution")
))
sys.path.insert(0, str(SATLAS_REPO))

import torch
from ssr.utils.model_utils import build_network


TARGET_RES = 9.555  # meters, as per S2-NAIP dataset
TILE_SIZE = 32
SCALE = 4
OUTPUT_TILE_SIZE = TILE_SIZE * SCALE  # 128


def load_and_reproject(paths, target_res=TARGET_RES):
    """Load GeoTIFFs and reproject to EPSG:3857 at target resolution.
    If target_res == 0, uses native resolution from first image."""
    images = []

    if target_res == 0:
        with rasterio.open(paths[0]) as src:
            native_res = src.res
            target_res = (native_res[0] + native_res[1]) / 2
            print(f"Using native resolution: {target_res:.3f}m")

    all_bounds = []
    for p in paths:
        with rasterio.open(p) as src:
            if src.crs.to_epsg() != 3857:
                dst_crs = rasterio.crs.CRS.from_epsg(3857)
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds,
                    resolution=(target_res, target_res)
                )
                left = transform.c
                top = transform.f
                right = left + width * target_res
                bottom = top - height * target_res
                all_bounds.append((left, bottom, right, top))
            else:
                all_bounds.append(tuple(src.bounds))

    left = min(b[0] for b in all_bounds)
    bottom = min(b[1] for b in all_bounds)
    right = max(b[2] for b in all_bounds)
    top = max(b[3] for b in all_bounds)

    width = int(np.ceil((right - left) / target_res))
    height = int(np.ceil((top - bottom) / target_res))
    width = int(np.ceil(width / TILE_SIZE) * TILE_SIZE)
    height = int(np.ceil(height / TILE_SIZE) * TILE_SIZE)

    right = left + width * target_res
    bottom = top - height * target_res

    dst_transform = from_bounds(left, bottom, right, top, width, height)
    dst_crs = rasterio.crs.CRS.from_epsg(3857)

    print(f"Output grid: {width}x{height} pixels ({width // TILE_SIZE}x{height // TILE_SIZE} tiles)")

    for p in paths:
        with rasterio.open(p) as src:
            dst_data = np.zeros((3, height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            images.append(dst_data)

    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 3,
        "crs": dst_crs,
        "transform": dst_transform,
        "actual_res": target_res,
    }

    return images, meta


def create_blend_mask(height, width, blend_type='linear'):
    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    xx, yy = np.meshgrid(x, y)
    dist_x = np.minimum(xx, 1 - xx) * 2
    dist_y = np.minimum(yy, 1 - yy) * 2
    dist = np.minimum(dist_x, dist_y)

    if blend_type == 'gaussian':
        mask = np.exp(-(1 - dist)**2 / (2 * 0.3**2))
    elif blend_type == 'cosine':
        mask = 0.5 * (1 - np.cos(np.pi * dist))
    else:  # linear
        mask = dist

    return np.clip(mask, 0, 1).astype(np.float32)


def tile_images(images, tile_size=TILE_SIZE):
    _, h, w = images[0].shape
    rows = h // tile_size
    cols = w // tile_size

    tiles = {}
    for r in range(rows):
        for c in range(cols):
            patches = [img[:, r * tile_size:(r + 1) * tile_size,
                              c * tile_size:(c + 1) * tile_size] for img in images]
            tiles[(r, c)] = np.stack(patches)

    return tiles, rows, cols


def tile_images_overlap(images, tile_size=TILE_SIZE, overlap=0.5):
    _, h, w = images[0].shape
    stride = max(1, int(tile_size * (1 - overlap)))

    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patches = [img[:, y:y + tile_size, x:x + tile_size] for img in images]
            tiles.append((y, x, np.stack(patches)))

    if (w - tile_size) % stride != 0:
        x = w - tile_size
        for y in range(0, h - tile_size + 1, stride):
            patches = [img[:, y:y + tile_size, x:x + tile_size] for img in images]
            tiles.append((y, x, np.stack(patches)))
    if (h - tile_size) % stride != 0:
        y = h - tile_size
        for x in range(0, w - tile_size + 1, stride):
            patches = [img[:, y:y + tile_size, x:x + tile_size] for img in images]
            tiles.append((y, x, np.stack(patches)))
    if (w - tile_size) % stride != 0 and (h - tile_size) % stride != 0:
        y, x = h - tile_size, w - tile_size
        patches = [img[:, y:y + tile_size, x:x + tile_size] for img in images]
        tiles.append((y, x, np.stack(patches)))

    return tiles


def _select_images(stack, n_lr_images):
    n_available = stack.shape[0]
    if n_available < n_lr_images:
        raise ValueError(
            f"Imagens insuficientes: modelo espera {n_lr_images}, "
            f"mas apenas {n_available} foram fornecidas. "
            f"Forneça exatamente {n_lr_images} imagens ou use o peso esrgan_{n_available}S2.pth."
        )
    # Prefer non-black tiles; fill remainder from nodata tiles if needed
    good_idx, bad_idx = [], []
    for j in range(n_available):
        if np.any(stack[j] == 0) and np.mean(stack[j] == 0) > 0.1:
            bad_idx.append(j)
        else:
            good_idx.append(j)

    if len(good_idx) >= n_lr_images:
        return good_idx[:n_lr_images]
    return (good_idx + bad_idx)[:n_lr_images]


def run_inference(tiles, model, device, n_lr_images):
    outputs = {}
    total = len(tiles)
    with torch.no_grad():
        for i, ((r, c), stack) in enumerate(tiles.items()):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processing tile {i + 1}/{total}...")
            indices = _select_images(stack, n_lr_images)
            selected = stack[indices]
            tensor = torch.from_numpy(selected).float().to(device) / 255.0
            tensor = tensor.reshape(1, -1, TILE_SIZE, TILE_SIZE)
            output = model(tensor)
            output = (torch.clamp(output, 0, 1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            outputs[(r, c)] = output
    return outputs


def run_inference_overlap(tiles, model, device, n_lr_images):
    outputs = []
    total = len(tiles)
    with torch.no_grad():
        for i, (y, x, stack) in enumerate(tiles):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processing tile {i + 1}/{total}...")
            indices = _select_images(stack, n_lr_images)
            selected = stack[indices]
            tensor = torch.from_numpy(selected).float().to(device) / 255.0
            tensor = tensor.reshape(1, -1, TILE_SIZE, TILE_SIZE)
            output = model(tensor)
            output = torch.clamp(output, 0, 1).squeeze(0).cpu().numpy()
            outputs.append((y, x, output))
    return outputs


def assemble_output(output_tiles, rows, cols, output_tile_size=OUTPUT_TILE_SIZE):
    h = rows * output_tile_size
    w = cols * output_tile_size
    result = np.zeros((3, h, w), dtype=np.uint8)
    for (r, c), tile in output_tiles.items():
        result[:,
               r * output_tile_size:(r + 1) * output_tile_size,
               c * output_tile_size:(c + 1) * output_tile_size] = tile
    return result


def assemble_output_blended(output_tiles, input_h, input_w, blend_type='linear'):
    out_h = input_h * SCALE
    out_w = input_w * SCALE
    mosaic = np.zeros((3, out_h, out_w), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)
    mask_cache = {}

    for (y, x, tile) in output_tiles:
        sr_y, sr_x = y * SCALE, x * SCALE
        h_final = min(tile.shape[1], out_h - sr_y)
        w_final = min(tile.shape[2], out_w - sr_x)
        if h_final <= 0 or w_final <= 0:
            continue
        tile = tile[:, :h_final, :w_final]
        key = (h_final, w_final)
        if key not in mask_cache:
            mask_cache[key] = create_blend_mask(h_final, w_final, blend_type)
        mask = mask_cache[key]
        mosaic[:, sr_y:sr_y + h_final, sr_x:sr_x + w_final] += tile * mask[np.newaxis, :, :]
        weights[sr_y:sr_y + h_final, sr_x:sr_x + w_final] += mask

    weights = np.maximum(weights, 1e-10)
    mosaic /= weights[np.newaxis, :, :]
    return np.clip(mosaic * 255, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Super-Resolution Inference — GeoTIFF output")
    parser.add_argument("--images", nargs="+", required=True, help="Input GeoTIFFs (same region)")
    parser.add_argument("--weights", required=True, help="Path to ESRGAN .pth weights")
    parser.add_argument("--output", required=True, help="Output GeoTIFF path")
    parser.add_argument("--n_lr_images", type=int, default=8)
    parser.add_argument("--target_res", type=float, default=TARGET_RES,
                        help=f"Target input resolution in meters (default: {TARGET_RES}m). "
                             "Use 0 for native resolution.")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Tile overlap fraction 0.0–0.75 (default: 0.5)")
    parser.add_argument("--blend", type=str, default="linear",
                        choices=["linear", "gaussian", "cosine"])
    args = parser.parse_args()

    if len(args.images) < args.n_lr_images:
        raise SystemExit(
            f"Erro: modelo esrgan_{args.n_lr_images}S2 requer {args.n_lr_images} imagens, "
            f"mas apenas {len(args.images)} foram fornecidas."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    opt = {
        "scale": SCALE,
        "n_lr_images": args.n_lr_images,
        "network_g": {
            "type": "SSR_RRDBNet",
            "num_in_ch": args.n_lr_images * 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
        }
    }
    model = build_network(opt)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict["params_ema"], strict=True)
    model = model.to(device).eval()
    print("Model loaded.")

    effective_res = args.target_res if args.target_res != 0 else 0
    print(f"Loading {len(args.images)} images...")
    images, meta = load_and_reproject(args.images, target_res=effective_res)
    actual_res = meta.pop("actual_res")
    print(f"Reprojected to {meta['width']}x{meta['height']} at {actual_res:.3f}m.")

    if args.overlap > 0:
        print(f"Tiling with {args.overlap * 100:.0f}% overlap, blend={args.blend}...")
        tiles = tile_images_overlap(images, overlap=args.overlap)
        print(f"Created {len(tiles)} overlapping tiles.")
        print("Running inference...")
        output_tiles = run_inference_overlap(tiles, model, device, args.n_lr_images)
        print("Assembling with weighted blending...")
        result = assemble_output_blended(output_tiles, meta["height"], meta["width"], args.blend)
    else:
        print("Tiling (no overlap)...")
        tiles, rows, cols = tile_images(images)
        print(f"Created {len(tiles)} tiles ({rows}x{cols} grid).")
        print("Running inference...")
        output_tiles = run_inference(tiles, model, device, args.n_lr_images)
        print("Assembling output...")
        result = assemble_output(output_tiles, rows, cols)

    out_width = meta["width"] * SCALE
    out_height = meta["height"] * SCALE
    out_res = actual_res / SCALE

    bounds = rasterio.transform.array_bounds(meta["height"], meta["width"], meta["transform"])
    out_transform = from_bounds(*bounds, out_width, out_height)

    out_meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": out_width,
        "height": out_height,
        "count": 3,
        "crs": meta["crs"],
        "transform": out_transform,
        "compress": "deflate",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(output_path), "w", **out_meta) as dst:
        dst.write(result)

    print(f"Saved: {output_path}")
    print(f"Output resolution: {out_res:.2f}m ({out_width}x{out_height} pixels)")


if __name__ == "__main__":
    main()
