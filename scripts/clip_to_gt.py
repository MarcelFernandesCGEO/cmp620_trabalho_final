"""
Clipa GeoTIFFs para o extent do GT de referência.

Lê o bounding box do GT (em qualquer CRS), reprojecta para o CRS de cada
imagem de entrada e executa o clip. A saída mantém o CRS e a resolução
nativa de cada imagem — nenhuma reprojeção é feita antes do clip.

Usage:
    python scripts/clip_to_gt.py \
        --reference dados/gt/gt_original/2970-3-SE.tif \
        --input     dados/sentinel/sentinel_original/*.jp2 \
        --outdir    dados/sentinel/sentinel_original/clipped/

    # Saída ao lado da original (sufixo _clipped):
    python scripts/clip_to_gt.py \
        --reference dados/gt/gt_original/2970-3-SE.tif \
        --input     dados/sentinel/sentinel_original/*.jp2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box


def gt_bounds_in_crs(gt_path: Path, target_crs: CRS) -> box:
    with rasterio.open(gt_path) as src:
        b = transform_bounds(src.crs, target_crs, *src.bounds)
    return box(b[0], b[1], b[2], b[3])


def clip_image(input_path: Path, gt_path: Path, output_path: Path):
    with rasterio.open(input_path) as src:
        geom = gt_bounds_in_crs(gt_path, src.crs)

        if not geom.intersects(box(*src.bounds)):
            print(f"  AVISO: {input_path.name} não sobrepõe o GT — pulando.")
            return False

        out_data, out_transform = rio_mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_data.shape[1],
            "width":  out_data.shape[2],
            "transform": out_transform,
            "compress": "deflate",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", **out_meta) as dst:
        dst.write(out_data)

    h, w = out_data.shape[1], out_data.shape[2]
    with rasterio.open(input_path) as src:
        res = src.res[0]
    print(f"  {input_path.name} → {output_path.name}  ({w}x{h} px, {res:.2f}m/px)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Clip GeoTIFFs to GT extent")
    parser.add_argument("--reference", required=True, help="GT de referência (define o extent)")
    parser.add_argument("--input", nargs="+", required=True, help="Imagens a clipar")
    parser.add_argument("--outdir", default=None,
                        help="Diretório de saída (padrão: mesmo da imagem, sufixo _clipped)")
    args = parser.parse_args()

    gt_path = Path(args.reference)
    if not gt_path.exists():
        print(f"Erro: GT não encontrado: {gt_path}", file=sys.stderr)
        sys.exit(1)

    paths = [Path(p) for p in args.input]
    ok = 0
    for p in paths:
        if not p.exists():
            print(f"  Não encontrado: {p}", file=sys.stderr)
            continue

        if args.outdir:
            out = Path(args.outdir) / p.name
        else:
            out = p.parent / f"{p.stem}_clipped{p.suffix}"

        if clip_image(p, gt_path, out):
            ok += 1

    print(f"\n{ok}/{len(paths)} imagens clipadas.")


if __name__ == "__main__":
    main()
