"""
Coleta metadados de GeoTIFFs e grava em JSON rastreável pelo git.

Para cada imagem gera um arquivo <nome>.meta.json com: CRS, dimensões,
resolução, extent em UTM e WGS84, bandas, dtype, compressão e tamanho em disco.

Usage:
    # Uma imagem:
    python scripts/collect_metadata.py --input dados/gt/gt_original/2970-3-SE.tif

    # Todos os tifs de uma pasta (recursivo):
    python scripts/collect_metadata.py --input dados/**/*.tif --recursive

    # Saída em diretório específico:
    python scripts/collect_metadata.py --input dados/gt/gt_original/2970-3-SE.tif \
        --outdir report/metadata/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def collect(path: Path) -> dict:
    with rasterio.open(path) as src:
        bounds_native = {
            "left":   round(src.bounds.left,   4),
            "bottom": round(src.bounds.bottom, 4),
            "right":  round(src.bounds.right,  4),
            "top":    round(src.bounds.top,    4),
        }
        try:
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            bounds_wgs84 = {"west": round(b[0], 6), "south": round(b[1], 6),
                            "east": round(b[2], 6), "north": round(b[3], 6)}
        except Exception:
            bounds_wgs84 = None

        w_m = abs(src.bounds.right - src.bounds.left)
        h_m = abs(src.bounds.top   - src.bounds.bottom)

        compression = src.profile.get("compress", "none")
        nodata = src.nodata

        return {
            "file":             path.name,
            "path":             str(path),
            "crs":              str(src.crs),
            "epsg":             src.crs.to_epsg(),
            "width_px":         src.width,
            "height_px":        src.height,
            "bands":            src.count,
            "dtype":            src.dtypes[0],
            "res_x_m":          round(src.res[0], 6),
            "res_y_m":          round(src.res[1], 6),
            "extent_km":        {"x": round(w_m / 1000, 4), "y": round(h_m / 1000, 4)},
            "bounds_native":    bounds_native,
            "bounds_wgs84":     bounds_wgs84,
            "nodata":           nodata,
            "compression":      compression,
            "size_mb_disk":     round(os.path.getsize(path) / 1e6, 2),
            "size_mb_uncompressed": round(src.width * src.height * src.count / 1e6, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="Coleta metadados de GeoTIFFs → JSON")
    parser.add_argument("--input", nargs="+", required=True, help="GeoTIFFs de entrada")
    parser.add_argument("--outdir", default=None,
                        help="Diretório de saída para os .meta.json "
                             "(padrão: mesmo diretório da imagem)")
    parser.add_argument("--recursive", action="store_true",
                        help="Expande globs recursivamente com rglob")
    args = parser.parse_args()

    paths = []
    for pattern in args.input:
        p = Path(pattern)
        if p.is_file():
            paths.append(p)
        elif args.recursive:
            paths.extend(sorted(Path(".").rglob(pattern)))
        else:
            paths.extend(sorted(Path(".").glob(pattern)))

    if not paths:
        print("Nenhum arquivo encontrado.", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        path = path.resolve()
        if not path.exists():
            print(f"Não encontrado: {path}", file=sys.stderr)
            continue

        try:
            meta = collect(path)
        except Exception as e:
            print(f"Erro em {path.name}: {e}", file=sys.stderr)
            continue

        if args.outdir:
            out_dir = Path(args.outdir)
        else:
            out_dir = path.parent

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.meta.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(
            f"{path.name:<40}  {meta['width_px']}x{meta['height_px']}  "
            f"{meta['res_x_m']:.4f}m  {meta['extent_km']['x']:.2f}x{meta['extent_km']['y']:.2f}km  "
            f"CRS={meta['epsg']}  {meta['size_mb_disk']:.0f}MB  → {out_path.name}"
        )


if __name__ == "__main__":
    main()
