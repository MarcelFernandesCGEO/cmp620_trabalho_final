"""
clip_to_aoi.py — Recorta GeoTIFFs/JP2 por uma geometria vetorial (GeoPackage).

Lê a geometria de uma AOI (.gpkg), reprojeta-a para o CRS de cada cena de entrada
e recorta. A saída mantém o CRS e a resolução NATIVOS de cada cena — nenhuma
reprojeção do raster é feita antes do recorte (isso preserva a exatidão das
coordenadas; a reprojeção para o grid comum acontece depois, em sr_core.load_stack).

Substitui o clip_to_gt.py (que recortava pelo bbox de um raster GT). Aqui o recorte
é definido por um vetor, sem depender de GT.

Usage:
    # Recorte pela geometria exata da AOI:
    python scripts/clip_to_aoi.py \
        --aoi    dados/bbox/minha_area.gpkg \
        --input  dados/sentinel/sentinel_original/*.jp2 \
        --outdir dados/sentinel/sentinel_original/clipped/

    # Recorte pelo bounding box da AOI (retângulo):
    python scripts/clip_to_aoi.py --bbox \
        --aoi dados/bbox/minha_area.gpkg \
        --input dados/sentinel/sentinel_original/*.tif \
        --outdir dados/sentinel/sentinel_original/clipped/

    # Camada específica do gpkg e saída ao lado da original (sufixo _clipped):
    python scripts/clip_to_aoi.py --layer minha_camada \
        --aoi dados/bbox/minha_area.gpkg \
        --input dados/sentinel/sentinel_original/*.jp2
"""

import argparse
import sys
from pathlib import Path

import fiona
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as shapely_transform, unary_union
from pyproj import CRS as PyCRS, Transformer


def load_aoi(aoi_path: Path, layer=None):
    """Lê a AOI do .gpkg e devolve (geometria shapely unificada, pyproj.CRS)."""
    open_kwargs = {"layer": layer} if layer else {}
    with fiona.open(str(aoi_path), **open_kwargs) as src:
        if len(src) == 0:
            raise ValueError(f"AOI vazia: {aoi_path}")
        crs = PyCRS.from_user_input(src.crs_wkt or src.crs)
        geoms = [shape(f["geometry"]) for f in src if f["geometry"] is not None]
    if not geoms:
        raise ValueError(f"Nenhuma geometria válida em {aoi_path}")
    return unary_union(geoms), crs


def aoi_in_crs(aoi_geom, aoi_crs: PyCRS, target_crs: PyCRS, use_bbox: bool):
    """Reprojeta a geometria da AOI para target_crs (e, se pedido, usa o bbox dela)."""
    geom = aoi_geom
    if not aoi_crs.equals(target_crs):
        tr = Transformer.from_crs(aoi_crs, target_crs, always_xy=True)
        geom = shapely_transform(lambda x, y, z=None: tr.transform(x, y), geom)
    if use_bbox:
        geom = box(*geom.bounds)
    return geom


def clip_image(input_path: Path, aoi_geom, aoi_crs, output_path: Path, use_bbox: bool):
    with rasterio.open(input_path) as src:
        target_crs = PyCRS.from_user_input(src.crs.to_wkt())
        geom = aoi_in_crs(aoi_geom, aoi_crs, target_crs, use_bbox)

        if not geom.intersects(box(*src.bounds)):
            print(f"  AVISO: {input_path.name} não sobrepõe a AOI — pulando.")
            return False

        out_data, out_transform = rio_mask(src, [mapping(geom)], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_data.shape[1],
            "width":  out_data.shape[2],
            "transform": out_transform,
            "compress": "deflate",
        })
        res = src.res[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", **out_meta) as dst:
        dst.write(out_data)

    h, w = out_data.shape[1], out_data.shape[2]
    print(f"  {input_path.name} → {output_path.name}  ({w}x{h} px, {res:.2f}m/px)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Clip rasters por AOI vetorial (.gpkg)")
    parser.add_argument("--aoi", required=True, help="GeoPackage com a geometria da AOI")
    parser.add_argument("--input", nargs="+", required=True, help="Cenas a recortar")
    parser.add_argument("--outdir", default=None,
                        help="Diretório de saída (padrão: ao lado da cena, sufixo _clipped)")
    parser.add_argument("--layer", default=None, help="Camada do gpkg (padrão: a primeira)")
    parser.add_argument("--bbox", action="store_true",
                        help="Recorta pelo bounding box da AOI em vez da geometria exata")
    args = parser.parse_args()

    aoi_path = Path(args.aoi)
    if not aoi_path.exists():
        print(f"Erro: AOI não encontrada: {aoi_path}", file=sys.stderr)
        sys.exit(1)

    aoi_geom, aoi_crs = load_aoi(aoi_path, args.layer)
    modo = "bbox" if args.bbox else "geometria exata"
    print(f"AOI: {aoi_path.name} | CRS {aoi_crs.to_authority() or aoi_crs.name} | recorte por {modo}")

    paths = [Path(p) for p in args.input]
    ok = 0
    for p in paths:
        if not p.exists():
            print(f"  Não encontrado: {p}", file=sys.stderr)
            continue
        out = (Path(args.outdir) / p.name) if args.outdir \
            else p.parent / f"{p.stem}_clipped{p.suffix}"
        if clip_image(p, aoi_geom, aoi_crs, out, args.bbox):
            ok += 1

    print(f"\n{ok}/{len(paths)} cenas recortadas.")


if __name__ == "__main__":
    main()
