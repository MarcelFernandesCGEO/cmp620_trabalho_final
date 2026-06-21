"""
run_all_regions.py — Orquestrador multi-região do experimento de invariância.

Convenção de entrada (sem duplicar cenas, sem gerenciar pastas):
    dados/bbox/<codigo>.gpkg                   ← uma AOI por região
    dados/sentinel/sentinel_original/*.jp2     ← POOL FLAT de cenas (nomes originais)

O casamento cena↔região é **geográfico**, não por nome: para cada `<codigo>.gpkg`,
a rotina reprojeta a AOI ao CRS de cada cena e seleciona as cenas que **contêm a
AOI** (cobertura ≥ --min_cover). Assim **uma mesma cena serve a vários `.gpkg`**
sem precisar duplicá-la, e gpkg em 4326 + cena em UTM funcionam (reprojeção embutida).

Para cada região com ≥ n_lr cenas cobrindo a AOI:
    1. recorta à AOI            → dados/sentinel/sentinel_original/clipped/<codigo>/
    2. shift_consistency.py     → resultados/<codigo>/shift_consistency/
    3. blend_compare.py         → resultados/<codigo>/blend_compare/

É IDEMPOTENTE e NÃO REMOVE NADA: etapa com CSV de saída presente é pulada (--force
recomputa). Depois rode  scripts/pool_regions.py  para o resumo em resultados/_pooled/.

Uso (no servidor, com GPU e .venv ativo):
    export SATLAS_REPO=/.../satlas-sr
    python scripts/run_all_regions.py \
        --weights /.../weights/esrgan_8S2.pth --n_lr 8
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import rasterio
from shapely.geometry import box
from pyproj import CRS as PyCRS
from clip_to_aoi import load_aoi, aoi_in_crs  # reaproveita a reprojeção da AOI

SCENE_EXTS = ("*.jp2", "*.tif", "*.tiff")


def list_scenes(scenes_dir: Path):
    """Cenas no pool flat (não entra em clipped/)."""
    hits = []
    for ext in SCENE_EXTS:
        hits.extend(scenes_dir.glob(ext))
    return sorted(set(hits))


def scenes_covering_aoi(scenes, aoi_geom, aoi_crs, min_cover):
    """Cenas cujo footprint (bbox) cobre ≥ min_cover da AOI.

    A AOI é reprojetada ao CRS de cada cena (gpkg pode estar em 4326 e a cena em
    UTM). A cobertura é a fração da área da AOI contida no bbox da cena.
    """
    sel = []
    for p in scenes:
        try:
            with rasterio.open(p) as src:
                target_crs = PyCRS.from_user_input(src.crs.to_wkt())
                geom = aoi_in_crs(aoi_geom, aoi_crs, target_crs, use_bbox=False)
                sbox = box(*src.bounds)
        except Exception as e:
            print(f"  ! erro lendo {p.name}: {e}")
            continue
        if not geom.intersects(sbox):
            continue
        cover = geom.intersection(sbox).area / geom.area if geom.area > 0 else 0.0
        if cover >= min_cover:
            sel.append(p)
    return sel


def run(cmd):
    """Roda subprocesso herdando o ambiente (SATLAS_REPO etc.); aborta se falhar."""
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Orquestrador multi-região (casamento geográfico)")
    ap.add_argument("--weights", required=True, help="pesos esrgan_<N>S2.pth")
    ap.add_argument("--bbox", default="dados/bbox", help="pasta com <codigo>.gpkg")
    ap.add_argument("--scenes", default="dados/sentinel/sentinel_original",
                    help="pool flat de cenas (*.jp2)")
    ap.add_argument("--clipdir", default="dados/sentinel/sentinel_original/clipped",
                    help="raiz dos recortes (subpasta por código)")
    ap.add_argument("--resultados", default="resultados", help="raiz das saídas")
    ap.add_argument("--n_lr", type=int, default=8)
    ap.add_argument("--min_cover", type=float, default=0.99,
                    help="fração mínima da AOI coberta pela cena (1.0 = contém tudo)")
    ap.add_argument("--max_anchors", type=int, default=40)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--only", nargs="*", default=None,
                    help="processa apenas estes códigos (default: todos)")
    ap.add_argument("--force", action="store_true", help="recomputa etapas já feitas")
    ap.add_argument("--skip_blend", action="store_true", help="roda só o shift_consistency")
    args = ap.parse_args()

    bbox = ROOT / args.bbox
    scenes_dir = ROOT / args.scenes
    clip_root = ROOT / args.clipdir
    res_root = ROOT / args.resultados
    weights = Path(args.weights)
    py = sys.executable

    gpkgs = sorted(bbox.glob("*.gpkg"))
    if args.only:
        gpkgs = [g for g in gpkgs if g.stem in set(args.only)]
    if not gpkgs:
        raise SystemExit(f"Nenhum .gpkg em {bbox} (filtro --only={args.only}).")

    pool = list_scenes(scenes_dir)
    print(f"Regiões: {[g.stem for g in gpkgs]}")
    print(f"Pool de cenas: {len(pool)} em {scenes_dir}")
    done, skipped, failed = [], [], []

    for gpkg in gpkgs:
        code = gpkg.stem
        print(f"\n=== Região: {code} ===")
        aoi_geom, aoi_crs = load_aoi(gpkg)
        scenes = scenes_covering_aoi(pool, aoi_geom, aoi_crs, args.min_cover)
        print(f"  {len(scenes)} cenas cobrem a AOI (≥{args.min_cover:.0%}): "
              f"{[s.name for s in scenes]}")
        if len(scenes) < args.n_lr:
            print(f"  ! {len(scenes)} < n_lr={args.n_lr}: PULANDO região.")
            skipped.append(code)
            continue

        # --- 1. recorte à AOI (idempotente) ---
        clip_dir = clip_root / code
        clipped = []
        for ext in SCENE_EXTS:
            clipped.extend(clip_dir.glob(ext))
        clipped = sorted(set(clipped))
        if args.force or len(clipped) < len(scenes):
            clip_dir.mkdir(parents=True, exist_ok=True)
            run([py, "scripts/clip_to_aoi.py", "--aoi", gpkg,
                 "--input", *scenes, "--outdir", clip_dir])
            clipped = []
            for ext in SCENE_EXTS:
                clipped.extend(clip_dir.glob(ext))
            clipped = sorted(set(clipped))
        else:
            print(f"  [clip] já existe ({len(clipped)} cenas) — pulado.")
        if len(clipped) < args.n_lr:
            print(f"  ! recorte resultou em {len(clipped)} < n_lr: PULANDO.")
            failed.append(code)
            continue

        region_out = res_root / code

        # --- 2. shift_consistency (idempotente) ---
        sc_out = region_out / "shift_consistency"
        if args.force or not (sc_out / "shift_metrics.csv").exists():
            run([py, "scripts/shift_consistency.py",
                 "--images", *clipped, "--weights", weights,
                 "--n_lr", args.n_lr, "--max_anchors", args.max_anchors,
                 "--outdir", sc_out])
        else:
            print("  [shift] shift_metrics.csv já existe — pulado.")

        # --- 3. blend_compare (idempotente; depende do border_profile) ---
        if not args.skip_blend:
            bl_out = region_out / "blend_compare"
            profile = sc_out / "border_profile.npy"
            if not profile.exists():
                print(f"  ! {profile} ausente — blend não roda. PULANDO blend.")
                failed.append(code)
            elif args.force or not (bl_out / "blend_metrics.csv").exists():
                run([py, "scripts/blend_compare.py",
                     "--images", *clipped, "--weights", weights,
                     "--n_lr", args.n_lr, "--max_size", args.max_size,
                     "--profile", profile, "--outdir", bl_out])
            else:
                print("  [blend] blend_metrics.csv já existe — pulado.")

        done.append(code)

    print("\n================= RESUMO =================")
    print(f"  processadas : {done}")
    print(f"  puladas     : {skipped}  (cenas insuficientes cobrindo a AOI)")
    if failed:
        print(f"  com falha   : {failed}")
    print(f"\nAgora rode:  python scripts/pool_regions.py")


if __name__ == "__main__":
    main()
