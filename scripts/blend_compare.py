"""
blend_compare.py — Compara estratégias de junção de tiles (blend) usando o perfil
de erro de borda medido por shift_consistency.py.

Monta um mosaico SR de uma região com tiles sobrepostos e funde os tiles por
diferentes máscaras de peso:
    none      — colagem dura (sobrescreve), sem fusão (baseline com artefatos)
    linear    — rampa linear até a borda
    cosine    — janela de cosseno (Hann)
    gaussian  — janela gaussiana
    data      — peso = confiabilidade derivada de border_profile.npy
                (peso baixo onde o modelo é comprovadamente menos invariante)

Métricas:
    seam_excess — gradiente médio nas linhas de junção / gradiente médio global
                  (ideal ≈ 1.0; > 1 indica costura visível)
    tv          — variação total do mosaico (suavidade global)
    psnr_ref/ssim_ref — fidelidade vs mosaico de referência por recorte central
                  (cada pixel vem do tile cujo centro é mais próximo = região
                  mais confiável de cada tile)

Saídas (em --outdir):
    blend_metrics.csv, fig_blend_seam.png e um GeoTIFF/PNG por blend.

Uso (no servidor):
    python blend_compare.py \
        --images dados/sentinel/sentinel_original/clipped/*.jp2 \
        --weights .../esrgan_8S2.pth \
        --profile resultados/shift_consistency/border_profile.npy \
        --outdir resultados/blend_compare
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sr_core
from sr_core import TILE_SIZE, SCALE


def tile_positions(H, W, tile, overlap):
    stride = max(1, int(round(tile * (1 - overlap))))
    ys = list(range(0, H - tile + 1, stride))
    xs = list(range(0, W - tile + 1, stride))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    return ys, xs, stride


def edge_dist_2d(T):
    a = np.minimum(np.arange(T), T - 1 - np.arange(T))
    return np.minimum.outer(a, a).astype(np.float64)  # [T,T] min dist a qualquer borda


def build_mask(kind, T, profile_mean=None):
    d = edge_dist_2d(T)
    dmax = d.max()
    if kind == "linear":
        m = d / dmax
    elif kind == "cosine":
        m = 0.5 * (1 - np.cos(np.pi * d / dmax))
    elif kind == "gaussian":
        sigma = 0.3 * dmax
        m = np.exp(-((dmax - d) ** 2) / (2 * sigma ** 2))
    elif kind == "data":
        prof = profile_mean.copy()
        prof[prof <= 0] = np.nan
        rel = 1.0 / (prof + 1e-3)              # confiabilidade ∝ 1/erro
        rel = np.nan_to_num(rel, nan=np.nanmin(rel))
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        di = np.clip(d.astype(int), 0, len(rel) - 1)
        m = rel[di]
    else:
        raise ValueError(kind)
    return np.clip(m, 1e-6, 1).astype(np.float32)


def assemble(tiles, H, W, T, mask):
    """tiles: list of (y, x, hr[3,T,T]). Retorna uint8 [3, H*S, W*S]."""
    out_h, out_w = H * SCALE, W * SCALE
    acc = np.zeros((3, out_h, out_w), np.float32)
    wsum = np.zeros((out_h, out_w), np.float32)
    for (y, x, hr) in tiles:
        sy, sx = y * SCALE, x * SCALE
        h = min(T, out_h - sy); w = min(T, out_w - sx)
        acc[:, sy:sy + h, sx:sx + w] += hr[:, :h, :w] * mask[np.newaxis, :h, :w]
        wsum[sy:sy + h, sx:sx + w] += mask[:h, :w]
    wsum = np.maximum(wsum, 1e-6)
    return np.clip(acc / wsum * 255, 0, 255).astype(np.uint8)


def assemble_none(tiles, H, W, T):
    out_h, out_w = H * SCALE, W * SCALE
    out = np.zeros((3, out_h, out_w), np.uint8)
    for (y, x, hr) in tiles:
        sy, sx = y * SCALE, x * SCALE
        h = min(T, out_h - sy); w = min(T, out_w - sx)
        out[:, sy:sy + h, sx:sx + w] = (hr[:, :h, :w] * 255).astype(np.uint8)
    return out


def assemble_center_ref(tiles, H, W, T):
    """Cada pixel vem do tile cujo centro é mais próximo (região mais confiável)."""
    out_h, out_w = H * SCALE, W * SCALE
    out = np.zeros((3, out_h, out_w), np.float32)
    best = np.full((out_h, out_w), np.inf, np.float32)
    yy, xx = np.mgrid[0:T, 0:T]
    cdist = np.sqrt((yy - T / 2) ** 2 + (xx - T / 2) ** 2).astype(np.float32)
    for (y, x, hr) in tiles:
        sy, sx = y * SCALE, x * SCALE
        h = min(T, out_h - sy); w = min(T, out_w - sx)
        region = best[sy:sy + h, sx:sx + w]
        better = cdist[:h, :w] < region
        for c in range(3):
            sub = out[c, sy:sy + h, sx:sx + w]
            sub[better] = (hr[c, :h, :w] * 255)[better]
        region[better] = cdist[:h, :w][better]
    return out.astype(np.uint8)


def seam_excess(img, ys, xs, stride):
    """grad médio nas colunas/linhas de junção vs grad médio global."""
    g = img.astype(np.float64).mean(axis=0)
    gx = np.abs(np.diff(g, axis=1)); gy = np.abs(np.diff(g, axis=0))
    glob = (gx.mean() + gy.mean()) / 2 + 1e-9
    seam_x = [x * SCALE for x in xs[1:-1]]
    seam_y = [y * SCALE for y in ys[1:-1]]
    vals = []
    for sx in seam_x:
        if 1 <= sx < gx.shape[1]:
            vals.append(gx[:, sx - 1:sx + 1].mean())
    for sy in seam_y:
        if 1 <= sy < gy.shape[0]:
            vals.append(gy[sy - 1:sy + 1, :].mean())
    return float(np.mean(vals) / glob) if vals else float("nan")


def metrics_vs_ref(img, ref):
    from skimage.metrics import structural_similarity as ssim
    A = img.astype(np.float64); B = ref.astype(np.float64)
    mse = ((A - B) ** 2).mean()
    psnr = 99.99 if mse < 1e-8 else 10 * np.log10(255.0 ** 2 / mse)
    ss = ssim(np.transpose(img, (1, 2, 0)), np.transpose(ref, (1, 2, 0)),
              channel_axis=2, data_range=255)
    return float(psnr), float(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--profile", default="resultados/shift_consistency/border_profile.npy")
    ap.add_argument("--outdir", default="resultados/blend_compare")
    ap.add_argument("--n_lr", type=int, default=8)
    ap.add_argument("--tile", type=int, default=TILE_SIZE)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--max_size", type=int, default=256,
                    help="recorta a região a max_size px LR p/ custo controlado")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile = args.tile; T = tile * SCALE

    stack, meta = sr_core.load_stack(args.images, reference=args.reference)
    frame_idx = sr_core.select_frames(stack, args.n_lr)
    model = sr_core.build_model(args.weights, args.n_lr, device)

    _, _, H, W = stack.shape
    H = min(H, args.max_size); W = min(W, args.max_size)
    ys, xs, stride = tile_positions(H, W, tile, args.overlap)
    print(f"Device {device} | região {W}x{H} | {len(ys)*len(xs)} tiles | stride {stride}")

    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((y, x, sr_core.sr_window(model, stack, frame_idx, y, x, device, tile)))

    prof = np.load(args.profile)[1] / np.maximum(np.load(args.profile)[2], 1)  # mean por dist

    ref = assemble_center_ref(tiles, H, W, T)
    results = {"none": assemble_none(tiles, H, W, T)}
    for kind in ["linear", "cosine", "gaussian", "data"]:
        results[kind] = assemble(tiles, H, W, T, build_mask(kind, T, prof))

    rows = []
    for kind, img in results.items():
        se = seam_excess(img, ys, xs, stride)
        tv = float((np.abs(np.diff(img.astype(np.float64), axis=1)).mean()
                    + np.abs(np.diff(img.astype(np.float64), axis=2)).mean()) / 2)
        ps, ss = metrics_vs_ref(img, ref)
        rows.append({"blend": kind, "seam_excess": se, "tv": tv,
                     "psnr_ref": ps, "ssim_ref": ss})
        _save_png(outdir / f"mosaic_{kind}.png", img)

    with open(outdir / "blend_metrics.csv", "w") as f:
        cols = ["blend", "seam_excess", "tv", "psnr_ref", "ssim_ref"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.5f}" if isinstance(r[c], float) else r[c]
                             for c in cols) + "\n")
    _save_png(outdir / "mosaic_center_ref.png", ref)
    _bar_fig(outdir, rows)
    print(f"OK. {outdir}/blend_metrics.csv")
    for r in rows:
        print(f"  {r['blend']:<9} seam_excess={r['seam_excess']:.3f}  "
              f"psnr_ref={r['psnr_ref']:.2f}  ssim_ref={r['ssim_ref']:.4f}")


def _save_png(path, img):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(str(path), np.transpose(img, (1, 2, 0)))
    except ImportError:
        pass


def _bar_fig(outdir, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [r["blend"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [r["seam_excess"] for r in rows], color="tab:blue")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="sem costura (ideal)")
    ax.set_ylabel("seam_excess (grad junção / grad global)")
    ax.set_title("Visibilidade de costura por estratégia de blend")
    ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "fig_blend_seam.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
