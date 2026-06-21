"""
shift_consistency.py — Experimento principal.

Mede a invariância a translação do gerador Satlas (ESRGAN). Para uma mesma área
física, roda SR com o tile em duas posições deslocadas horizontalmente por Δ
pixels LR. Na região sobreposta (mesmos pixels físicos), as duas saídas SR
deveriam ser idênticas num modelo perfeitamente equivariante. A discordância
medida quantifica a não-invariância e localiza onde ela se concentra.

Saídas (em --outdir):
    shift_metrics.csv   — por Δ: overlap%, PSNR, SSIM, MAE, RMSE (média sobre âncoras)
    shift_metrics_raw.csv — uma linha por (âncora, Δ)
    border_profile.csv  — erro médio vs distância à borda do tile mais próxima
    border_profile.npy  — (dist_px, soma_erro, n) para reuso no blend_compare
    fig_consistency.png — PSNR/SSIM vs overlap
    fig_border.png      — erro vs distância à borda
    fig_example.png     — exemplo visual (SR_0 | SR_Δ | |diff|) para Δ=tile/2
    config.json         — parâmetros da execução

Uso (no servidor com GPU):
    python shift_consistency.py \
        --images dados/sentinel/sentinel_original/clipped/*.jp2 \
        --weights ~/Desktop/allenai/satlas-super-resolution/pretrained_models/esrgan_8S2.pth \
        --outdir resultados/shift_consistency
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

import sr_core
from sr_core import TILE_SIZE, SCALE


# ---------------------------------------------------------------------------
# Métricas sobre a região de sobreposição
# ---------------------------------------------------------------------------
def overlap_metrics(a, b):
    """a, b: float [3, H, W] em [0,1], mesmos pixels físicos. Retorna dict."""
    A = a.astype(np.float64) * 255.0
    B = b.astype(np.float64) * 255.0
    mae = float(np.abs(A - B).mean())
    mse = float(((A - B) ** 2).mean())
    rmse = float(np.sqrt(mse))
    psnr = 99.99 if mse < 1e-8 else float(10.0 * np.log10(255.0 ** 2 / mse))
    a_hwc = np.transpose(a, (1, 2, 0))
    b_hwc = np.transpose(b, (1, 2, 0))
    # Sobreposições estreitas (Δ grande, p.ex. Δ=31 → 4 px HR) têm menos que os
    # 7 px da janela default do SSIM. Adapta win_size ao menor lado (ímpar) e
    # marca NaN quando a faixa é estreita demais (<3 px) para qualquer janela.
    smaller = min(a_hwc.shape[0], a_hwc.shape[1])
    win = min(7, smaller if smaller % 2 else smaller - 1)
    ss = (float("nan") if smaller < 3
          else float(ssim(a_hwc, b_hwc, channel_axis=2,
                          data_range=1.0, win_size=win)))
    return {"psnr": psnr, "ssim": ss, "mae": mae, "rmse": rmse}


def pick_anchors(stack, frame_idx, tile, max_shift, n_anchors, seed=0):
    """Janelas interiores sem nodata, em grade regular, até n_anchors."""
    _, _, H, W = stack.shape
    sel = stack[frame_idx]
    margin = tile
    ys = range(margin, H - tile - margin, tile)
    xs = range(margin, W - tile - max_shift - margin, tile)
    cands = []
    for y in ys:
        for x in xs:
            win = sel[:, :, y:y + tile, x:x + tile + max_shift]
            if not np.any(win == 0):
                cands.append((y, x))
    rng = np.random.default_rng(seed)
    if len(cands) > n_anchors:
        idx = rng.choice(len(cands), n_anchors, replace=False)
        cands = [cands[i] for i in sorted(idx)]
    return cands


def main():
    ap = argparse.ArgumentParser(description="Experimento de invariância a translação")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--outdir", default="resultados/shift_consistency")
    ap.add_argument("--n_lr", type=int, default=8)
    ap.add_argument("--tile", type=int, default=TILE_SIZE)
    ap.add_argument("--reference", default=None, help="GeoTIFF p/ snapping de grid")
    ap.add_argument("--max_anchors", type=int, default=40)
    ap.add_argument("--shifts", default=None,
                    help="Lista de Δ px LR (ex '0,3,6,10,16,22,29,31'). "
                         "Default: derivado de overlaps 100..~3%%.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile = args.tile

    if args.shifts:
        shifts = sorted({int(s) for s in args.shifts.split(",")})
    else:
        overlaps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03]
        shifts = sorted({int(round(tile * (1 - ov))) for ov in overlaps})
    shifts = [d for d in shifts if 0 <= d < tile]
    max_shift = max(shifts)

    print(f"Device: {device} | tile={tile} | shifts={shifts}")
    stack, meta = sr_core.load_stack(args.images, reference=args.reference)
    frame_idx = sr_core.select_frames(stack, args.n_lr)
    print(f"Grid {meta['width']}x{meta['height']} @ {meta['actual_res']:.3f} m | "
          f"frames fixos: {frame_idx}")
    model = sr_core.build_model(args.weights, args.n_lr, device)

    anchors = pick_anchors(stack, frame_idx, tile, max_shift, args.max_anchors)
    if not anchors:
        raise SystemExit("Nenhuma âncora interior sem nodata encontrada. "
                         "Reduza margens ou verifique cobertura das cenas.")
    print(f"{len(anchors)} âncoras selecionadas.")

    raw_rows = []
    # perfil de borda: bins por distância (px HR) à borda de tile mais próxima
    prof_sum = np.zeros(tile * SCALE, dtype=np.float64)
    prof_cnt = np.zeros(tile * SCALE, dtype=np.float64)
    example = None

    # distâncias à borda (mesmas p/ todo tile HR): min(c, T-1-c)
    T = tile * SCALE
    edge_dist = np.minimum(np.arange(T), T - 1 - np.arange(T))

    for (y, x) in anchors:
        sr0 = sr_core.sr_window(model, stack, frame_idx, y, x, device, tile)
        for d in shifts:
            srD = sr_core.sr_window(model, stack, frame_idx, y, x + d, device, tile)
            ow = (tile - d) * SCALE  # largura da sobreposição em px HR
            if ow <= 0:
                continue
            reg0 = sr0[:, :, d * SCALE:]          # parte direita de sr0
            regD = srD[:, :, :ow]                 # parte esquerda de srD
            m = overlap_metrics(reg0, regD)
            m.update({"y": y, "x": x, "delta": d,
                      "overlap_pct": 100.0 * (tile - d) / tile})
            raw_rows.append(m)

            # perfil: |diff| por coluna vs distância mín. à borda dos dois tiles
            diff_col = np.abs(reg0 - regD).mean(axis=(0, 1)) * 255.0  # [ow]
            d0 = edge_dist[d * SCALE:]            # dist. na grade de sr0
            dD = edge_dist[:ow]                   # dist. na grade de srD
            dmin = np.minimum(d0, dD)
            for dist, val in zip(dmin, diff_col):
                prof_sum[dist] += val
                prof_cnt[dist] += 1

            if example is None and d == tile // 2:
                example = (reg0.copy(), regD.copy())

    # ---- agregação por Δ ----
    raw_rows.sort(key=lambda r: (r["delta"], r["y"], r["x"]))
    deltas = sorted({r["delta"] for r in raw_rows})
    agg = []
    for d in deltas:
        rs = [r for r in raw_rows if r["delta"] == d]
        agg.append({
            "delta": d,
            "overlap_pct": rs[0]["overlap_pct"],
            "psnr": np.mean([r["psnr"] for r in rs]),
            "ssim": np.mean([r["ssim"] for r in rs]),
            "mae": np.mean([r["mae"] for r in rs]),
            "rmse": np.mean([r["rmse"] for r in rs]),
            "n": len(rs),
        })

    # ---- gravação CSV ----
    def write_csv(path, rows, cols):
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(f"{r[c]:.5f}" if isinstance(r[c], float) else str(r[c])
                                 for c in cols) + "\n")

    write_csv(outdir / "shift_metrics.csv", agg,
              ["delta", "overlap_pct", "psnr", "ssim", "mae", "rmse", "n"])
    write_csv(outdir / "shift_metrics_raw.csv", raw_rows,
              ["y", "x", "delta", "overlap_pct", "psnr", "ssim", "mae", "rmse"])

    prof_mean = np.divide(prof_sum, prof_cnt, out=np.zeros_like(prof_sum),
                          where=prof_cnt > 0)
    np.save(outdir / "border_profile.npy",
            np.vstack([np.arange(len(prof_mean)), prof_sum, prof_cnt]))
    with open(outdir / "border_profile.csv", "w") as f:
        f.write("dist_px,mean_abs_diff,n\n")
        for i in range(len(prof_mean)):
            f.write(f"{i},{prof_mean[i]:.5f},{int(prof_cnt[i])}\n")

    with open(outdir / "config.json", "w") as f:
        json.dump({"images": [str(p) for p in args.images], "weights": args.weights,
                   "n_lr": args.n_lr, "tile": tile, "shifts": shifts,
                   "frame_idx": frame_idx, "n_anchors": len(anchors),
                   "grid": [meta["width"], meta["height"]],
                   "res_m": meta["actual_res"]}, f, indent=2)

    _make_figures(outdir, agg, prof_mean, prof_cnt, example, tile)
    print(f"OK. Resultados em {outdir}")


def _make_figures(outdir, agg, prof_mean, prof_cnt, example, tile):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib ausente — figuras puladas.")
        return

    ov = [a["overlap_pct"] for a in agg]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ov, [a["psnr"] for a in agg], "o-", color="tab:blue", label="PSNR")
    ax1.set_xlabel("Sobreposição (%)"); ax1.set_ylabel("Shift-consistency PSNR (dB)",
                                                        color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ov, [a["ssim"] for a in agg], "s--", color="tab:red", label="SSIM")
    ax2.set_ylabel("SSIM", color="tab:red")
    ax1.set_title("Consistência sob deslocamento vs sobreposição")
    fig.tight_layout(); fig.savefig(outdir / "fig_consistency.png", dpi=150)
    plt.close(fig)

    valid = prof_cnt > 0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(prof_mean))[valid], prof_mean[valid], "-")
    ax.set_xlabel("Distância à borda de tile mais próxima (px HR)")
    ax.set_ylabel("|diferença| média (DN)")
    ax.set_title("Onde a não-invariância se concentra")
    fig.tight_layout(); fig.savefig(outdir / "fig_border.png", dpi=150)
    plt.close(fig)

    if example is not None:
        reg0, regD = example
        diff = np.abs(reg0 - regD).mean(axis=0)
        fig, axs = plt.subplots(1, 3, figsize=(11, 4))
        axs[0].imshow(np.transpose(reg0, (1, 2, 0))); axs[0].set_title("SR offset 0")
        axs[1].imshow(np.transpose(regD, (1, 2, 0))); axs[1].set_title(f"SR offset {tile//2}px")
        im = axs[2].imshow(diff, cmap="magma"); axs[2].set_title("|diferença|")
        fig.colorbar(im, ax=axs[2], fraction=0.046)
        for a in axs:
            a.axis("off")
        fig.tight_layout(); fig.savefig(outdir / "fig_example.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
