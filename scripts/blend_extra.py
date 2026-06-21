"""
blend_extra.py — Estratégias ADICIONAIS de junção de tiles (experimentais).

Complementa `blend_compare.py` SEM alterá-lo: importa dele as funções de mosaico e
métrica (mesma definição de seam_excess / psnr_ref / ssim_ref → resultados
comparáveis) e acrescenta novas estratégias. Saída em pasta própria
(`blend_extra/`), então o pipeline atual (`blend_compare/`) fica intacto como
resultado de referência.

Estratégias:
    smoothed   — data-driven com o perfil de borda SUAVIZADO/monótono (corrige o
                 ruído que fez o data-driven cru perder para o gaussiano).
    innercrop  — descarta os k px de borda (onde o erro de equivariância é máximo)
                 e remonta só os interiores; k sai do perfil (--crop_px p/ fixar).
    tukey      — janela de Tukey (topo plano + bordas em cosseno), --tukey_alpha.
    ensemble   — *shift-ensemble*: roda SR em MUITOS offsets (stride pequeno) e faz
                 a média por pixel. Como o modelo NÃO é equivariante, cada pixel é
                 coberto por tiles em posições relativas diferentes e o artefato de
                 borda se cancela. Usa a não-invariância a favor. (mais inferência)
    poisson    — *gradient-domain*: reconstrói o mosaico a partir de um campo de
                 gradientes SEM costura (gradiente vindo sempre do tile "dono" de
                 cada pixel) resolvendo Poisson via DCT (Neumann). Ataca direto a
                 métrica seam_excess (razão de gradientes). Requer scipy.

Uso (no servidor, por região):
    python scripts/blend_extra.py \
        --images dados/sentinel/sentinel_original/clipped/<codigo>/*.jp2 \
        --weights $WEIGHTS \
        --profile resultados/<codigo>/shift_consistency/border_profile.npy \
        --outdir  resultados/<codigo>/blend_extra
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import sr_core
from sr_core import TILE_SIZE, SCALE
# Reuso (sem alterar) das funções já validadas do blend_compare:
from blend_compare import (
    tile_positions, edge_dist_2d, build_mask, assemble, assemble_none,
    assemble_center_ref, seam_excess, metrics_vs_ref, _save_png, _bar_fig,
)


# ---------------------------------------------------------------------------
# Perfil de borda suavizado (corrige o data-driven cru)
# ---------------------------------------------------------------------------
def smooth_profile(prof, win=5):
    """Suaviza o perfil e o torna não-crescente com a distância à borda.

    O erro decai para o interior; impor monotonicidade remove o ruído de
    amostragem que prejudicou o data-driven. Bins sem amostra (cauda) recebem o
    último valor válido.
    """
    p = np.asarray(prof, float).copy()
    valid = np.where(p > 0)[0]
    if len(valid) == 0:
        return np.maximum(p, 1e-6)
    p[valid[-1] + 1:] = p[valid[-1]]                 # estende último válido
    k = np.ones(int(win)) / float(win)
    ps = np.convolve(p, k, mode="same")
    ps = np.minimum.accumulate(ps)                   # não-crescente
    return np.maximum(ps, 1e-6)


def derive_crop_px(prof_smooth, T, frac=0.5):
    """k = 1ª distância onde o erro cai abaixo de frac*erro_na_borda."""
    thr = frac * prof_smooth[0]
    below = np.where(prof_smooth <= thr)[0]
    return int(below[0]) if len(below) else T // 4


# ---------------------------------------------------------------------------
# Máscaras novas
# ---------------------------------------------------------------------------
def build_mask_innercrop(T, k):
    """Peso 1 a partir de k px da borda; ~0 na faixa de borda descartada."""
    d = edge_dist_2d(T)
    return np.clip((d >= k).astype(np.float32), 1e-6, 1).astype(np.float32)


def build_mask_tukey(T, alpha=0.5):
    """Topo plano (peso 1) no interior + taper em cosseno perto da borda."""
    d = edge_dist_2d(T)
    dn = d / d.max()
    a = max(alpha, 1e-6)
    m = np.where(dn >= a, 1.0, 0.5 * (1 - np.cos(np.pi * dn / a)))
    return np.clip(m, 1e-6, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Shift-ensemble: média de muitos offsets
# ---------------------------------------------------------------------------
def assemble_ensemble(model, stack, frame_idx, H, W, T, device, tile, ens_stride):
    """Roda SR em grade densa (stride pequeno) e faz média uniforme por pixel.

    Cada pixel é coberto por vários tiles em posições relativas diferentes; como o
    erro de borda depende da posição, a média o cancela (test-time augmentation).
    """
    ys, xs, _ = tile_positions(H, W, tile, 1.0 - ens_stride / tile)
    n = len(ys) * len(xs)
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((y, x, sr_core.sr_window(model, stack, frame_idx, y, x, device, tile)))
    ones = np.ones((T, T), np.float32)
    return assemble(tiles, H, W, T, ones), n


# ---------------------------------------------------------------------------
# Gradient-domain (Poisson via DCT, Neumann)
# ---------------------------------------------------------------------------
def assemble_poisson(tiles, H, W, T):
    """Reconstrução em domínio de gradiente, sem costura (Dirichlet).

    Para cada pixel, o gradiente-guia vem do tile "dono" (centro mais próximo),
    de modo que não há salto de gradiente nas junções. Resolve-se min_I ||∇I - G||²
    (equação de Poisson ∇²I = div G) com **condição de Dirichlet**: a borda do
    mosaico é fixada ao composto center-ref (I=C na borda). Substituindo u=I-C
    (u=0 na borda) e resolvendo ∇²u = div G - ∇²C via DST-I, a intensidade fica
    ancorada e não há deriva de tom (o que penalizava o PSNR na versão Neumann).
    Requer scipy.
    """
    try:
        import scipy.fft as sfft
    except ImportError:
        print("  [poisson] scipy ausente — pulado.")
        return None

    out_h, out_w = H * SCALE, W * SCALE
    best = np.full((out_h, out_w), np.inf, np.float32)
    yy, xx = np.mgrid[0:T, 0:T]
    cdist = np.sqrt((yy - T / 2) ** 2 + (xx - T / 2) ** 2).astype(np.float32)

    C = np.zeros((3, out_h, out_w), np.float32)   # composto center-ref (DN)
    Gx = np.zeros((3, out_h, out_w), np.float32)  # gradiente-guia (forward, x)
    Gy = np.zeros((3, out_h, out_w), np.float32)  # gradiente-guia (forward, y)

    for (y, x, hr) in tiles:
        sy, sx = y * SCALE, x * SCALE
        h = min(T, out_h - sy); w = min(T, out_w - sx)
        reg_best = best[sy:sy + h, sx:sx + w]
        better = cdist[:h, :w] < reg_best
        hrd = hr[:, :h, :w] * 255.0
        tgx = np.zeros((3, h, w), np.float32); tgx[:, :, :-1] = hrd[:, :, 1:] - hrd[:, :, :-1]
        tgy = np.zeros((3, h, w), np.float32); tgy[:, :-1, :] = hrd[:, 1:, :] - hrd[:, :-1, :]
        for c in range(3):
            subC = C[c, sy:sy + h, sx:sx + w];  subC[better] = hrd[c][better]
            sGx = Gx[c, sy:sy + h, sx:sx + w];  sGx[better] = tgx[c][better]
            sGy = Gy[c, sy:sy + h, sx:sx + w];  sGy[better] = tgy[c][better]
        reg_best[better] = cdist[:h, :w][better]

    if out_h < 3 or out_w < 3:
        print("  [poisson] região pequena demais — pulado.")
        return None
    # autovalores do Laplaciano 5-pontos com Dirichlet (DST-I) no interior
    nin, win = out_h - 2, out_w - 2
    ii = np.arange(1, nin + 1)[:, None]
    jj = np.arange(1, win + 1)[None, :]
    denom = (2 * np.cos(np.pi * ii / (nin + 1)) - 2) + (2 * np.cos(np.pi * jj / (win + 1)) - 2)

    out = np.zeros((3, out_h, out_w), np.float32)
    for c in range(3):
        gx, gy = Gx[c], Gy[c]
        divx = np.zeros_like(gx); divx[:, 1:] = gx[:, 1:] - gx[:, :-1]; divx[:, 0] = gx[:, 0]
        divy = np.zeros_like(gy); divy[1:, :] = gy[1:, :] - gy[:-1, :]; divy[0, :] = gy[0, :]
        div = divx + divy
        Cc = C[c]
        lapC = (Cc[2:, 1:-1] + Cc[:-2, 1:-1] + Cc[1:-1, 2:] + Cc[1:-1, :-2]
                - 4 * Cc[1:-1, 1:-1])                 # ∇²C no interior (5-pontos)
        rhs = div[1:-1, 1:-1] - lapC                  # ∇²u = div G - ∇²C
        u = sfft.idstn(sfft.dstn(rhs, type=1, norm="ortho") / denom, type=1, norm="ortho")
        I = Cc.copy()
        I[1:-1, 1:-1] = Cc[1:-1, 1:-1] + u            # I = C + u, com I=C na borda
        out[c] = I
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Estratégias extras de blend (experimentais)")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--profile", default="resultados/shift_consistency/border_profile.npy")
    ap.add_argument("--outdir", default="resultados/blend_extra")
    ap.add_argument("--n_lr", type=int, default=8)
    ap.add_argument("--tile", type=int, default=TILE_SIZE)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--methods", default="smoothed,innercrop,tukey,ensemble,poisson",
                    help="lista separada por vírgula")
    ap.add_argument("--ens_stride", type=int, default=8,
                    help="stride (px LR) da grade densa do shift-ensemble (menor = mais média)")
    ap.add_argument("--crop_px", type=int, default=None,
                    help="px HR de borda descartados no innercrop (default: do perfil)")
    ap.add_argument("--tukey_alpha", type=float, default=0.5)
    ap.add_argument("--smooth_win", type=int, default=5)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
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
    print(f"Métodos: {methods}")

    # tiles base (overlap padrão) — usados por smoothed/innercrop/tukey/poisson
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((y, x, sr_core.sr_window(model, stack, frame_idx, y, x, device, tile)))

    pa = np.load(args.profile)
    prof = pa[1] / np.maximum(pa[2], 1)              # erro médio por distância
    prof_s = smooth_profile(prof, args.smooth_win)
    crop_k = args.crop_px if args.crop_px is not None else derive_crop_px(prof_s, T)

    ref = assemble_center_ref(tiles, H, W, T)
    results = {}
    if "smoothed" in methods:
        results["smoothed"] = assemble(tiles, H, W, T, build_mask("data", T, prof_s))
    if "innercrop" in methods:
        print(f"  innercrop: descartando {crop_k} px de borda")
        results["innercrop"] = assemble(tiles, H, W, T, build_mask_innercrop(T, crop_k))
    if "tukey" in methods:
        results["tukey"] = assemble(tiles, H, W, T, build_mask_tukey(T, args.tukey_alpha))
    if "ensemble" in methods:
        img, n = assemble_ensemble(model, stack, frame_idx, H, W, T, device, tile, args.ens_stride)
        print(f"  ensemble: {n} tiles (stride {args.ens_stride} px LR)")
        results["ensemble"] = img
    if "poisson" in methods:
        img = assemble_poisson(tiles, H, W, T)
        if img is not None:
            results["poisson"] = img

    rows = []
    for kind, img in results.items():
        se = seam_excess(img, ys, xs, stride)
        tv = float((np.abs(np.diff(img.astype(np.float64), axis=1)).mean()
                    + np.abs(np.diff(img.astype(np.float64), axis=2)).mean()) / 2)
        ps, ss = metrics_vs_ref(img, ref)
        rows.append({"blend": kind, "seam_excess": se, "tv": tv,
                     "psnr_ref": ps, "ssim_ref": ss})
        _save_png(outdir / f"mosaic_{kind}.png", img)

    cols = ["blend", "seam_excess", "tv", "psnr_ref", "ssim_ref"]
    with open(outdir / "blend_extra_metrics.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.5f}" if isinstance(r[c], float) else r[c]
                             for c in cols) + "\n")
    _bar_fig(outdir, rows)   # reaproveita a figura de barras (salva fig_blend_seam.png)
    try:
        (outdir / "fig_blend_seam.png").rename(outdir / "fig_blend_extra_seam.png")
    except OSError:
        pass

    print(f"OK. {outdir}/blend_extra_metrics.csv")
    for r in rows:
        print(f"  {r['blend']:<10} seam_excess={r['seam_excess']:.3f}  "
              f"psnr_ref={r['psnr_ref']:.2f}  ssim_ref={r['ssim_ref']:.4f}")


if __name__ == "__main__":
    main()
