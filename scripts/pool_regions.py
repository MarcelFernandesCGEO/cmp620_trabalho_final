"""
pool_regions.py — Sumariza (pooled) os resultados de TODAS as regiões.

Lê cada  resultados/<codigo>/{shift_consistency,blend_compare}/  e gera, em
resultados/_pooled/ :

    shift_metrics_pooled.csv   média±dp entre regiões por sobreposição (Δ)
    border_profile_pooled.csv  perfil de borda combinado (média ponderada por n)
    blend_metrics_pooled.csv   média±dp entre regiões por estratégia de blend
    fig_consistency_pooled.png scPSNR vs overlap: uma curva por região + pooled
    fig_border_pooled.png      |diferença| vs distância à borda por região + pooled
    fig_blend_pooled.png       seam_excess por blend (média entre regiões, com dp)
    summary.txt                tabela-resumo legível

Depende só de numpy + matplotlib (sem pandas/torch); roda no servidor ou local.

Uso:
    python scripts/pool_regions.py
    python scripts/pool_regions.py --resultados resultados --out resultados/_pooled
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def read_csv(path):
    """Lê um CSV com cabeçalho como lista de dicts {coluna: str}."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_regions(res_root: Path):
    """Subpastas de resultados que NÃO começam com '_' e têm shift_metrics.csv."""
    regions = []
    for d in sorted(res_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "shift_consistency" / "shift_metrics.csv").exists():
            regions.append(d.name)
    return regions


def weighted_mean(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    w = weights.sum()
    return float((values * weights).sum() / w) if w > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Pooling multi-região")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--out", default="resultados/_pooled")
    args = ap.parse_args()

    res_root = ROOT / args.resultados
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    regions = find_regions(res_root)
    if not regions:
        raise SystemExit(f"Nenhuma região com resultados em {res_root}/<codigo>/.")
    print(f"Regiões: {regions}")

    # ===================================================================
    # 1. Shift-consistency  (chave = delta)
    # ===================================================================
    sc = {}  # region -> {delta: row}
    for r in regions:
        rows = read_csv(res_root / r / "shift_consistency" / "shift_metrics.csv")
        sc[r] = {int(round(float(x["delta"]))): x for x in rows}
    deltas = sorted({d for r in regions for d in sc[r]})

    pooled_sc = []
    for d in deltas:
        present = [sc[r][d] for r in regions if d in sc[r]]
        ov = float(present[0]["overlap_pct"])
        psnr = [float(x["psnr"]) for x in present]
        ssim = [float(x["ssim"]) for x in present]
        mae = [float(x["mae"]) for x in present]
        rmse = [float(x["rmse"]) for x in present]
        pooled_sc.append({
            "delta": d, "overlap_pct": round(ov, 3), "n_regions": len(present),
            "psnr_mean": round(np.mean(psnr), 4), "psnr_std": round(np.std(psnr), 4),
            "ssim_mean": round(np.mean(ssim), 4), "ssim_std": round(np.std(ssim), 4),
            "mae_mean": round(np.mean(mae), 4), "rmse_mean": round(np.mean(rmse), 4),
        })
    with open(out / "shift_metrics_pooled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pooled_sc[0].keys()))
        w.writeheader(); w.writerows(pooled_sc)

    # figura: scPSNR vs overlap — regiões (cinza) + pooled (faixa ±dp)
    plt.figure(figsize=(7, 5))
    for i, r in enumerate(regions):
        ov = [float(sc[r][d]["overlap_pct"]) for d in sorted(sc[r])]
        ps = [float(sc[r][d]["psnr"]) for d in sorted(sc[r])]
        plt.plot(ov, ps, color="0.7", lw=0.6, alpha=0.5, zorder=1,
                 label=(f"regiões (n={len(regions)})" if i == 0 else None))
    ov_p = np.array([x["overlap_pct"] for x in pooled_sc])
    ps_p = np.array([x["psnr_mean"] for x in pooled_sc])
    sd_p = np.array([x["psnr_std"] for x in pooled_sc])
    plt.fill_between(ov_p, ps_p - sd_p, ps_p + sd_p, color="tab:blue",
                     alpha=0.2, lw=0, zorder=2, label="pooled ±dp")
    plt.plot(ov_p, ps_p, color="tab:blue", marker="s", lw=2.5, zorder=3, label="pooled")
    plt.xlabel("Sobreposição (%)"); plt.ylabel("Shift-consistency PSNR (dB)")
    plt.title(f"Consistência sob deslocamento — {len(regions)} regiões")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "fig_consistency_pooled.png", dpi=150); plt.close()

    # ===================================================================
    # 2. Perfil de borda  (chave = dist_px; média ponderada por n)
    # ===================================================================
    bp = {}  # region -> {dist: (mean_abs_diff, n)}
    for r in regions:
        rows = read_csv(res_root / r / "shift_consistency" / "border_profile.csv")
        bp[r] = {int(float(x["dist_px"])): (float(x["mean_abs_diff"]), float(x["n"]))
                 for x in rows}
    dists = sorted({d for r in regions for d in bp[r]})
    pooled_bp = []
    for d in dists:
        vals = [bp[r][d][0] for r in regions if d in bp[r]]
        ns = [bp[r][d][1] for r in regions if d in bp[r]]
        pooled_bp.append({"dist_px": d,
                          "mean_abs_diff": round(weighted_mean(vals, ns), 5),
                          "n": int(sum(ns))})
    with open(out / "border_profile_pooled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dist_px", "mean_abs_diff", "n"])
        w.writeheader(); w.writerows(pooled_bp)

    # corta a cauda de bins sem amostra (dist profunda só vem do controle Δ=0 → 0)
    dd_p = [x["dist_px"] for x in pooled_bp]
    vv_p = [x["mean_abs_diff"] for x in pooled_bp]
    pos = [d for d, v in zip(dd_p, vv_p) if v > 0.05]
    xmax = max(pos) if pos else (dd_p[-1] if dd_p else 1)

    plt.figure(figsize=(7, 5))
    for i, r in enumerate(regions):
        dd = sorted(bp[r]); vv = [bp[r][d][0] for d in dd]
        plt.plot(dd, vv, color="0.7", lw=0.6, alpha=0.5, zorder=1,
                 label=(f"regiões (n={len(regions)})" if i == 0 else None))
    plt.plot(dd_p, vv_p, color="tab:red", lw=2.5, zorder=3, label="pooled")
    plt.xlim(0, xmax)
    plt.xlabel("Distância à borda de tile mais próxima (px HR)")
    plt.ylabel("|diferença| média (DN)")
    plt.title(f"Onde a não-invariância se concentra — {len(regions)} regiões")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "fig_border_pooled.png", dpi=150); plt.close()

    # ===================================================================
    # 3. Blends  (chave = blend)
    # ===================================================================
    bl = {}  # region -> {blend: row}
    have_blend = []
    for r in regions:
        p = res_root / r / "blend_compare" / "blend_metrics.csv"
        if p.exists():
            bl[r] = {x["blend"]: x for x in read_csv(p)}
            have_blend.append(r)
    pooled_bl = []
    if have_blend:
        blends = [x["blend"] for x in read_csv(
            res_root / have_blend[0] / "blend_compare" / "blend_metrics.csv")]
        for b in blends:
            present = [bl[r][b] for r in have_blend if b in bl[r]]
            se = [float(x["seam_excess"]) for x in present]
            pr = [float(x["psnr_ref"]) for x in present]
            ss = [float(x["ssim_ref"]) for x in present]
            pooled_bl.append({
                "blend": b, "n_regions": len(present),
                "seam_excess_mean": round(np.mean(se), 4),
                "seam_excess_std": round(np.std(se), 4),
                "psnr_ref_mean": round(np.mean(pr), 4),
                "ssim_ref_mean": round(np.mean(ss), 4),
            })
        with open(out / "blend_metrics_pooled.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pooled_bl[0].keys()))
            w.writeheader(); w.writerows(pooled_bl)

        names = [x["blend"] for x in pooled_bl]
        means = [x["seam_excess_mean"] for x in pooled_bl]
        stds = [x["seam_excess_std"] for x in pooled_bl]
        best = int(np.argmin(means))                  # menor seam_excess = melhor
        colors = ["tab:green" if i == best else "tab:blue" for i in range(len(names))]
        plt.figure(figsize=(7, 5))
        bars = plt.bar(names, means, yerr=stds, capsize=4, color=colors)
        for i, (m, s) in enumerate(zip(means, stds)):
            plt.text(i, m + s + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        plt.axhline(1.0, ls="--", color="k", lw=1, label="sem costura (ideal)")
        plt.ylabel("seam_excess (média entre regiões; verde = melhor)")
        plt.title(f"Visibilidade de costura por blend — {len(have_blend)} regiões")
        plt.ylim(0.95, max(means) + max(stds) + 0.05)
        plt.legend(); plt.tight_layout()
        plt.savefig(out / "fig_blend_pooled.png", dpi=150); plt.close()

    # ===================================================================
    # 4. Resumo legível
    # ===================================================================
    lines = [f"POOLED — {len(regions)} regiões: {', '.join(regions)}", ""]
    lines.append("Shift-consistency (média entre regiões):")
    lines.append(f"  {'overlap%':>9} {'scPSNR':>8} {'±dp':>6} {'SSIM':>6} {'±dp':>6}")
    for x in pooled_sc:
        lines.append(f"  {x['overlap_pct']:>9.1f} {x['psnr_mean']:>8.2f} "
                     f"{x['psnr_std']:>6.2f} {x['ssim_mean']:>6.3f} {x['ssim_std']:>6.3f}")
    if pooled_bl:
        lines += ["", "Blends (média entre regiões):",
                  f"  {'blend':>11} {'seam':>7} {'±dp':>6} {'PSNR':>7} {'SSIM':>6}"]
        for x in pooled_bl:
            lines.append(f"  {x['blend']:>11} {x['seam_excess_mean']:>7.3f} "
                         f"{x['seam_excess_std']:>6.3f} {x['psnr_ref_mean']:>7.2f} "
                         f"{x['ssim_ref_mean']:>6.3f}")
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nPooled salvo em: {out}")


if __name__ == "__main__":
    main()
