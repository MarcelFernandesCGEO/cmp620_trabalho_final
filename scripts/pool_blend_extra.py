"""
pool_blend_extra.py — Pooling das estratégias de blend EXTRAS + comparação com as
de referência, num RANKING UNIFICADO entre regiões.

NÃO altera nada já alcançado: lê (somente leitura)
    resultados/<codigo>/blend_compare/blend_metrics.csv        (none/linear/cosine/gaussian/data)
    resultados/<codigo>/blend_extra/blend_extra_metrics.csv    (smoothed/innercrop/tukey/ensemble/poisson)
e grava num diretório NOVO e separado (default resultados/_pooled_blend_extra/):
    blend_all_pooled.csv   média±dp por estratégia (todas), ordenado por seam_excess
    fig_blend_all_pooled.png
    summary.txt

Depende só de numpy + matplotlib (sem torch). Subpastas com prefixo `_` são ignoradas.

Uso:
    python scripts/pool_blend_extra.py
    python scripts/pool_blend_extra.py --resultados resultados --out resultados/_pooled_blend_extra
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# ordem de referência (controle + clássicos) e extras, p/ exibição estável
REF_ORDER = ["none", "linear", "cosine", "gaussian", "data"]
EXTRA_ORDER = ["smoothed", "innercrop", "tukey", "ensemble", "poisson"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect(res_root, fname_map):
    """Junta {método: {seam_excess:[...], psnr_ref:[...], ssim_ref:[...]}} por região.

    fname_map: lista de (subpasta, arquivo) a ler em cada região.
    """
    data = {}            # metodo -> {coluna: lista}
    regions_used = set()
    for d in sorted(res_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for sub, fname in fname_map:
            p = d / sub / fname
            if not p.exists():
                continue
            for row in read_csv(p):
                m = row["blend"]
                bucket = data.setdefault(m, {"seam_excess": [], "psnr_ref": [], "ssim_ref": []})
                bucket["seam_excess"].append(float(row["seam_excess"]))
                bucket["psnr_ref"].append(float(row["psnr_ref"]))
                bucket["ssim_ref"].append(float(row["ssim_ref"]))
                regions_used.add(d.name)
    return data, sorted(regions_used)


def main():
    ap = argparse.ArgumentParser(description="Pooling unificado de blends (clássicos + extras)")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--out", default="resultados/_pooled_blend_extra")
    args = ap.parse_args()

    res_root = ROOT / args.resultados
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    data, regions = collect(res_root, [
        ("blend_compare", "blend_metrics.csv"),
        ("blend_extra", "blend_extra_metrics.csv"),
    ])
    if not data:
        raise SystemExit(f"Nenhum blend_metrics encontrado em {res_root}/<codigo>/.")

    # ordena: referência primeiro, extras depois, demais ao fim
    order = [m for m in REF_ORDER + EXTRA_ORDER if m in data]
    order += [m for m in data if m not in order]

    rows = []
    for m in order:
        b = data[m]
        n = len(b["seam_excess"])
        rows.append({
            "blend": m, "n_regions": n,
            "seam_excess_mean": round(float(np.mean(b["seam_excess"])), 4),
            "seam_excess_std": round(float(np.std(b["seam_excess"])), 4),
            "psnr_ref_mean": round(float(np.mean(b["psnr_ref"])), 4),
            "ssim_ref_mean": round(float(np.mean(b["ssim_ref"])), 4),
        })

    with open(out / "blend_all_pooled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # figura: seam_excess por estratégia (referência azul, extras laranja, melhor verde)
    ranked = sorted(rows, key=lambda r: r["seam_excess_mean"])
    names = [r["blend"] for r in ranked]
    means = [r["seam_excess_mean"] for r in ranked]
    stds = [r["seam_excess_std"] for r in ranked]
    best = 0  # já ordenado por seam crescente
    colors = []
    for i, r in enumerate(ranked):
        if i == best:
            colors.append("tab:green")
        elif r["blend"] in EXTRA_ORDER:
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")
    plt.figure(figsize=(9, 5))
    plt.bar(names, means, yerr=stds, capsize=4, color=colors)
    for i, (mn, sd) in enumerate(zip(means, stds)):
        plt.text(i, mn + sd + 0.004, f"{mn:.3f}", ha="center", va="bottom", fontsize=8)
    plt.axhline(1.0, ls="--", color="k", lw=1, label="sem costura (ideal)")
    plt.ylabel("seam_excess (média entre regiões)")
    plt.title(f"Blends clássicos (azul) vs extras (laranja) — {len(regions)} regiões")
    plt.ylim(0.95, max(means) + max(stds) + 0.06)
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "fig_blend_all_pooled.png", dpi=150); plt.close()

    # resumo legível (ordenado por seam_excess)
    lines = [f"POOLED BLEND (clássicos + extras) — {len(regions)} regiões", "",
             f"  {'blend':>11} {'n':>3} {'seam':>7} {'±dp':>6} {'PSNR':>7} {'SSIM':>6}"]
    for r in sorted(rows, key=lambda x: x["seam_excess_mean"]):
        lines.append(f"  {r['blend']:>11} {r['n_regions']:>3} "
                     f"{r['seam_excess_mean']:>7.3f} {r['seam_excess_std']:>6.3f} "
                     f"{r['psnr_ref_mean']:>7.2f} {r['ssim_ref_mean']:>6.3f}")
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
