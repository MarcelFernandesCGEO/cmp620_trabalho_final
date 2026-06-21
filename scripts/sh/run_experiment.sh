#!/usr/bin/env bash
# Roda o pipeline completo de invariância a translação no servidor (com GPU).
#
# No servidor (~/cmp620_final):
#   bash scripts/sh/run_experiment.sh
#
# Variáveis de ambiente opcionais (defaults já apontam para o servidor):
#   SATLAS_REPO  — caminho do repo allenai (default: ~/satlas-sr)
#   WEIGHTS      — caminho dos pesos      (default: ~/weights/esrgan_8S2.pth)
#
# Pré-requisitos (instalar uma vez):
#   pip install torch rasterio scikit-image matplotlib
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGES=(dados/sentinel/sentinel_original/clipped/*.jp2)
SATLAS_REPO="${SATLAS_REPO:-$HOME/satlas-sr}"
WEIGHTS="${WEIGHTS:-$HOME/weights/esrgan_8S2.pth}"
N_LR="${1:-8}"
GPU="${2:-0}"                          # segunda posição: índice da GPU (default 0)
export CUDA_VISIBLE_DEVICES="$GPU"

export SATLAS_REPO

# --- validações rápidas antes de rodar ---
if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERRO: pesos não encontrados em $WEIGHTS" >&2; exit 1
fi
if [[ ! -d "$SATLAS_REPO/ssr" ]]; then
    echo "ERRO: SATLAS_REPO inválido ou sem subpasta ssr/: $SATLAS_REPO" >&2; exit 1
fi
if ! compgen -G "dados/sentinel/sentinel_original/clipped/*.jp2" > /dev/null 2>&1; then
    echo "ERRO: nenhuma imagem .jp2 em dados/sentinel/sentinel_original/clipped/" >&2
    echo "  Rode primeiro: python scripts/clip_to_aoi.py ..." >&2; exit 1
fi

echo "[1/2] Experimento de consistência sob deslocamento..."
python scripts/shift_consistency.py \
    --images "${IMAGES[@]}" \
    --weights "$WEIGHTS" \
    --n_lr "$N_LR" \
    --outdir resultados/shift_consistency

echo "[2/2] Comparação de estratégias de blend..."
python scripts/blend_compare.py \
    --images "${IMAGES[@]}" \
    --weights "$WEIGHTS" \
    --n_lr "$N_LR" \
    --profile resultados/shift_consistency/border_profile.npy \
    --outdir resultados/blend_compare

echo "Concluído. Veja resultados/."
