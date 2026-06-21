"""
sr_core.py — Núcleo compartilhado para os experimentos de invariância a translação
do gerador Satlas (ESRGAN / SSR_RRDBNet).

Diferente do `infer_geotiff.py` (que monta um mosaico SR completo), este módulo
expõe funções de baixo nível usadas pelos experimentos:

    1. load_stack()      — carrega N GeoTIFFs Sentinel-2, reprojeta para um ÚNICO
                           grid comum (EPSG:3857) e devolve um array fixo em memória.
    2. select_frames()   — escolhe os N_LR frames UMA ÚNICA VEZ (seleção global,
                           determinística). Isto remove o confound do
                           `_select_images` do infer_geotiff, que reescolhe frames
                           por tile e portanto pode mudar a saída só por causa da
                           posição do tile.
    3. build_model()     — instancia o SSR_RRDBNet e carrega os pesos.
    4. sr_window()       — roda SR em UMA janela LR arbitrária [y:y+tile, x:x+tile]
                           usando a seleção fixa de frames; devolve o tile HR.

A invariância a translação só é testável de forma limpa quando (a) os mesmos
frames alimentam todas as posições e (b) os deslocamentos são inteiros em pixels
LR (o upsampling nearest do gerador é equivariante a shifts inteiros: Δ px LR
→ 4·Δ px HR exatos).

Ambiente:
    SATLAS_REPO — caminho do repo allenai/satlas-super-resolution
                  (default: ~/Desktop/allenai/satlas-super-resolution)
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from pathlib import Path

SATLAS_REPO = Path(os.environ.get(
    "SATLAS_REPO",
    str(Path.home() / "satlas-sr")
))
sys.path.insert(0, str(SATLAS_REPO))

import torch
from ssr.utils.model_utils import build_network

# Constantes do modelo Satlas treinado em S2-NAIP
TARGET_RES = 9.555   # metros/pixel da entrada (grid S2-NAIP)
TILE_SIZE = 32       # tile de entrada LR (px)
SCALE = 4            # fator de super-resolução
OUTPUT_TILE_SIZE = TILE_SIZE * SCALE  # 128


# ---------------------------------------------------------------------------
# Carregamento do stack em grid comum
# ---------------------------------------------------------------------------
def load_stack(paths, target_res=TARGET_RES, reference=None):
    """Carrega N GeoTIFFs e reprojeta todos para o MESMO grid (EPSG:3857).

    Args:
        paths: lista de caminhos para GeoTIFFs Sentinel-2 (RGB nas bandas 1,2,3).
        target_res: resolução de saída em metros (default 9.555 m).
        reference: GeoTIFF opcional cujo grid (origem/extent) será reaproveitado,
                   garantindo snapping pixel-a-pixel.

    Returns:
        stack: np.ndarray uint8 [N, 3, H, W] — todas as cenas no grid comum.
        meta:  dict com transform, crs, width, height, actual_res.
    """
    dst_crs = rasterio.crs.CRS.from_epsg(3857)

    if reference is not None:
        with rasterio.open(reference) as ref:
            dst_transform = ref.transform
            width, height = ref.width, ref.height
            target_res = ref.transform.a
    else:
        all_bounds = []
        for p in paths:
            with rasterio.open(p) as src:
                if src.crs.to_epsg() != 3857:
                    t, w, h = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds,
                        resolution=(target_res, target_res))
                    left, top = t.c, t.f
                    all_bounds.append((left, top - h * target_res,
                                       left + w * target_res, top))
                else:
                    all_bounds.append(tuple(src.bounds))
        left = min(b[0] for b in all_bounds)
        bottom = min(b[1] for b in all_bounds)
        right = max(b[2] for b in all_bounds)
        top = max(b[3] for b in all_bounds)
        width = int(np.ceil((right - left) / target_res))
        height = int(np.ceil((top - bottom) / target_res))
        right = left + width * target_res
        bottom = top - height * target_res
        dst_transform = from_bounds(left, bottom, right, top, width, height)

    stack = np.zeros((len(paths), 3, height, width), dtype=np.uint8)
    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=stack[i],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    meta = {"driver": "GTiff", "dtype": "uint8", "count": 3,
            "width": width, "height": height, "crs": dst_crs,
            "transform": dst_transform, "actual_res": float(target_res)}
    return stack, meta


# ---------------------------------------------------------------------------
# Seleção fixa de frames (uma vez, global)
# ---------------------------------------------------------------------------
def select_frames(stack, n_lr_images):
    """Escolhe n_lr_images frames UMA vez para todo o experimento.

    Critério: menor fração global de pixels nodata (== 0). Determinístico.
    Garante que toda janela/posição use exatamente os mesmos frames — condição
    necessária para medir invariância do MODELO e não da seleção de entrada.
    """
    n = stack.shape[0]
    if n < n_lr_images:
        raise ValueError(f"Modelo espera {n_lr_images} frames, há apenas {n}.")
    nodata_frac = [(stack[j] == 0).mean() for j in range(n)]
    order = np.argsort(nodata_frac)  # menos nodata primeiro
    return sorted(order[:n_lr_images].tolist())


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def build_model(weights, n_lr_images, device):
    opt = {"scale": SCALE, "n_lr_images": n_lr_images,
           "network_g": {"type": "SSR_RRDBNet", "num_in_ch": n_lr_images * 3,
                         "num_out_ch": 3, "num_feat": 64, "num_block": 23,
                         "num_grow_ch": 32}}
    model = build_network(opt)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state["params_ema"], strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# SR de uma janela arbitrária
# ---------------------------------------------------------------------------
def sr_window(model, stack, frame_idx, y, x, device, tile=TILE_SIZE):
    """Roda SR na janela LR [y:y+tile, x:x+tile] com os frames fixos `frame_idx`.

    Returns:
        np.ndarray float32 [3, tile*SCALE, tile*SCALE] em [0,1].
    """
    patch = stack[frame_idx, :, y:y + tile, x:x + tile]  # [N,3,t,t]
    tensor = torch.from_numpy(patch).float().to(device) / 255.0
    tensor = tensor.reshape(1, -1, tile, tile)
    with torch.no_grad():
        out = model(tensor)
    return torch.clamp(out, 0, 1).squeeze(0).cpu().numpy()
