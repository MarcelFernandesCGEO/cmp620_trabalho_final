# Invariância a Translação na Super-Resolução por Tiles

Trabalho final da disciplina **CMP620** (PPGC/UFRGS) — análise da (não-)invariância a
translação do gerador de super-resolução do **Satlas** (ESRGAN / `SSR_RRDBNet`) sobre
imagens Sentinel-2, e derivação da estratégia ótima de junção de *tiles* em mosaicos.

Autor: **Marcel Fernandes Gomes** · marcelgfernandes@gmail.com

## O que há neste repositório

Este é o **pacote de reprodutibilidade** do artigo. Por não ser permitido redistribuir os
rasters Sentinel-2, as **imagens de satélite não estão versionadas** (`.tif/.jp2`) — apenas
os **metadados de proveniência** (que permitem o redownload no Copernicus Data Space), as
AOIs vetoriais, o código e todos os resultados.

```
relatorio/        artigo IEEE (artigo.tex + PDF) e classe/estilos IEEEtran
apresentacao/     slides Beamer (apresentacao.tex + PDF)
scripts/          pipeline completo (sr_core, shift_consistency, blend_compare, ...)
dados/bbox/       57 AOIs (<codigo>.gpkg)
dados/sentinel/   metadados (provenance.json / *.meta.json) — SEM os rasters
resultados/       saídas por região + _pooled/ e _pooled_blend_extra/ (CSV + figuras PNG)
CLAUDE.md         documentação técnica detalhada do projeto e do pipeline
revisao_apresentacao.md  perguntas e respostas preparadas para a banca
```

## Reprodução (resumo)

1. Baixar as cenas Sentinel-2 (TCI) indicadas em
   `dados/sentinel/sentinel_original/provenance.json` no Copernicus Data Space.
2. Recortar à AOI: `scripts/clip_to_aoi.py` (ou casar geograficamente com
   `scripts/run_all_regions.py`).
3. Rodar `scripts/shift_consistency.py` e `scripts/blend_compare.py` (GPU + PyTorch);
   pesos `esrgan_8S2.pth` de https://github.com/allenai/satlas-super-resolution.
4. Agregar com `scripts/pool_regions.py` e `scripts/pool_blend_extra.py`.

Detalhes completos de ambiente, parâmetros e armadilhas em [`CLAUDE.md`](CLAUDE.md).

## Principais resultados (pooled, 57 regiões)

- O gerador **não** é invariante a translação: a auto-consistência (scPSNR) cai de
  **30,8 dB** (90 % de sobreposição) para **24,7 dB** (3 %).
- O erro se **concentra na borda** do tile: diferença média de **10,1 DN** na borda →
  ~2,0 DN no interior (campo receptivo ≫ tile + *zero-padding*).
- Para junção: o *blend* **gaussiano** lidera entre os clássicos (seam 1,05; PSNR 34,4 dB);
  **Poisson/Dirichlet** e **data-driven suavizado** o superam.
