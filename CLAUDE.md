## Objetivo

Trabalho final da disciplina CMP620.

**Tema:** Investigar a **invariância a translação (shift-equivariance)** do gerador de
super-resolução do Satlas e seu impacto na junção de tiles em áreas maiores.

O modelo Satlas (ESRGAN / `SSR_RRDBNet`) recebe tiles de tamanho fixo (32×32 LR → 128×128 HR,
fator 4×), mas **não fixa a posição** do tile. A hipótese: se o modelo fosse perfeitamente
equivariante a translação, rodar SR do mesmo conteúdo físico com o tile deslocado deveria
produzir, na região sobreposta, **exatamente os mesmos pixels**. Onde houver divergência,
investigamos a causa e derivamos o **melhor blend** possível para montar mosaicos sem costura.

Análise restrita ao **Sentinel-2** (domínio de treino do Satlas) — sem viés de domain-shift
multissensor. A frente multissensor anterior (Planet/CBERS) foi descontinuada; seus dados
foram preservados localmente em `dados/_legacy_multissensor/` (fora do git).

---

## Hipótese e fundamentação técnica

O gerador `SSR_RRDBNet` (`ssr/archs/rrdbnet_arch.py` no repo allenai) define o experimento:

1. **Campo receptivo ≫ tile.** São `num_block: 23` blocos RRDB (cada um = 3 RDBs × 5 convs 3×3),
   ~350 convoluções 3×3 com padding "same". O campo receptivo fica na casa de centenas de
   pixels LR, enquanto o tile de entrada é só 32×32. **Consequência:** todo pixel de saída
   "enxerga" a borda do tile (zero-padding). Não existe interior limpo → esta é a causa
   principal da não-invariância e dos artefatos de junção. **É o achado central do trabalho.**

2. **Upsampling `F.interpolate(mode='nearest')` ×2 duas vezes (×4).** Nearest é equivariante
   para deslocamentos **inteiros** em LR: Δ px LR → 4·Δ px HR exatos. Logo, deslocando por
   pixels LR inteiros, o grid HR alinha perfeitamente e o upsampling **não** é fonte de erro
   (confound removido). **Por isso todos os deslocamentos são inteiros em px LR.**

3. **Confound da seleção de frames.** O `infer_geotiff.py` original reescolhe quais cenas usar
   por tile (via `_select_images`, baseado em nodata). Se a seleção mudar entre posições, a
   diferença vem do *input*, não do modelo. Os scripts novos usam **seleção global fixa**
   (`sr_core.select_frames`, calculada uma vez).

---

## Conexão com a disciplina (CMP620)

O trabalho é uma **verificação empírica direta de um conceito apresentado em sala** —
`resumos_claude/aula06_resumo_cnns_conv_layers_parametros.md`, seção 8.3
("Equivariância à translação"):

> "Camadas convolucionais são **equivariantes à translação**: deslocar o padrão na
> entrada desloca o padrão na saída. CNNs completas (com pooling) **não são formalmente
> equivariantes**, mas são na prática invariantes..."

Esta é exatamente a hipótese sob teste. O experimento mede *quanto* essa equivariância
formal se quebra num gerador de SR real e *por quê*, mobilizando outros conceitos da
mesma aula:

- **Equivariância vs. invariância (Aula 06, §8.3):** num modelo idealmente equivariante,
  deslocar o tile de Δ px deslocaria a saída de 4·Δ px sem alterar os valores na
  sobreposição. O `shift_consistency.py` mede o desvio dessa propriedade.
- **Padding "same"/zero-padding (Aula 06, §6):** o preenchimento de zeros na borda é a
  fonte concreta da não-equivariância — o que o `border_profile` localiza
  quantitativamente.
- **Campo receptivo efetivo (Aula 06, §5 e §7; ampliado por pooling/profundidade):** com
  ~350 convoluções 3×3, o campo receptivo excede o tile 32×32, então o efeito de borda
  contamina toda a saída. É a explicação central do achado.
- **Weight sharing / equivariância do upsampling nearest (Aula 06, §3 e §8.3):**
  fundamenta por que deslocamentos *inteiros* em LR preservam o alinhamento HR.

Conexões secundárias: **GANs e super-resolução como modelo generativo**
(`aula23_resumo_generative_models.md`, que lista SR entre as aplicações de GANs — o
Satlas usa ESRGAN) e **perda perceptual** (`aula22_resumo_neural_style_transfer.md`),
componente da loss do ESRGAN. Estas devem aparecer na fundamentação do artigo; a
âncora principal, porém, é a **Aula 06**.

---

## Desenho do experimento

1. Construir **uma vez** o stack LR fixo (8 cenas S2 reprojetadas para o mesmo grid EPSG:3857).
2. Selecionar âncoras interiores sem nodata.
3. Para cada âncora e cada Δ ∈ {px LR inteiros}: rodar SR no offset 0 e no offset Δ (horizontal),
   alinhar na sobreposição (shift HR = 4·Δ) e medir a discordância.
4. Variar a sobreposição: 100% (Δ=0, controle → erro ~0), 90%, 80%, …, ~3%.
5. Construir o **perfil de erro vs distância à borda do tile** — diagnóstico que localiza a
   não-invariância e **define diretamente a máscara de blend ótima** (peso ∝ 1/erro).
6. Comparar blends (`none`/`linear`/`cosine`/`gaussian`/`data-driven`) num mosaico real pela
   métrica de costura.

### Métricas
- **Primária — Shift-Consistency PSNR + SSIM** na sobreposição entre as duas saídas SR
  (auto-consistência; não requer GT). Reporta também MAE/RMSE em DN.
- **Diagnóstica — perfil de erro vs distância à borda** (`border_profile`).
- **Costura — `seam_excess`** (gradiente nas linhas de junção / gradiente global; ideal ≈ 1)
  e fidelidade vs mosaico de referência por recorte central.

A GT (35 cm degradada) é apenas referência secundária opcional; o coração do trabalho dispensa GT.

---

## Pipeline end-to-end e o papel das 8 cenas

> **O Satlas é multi-frame.** Ele não processa 1 imagem por vez: recebe as **8 cenas
> empilhadas** (8×3 = 24 canais) numa janela 32×32 e produz **uma única** saída SR 128×128,
> explorando informação sub-pixel entre as datas. No experimento as 8 cenas são uma
> **entrada fixa** — o que varia é a **posição do tile**, não as imagens.

```
8 cenas S2 brutas (mesma região, datas diferentes)
   │  [1] clip_to_aoi.py — recorta cada cena à AOI (.gpkg)
8 cenas recortadas (clipped/)
   │  [2] sr_core.load_stack — reprojeta as 8 ao MESMO grid (EPSG:3857 @ 9,555 m)
stack fixo [8, 3, H, W]
   │  [3] sr_core.select_frames — fixa os 8 frames (determinístico; remove confound)
   ├──────────────┬───────────────┐
   ▼              ▼
 [4] shift_consistency.py     [5] blend_compare.py
   │                             │
   ▼                             ▼
 resultados/shift_consistency/  resultados/blend_compare/
   └──────────────┴──────► [6] preencher relatorio/artigo.tex
```

- **[4] `shift_consistency.py` (coração):** para cada janela-âncora, roda SR no offset 0 e no
  offset Δ (mesmo conteúdo, tile deslocado). Como Δ px LR = 4·Δ px HR, a sobreposição são os
  **mesmos pixels físicos** → mede a discordância (scPSNR/SSIM/MAE/RMSE) e o perfil de borda,
  variando a sobreposição de 100% a ~3%.
- **[5] `blend_compare.py`:** usa o `border_profile` para a máscara *data-driven* e compara
  blends num mosaico real, medindo `seam_excess`.

Com exatamente 8 cenas + `esrgan_8S2`, **as 8 alimentam toda posição**; a seleção fixa só
garante que o conjunto não muda entre offsets (do contrário, a diferença viria do input).

---

## Estrutura do projeto

```
trabalho_final/
├── CLAUDE.md
├── scripts/
│   ├── sr_core.py            ← núcleo: load_stack, select_frames, build_model, sr_window
│   ├── shift_consistency.py  ← EXPERIMENTO PRINCIPAL (consistência sob deslocamento)
│   ├── blend_compare.py      ← comparação de estratégias de blend + blend data-driven
│   ├── run_all_regions.py    ← orquestrador multi-região (por <codigo>, idempotente)
│   ├── pool_regions.py       ← sumarização pooled de todas as regiões → _pooled/
│   ├── blend_extra.py        ← blends EXPERIMENTAIS (smoothed/innercrop/tukey/ensemble/poisson)
│   ├── pool_blend_extra.py   ← pooling unificado blends clássicos+extras → _pooled_blend_extra/
│   ├── infer_geotiff.py      ← motor de SR de mosaico completo (reuso/baseline)
│   ├── clip_to_aoi.py        ← recorta cenas por AOI vetorial (.gpkg) — corte do pipeline
│   ├── clip_to_gt.py         ← (legado) recorte pelo bbox de um raster GT
│   ├── degrade_gt.py         ← degrada GT 35 cm → ~2,4 m (referência secundária)
│   ├── collect_metadata.py   ← gera .meta.json por imagem
│   └── sh/run_experiment.sh  ← roda shift_consistency + blend_compare no servidor
├── dados/
│   ├── bbox/                 ← <codigo>.gpkg (uma AOI por região)
│   ├── sentinel/sentinel_original/         ← POOL FLAT de cenas S2 TCI (nomes originais)
│   ├── sentinel/sentinel_original/clipped/ ← recorte por região: clipped/<codigo>/
│   ├── sentinel/sentinel_sr/               ← saídas SR de mosaico
│   └── _legacy_multissensor/ ← FORA do git: Planet/CBERS + GT (gt/) — referência opcional
├── resultados/              ← saídas por região: <codigo>/{shift_consistency,blend_compare}/ + _pooled/
├── relatorio/              ← artigo IEEE (esqueleto)
└── templates/, guidelines_padrao.pdf
```

`.gitignore` exclui `*.tif/*.tiff/*.jp2/*.aux.xml`, `dados/_legacy_multissensor/` e `resultados/**/*.npy`.

---

## Zero-shot: insumos e organização

"Zero-shot" = inferência com os pesos pré-treinados, **sem qualquer retreino/finetuning**
(é o que todo o pipeline faz). Lista do que é preciso e como organizar.

> **GT não é necessário.** A métrica do experimento é de **auto-consistência** (duas saídas
> SR da mesma área comparadas entre si, não contra uma verdade externa). O GT só serve, e de
> forma **opcional**, para uma checagem secundária "vs verdade". O recorte é feito por uma
> **geometria vetorial (`.gpkg`)**, não por um raster de referência.

### Insumos necessários
1. **Modelo + pesos.** Repo allenai em `$SATLAS_REPO` (default `~/Desktop/allenai/satlas-super-resolution`)
   e os pesos `pretrained_models/esrgan_8S2.pth` (para `n_lr=8`) ou `esrgan_16S2.pth` (`n_lr=16`).
   **O nº de cenas deve casar com o peso** (8 ↔ `esrgan_8S2`, 16 ↔ `esrgan_16S2`).
2. **Cenas Sentinel-2.** Pelo menos `n_lr` cenas RGB da **mesma área** (mesmo tile MGRS),
   em datas distintas e com pouca nuvem. Produto **TCI** (3 bandas R,G,B, uint8, 10 m) —
   os scripts leem as bandas 1,2,3. Mais cenas que `n_lr` são aceitas (a seleção fixa
   escolhe `n_lr`); menos que `n_lr` gera erro.
3. **AOI vetorial (recorte).** Um **GeoPackage** (`.gpkg`) com a geometria da área de
   interesse — criado/escolhido por você no servidor e guardado em `dados/bbox/`. É o que
   define o recorte; **dispensa GT**. O CRS do `.gpkg` é reprojetado para o de cada cena
   antes do corte (ver "Exatidão de coordenadas").
4. **Deps Python (no servidor):** `torch rasterio scikit-image matplotlib` (`lpips` opcional).
5. **(Opcional) GT de alta resolução** — só se quiser métricas contra a verdade; o
   experimento de invariância **não** precisa de GT.

### Por que há etapa de corte (obrigatória)
As cenas Sentinel-2 baixadas vêm como *granule* completo (~10.980×10.980 px, ~110 km de
lado). O modelo opera em tiles de **32×32 px LR** (~305 m a 9,555 m/px). Rodar SR na cena
inteira é inviável e desnecessário: **recorta-se primeiro à AOI** (geometria `.gpkg`); em
seguida os scripts reprojetam para o grid comum (EPSG:3857 @ 9,555 m) e **fatiam em 32×32
automaticamente** (`sr_core.sr_window` / `tile_positions`). Não é preciso cortar manualmente
em 32×32.

### Tamanho da AOI — **não** é 32×32
O `.gpkg` define a **região**, não um tile. Deve ser **bem maior** que 32×32: o tile é
interno e fatiado automaticamente. A AOI precisa caber tile (32) + deslocamento máximo
(Δ até 31) + margem + **várias âncoras**. Se tivesse o tamanho de um tile, não haveria espaço
para deslocar nem para amostrar âncoras.

| Cenário | px LR (~9,555 m) | metros (lado) |
|---|---|---|
| Mínimo para rodar | ~128 px | ~1,2 km |
| **Recomendado** | **256–512 px** | **~2,5–5 km** |
| Mais âncoras (melhor estatística) | maior | (mais custo de GPU) |

Requisitos da AOI: terreno **homogêneo, sem nuvem/nodata em todas as 8 cenas**
(`pick_anchors` descarta janelas com pixel zerado); qualquer CRS (reprojetado no corte);
**não** precisa alinhar a múltiplos de 32. Limites de custo: `blend_compare.py --max_size`
(lado do mosaico de teste, px LR) e `shift_consistency.py --max_anchors` (nº de janelas).

### Exatidão de coordenadas
- A AOI vive no CRS do `.gpkg` (ex.: WGS84 ou UTM). No corte, a geometria é **reprojetada
  para o CRS de cada cena** e o recorte é feito por máscara da geometria — sem arredondar
  coordenadas no espaço errado.
- Para garantir **alinhamento pixel-a-pixel entre cenas e entre deslocamentos**, o
  `sr_core.load_stack` reprojeta tudo para um grid comum (EPSG:3857 @ 9,555 m). Use
  `--reference <uma cena>` para *snapar* o grid a uma origem fixa, evitando meio-pixel de
  offset entre cenas.
- Deslocamentos do experimento são **inteiros em px LR** (única forma de o grid HR alinhar
  sem reamostragem).

### Organização das pastas (entrada do zero-shot)
```
dados/bbox/<aoi>.gpkg                      ← geometria(s) da AOI (define o recorte) — você cria
dados/sentinel/sentinel_original/         ← cenas S2 BRUTAS baixadas (*.jp2/*.tif), ≥ n_lr
dados/sentinel/sentinel_original/clipped/ ← saída do corte → ENTRADA dos scripts
dados/sentinel/sentinel_sr/               ← saída SR de mosaico (infer_geotiff.py)
$SATLAS_REPO/pretrained_models/esrgan_8S2.pth ← pesos
(dados/_legacy_multissensor/gt/ ...)       ← opcional, só para checagem "vs verdade" (fora do git)
```

### Fluxo do zero-shot
```bash
# 1. Recorte à AOI por geometria vetorial
python scripts/clip_to_aoi.py \
    --aoi    dados/bbox/<aoi>.gpkg \
    --input  dados/sentinel/sentinel_original/*.jp2 \
    --outdir dados/sentinel/sentinel_original/clipped/

# 2a. SR de mosaico (zero-shot puro, GeoTIFF georreferenciado a ~2,4 m)
python scripts/infer_geotiff.py \
    --images  dados/sentinel/sentinel_original/clipped/*.jp2 \
    --weights $SATLAS_REPO/pretrained_models/esrgan_8S2.pth \
    --output  dados/sentinel/sentinel_sr/sr_output.tif \
    --n_lr_images 8

# 2b. ou os experimentos de invariância (fatiam em 32x32 internamente)
bash scripts/sh/run_experiment.sh 8
```

---

## Scripts

### `sr_core.py`
Biblioteca compartilhada. `load_stack()` reprojeta N GeoTIFFs S2 para um grid comum;
`select_frames()` fixa os N_LR frames globalmente; `build_model()` carrega o ESRGAN;
`sr_window()` roda SR em uma janela LR arbitrária com seleção fixa.
Variável `SATLAS_REPO` aponta o repo allenai (default `~/Desktop/allenai/satlas-super-resolution`).

### `shift_consistency.py` — experimento principal
```bash
python scripts/shift_consistency.py \
    --images dados/sentinel/sentinel_original/clipped/*.jp2 \
    --weights $SATLAS_REPO/pretrained_models/esrgan_8S2.pth \
    --outdir resultados/shift_consistency
```
Saídas: `shift_metrics.csv`, `shift_metrics_raw.csv`, `border_profile.{csv,npy}`,
`fig_consistency.png`, `fig_border.png`, `fig_example.png`, `config.json`.

### `blend_compare.py`
```bash
python scripts/blend_compare.py \
    --images dados/sentinel/sentinel_original/clipped/*.jp2 \
    --weights $SATLAS_REPO/pretrained_models/esrgan_8S2.pth \
    --profile resultados/shift_consistency/border_profile.npy \
    --outdir resultados/blend_compare
```
Saídas: `blend_metrics.csv`, `mosaic_<blend>.png`, `mosaic_center_ref.png`, `fig_blend_seam.png`.

### `blend_extra.py` — estratégias EXPERIMENTAIS (não altera o `blend_compare`)
Script **separado e aditivo**: importa (sem modificar) as funções de mosaico/métrica do
`blend_compare.py` — então as métricas são **comparáveis** — e grava numa pasta própria
(`blend_extra/`), deixando o resultado de referência (`blend_compare/`) intacto. Estratégias:
- `smoothed` — data-driven com o **perfil de borda suavizado/monótono** (corrige o ruído
  que fez o data-driven cru perder para o gaussiano).
- `innercrop` — **descarta os k px de borda** (erro máximo) e remonta só os interiores;
  `k` derivado do perfil (`--crop_px` para fixar).
- `tukey` — janela de Tukey (topo plano + bordas em cosseno; `--tukey_alpha`).
- `ensemble` — **shift-ensemble**: SR em muitos offsets (grade densa, `--ens_stride`) e
  média por pixel; usa a não-invariância a favor (cancela o artefato de borda). Mais inferência.
- `poisson` — **gradient-domain**: reconstrução Poisson via DST (condição de **Dirichlet**
  ancorada ao center-ref) a partir de um campo de gradientes sem costura (gradiente do tile
  "dono" de cada pixel); ataca direto o `seam_excess` e a Dirichlet evita a deriva de tom
  (intensidade ancorada à borda). Requer `scipy`.
```bash
python scripts/blend_extra.py \
    --images dados/sentinel/sentinel_original/clipped/<codigo>/*.jp2 \
    --weights $WEIGHTS \
    --profile resultados/<codigo>/shift_consistency/border_profile.npy \
    --outdir  resultados/<codigo>/blend_extra
# --methods smoothed,innercrop,tukey,ensemble,poisson   (escolhe quais)
# --ens_stride 8   --crop_px N   --tukey_alpha 0.5
```
Saídas: `blend_extra_metrics.csv` (mesmas colunas do `blend_compare`), `mosaic_<m>.png`,
`fig_blend_extra_seam.png`. **Experimental**: validado por compilação; confirmar num run real.
Rodar em todas as regiões já processadas (não mexe no orquestrador):
```bash
for d in resultados/*/; do c=$(basename "$d"); [ "${c#_}" = "$c" ] || continue
  python scripts/blend_extra.py \
    --images dados/sentinel/sentinel_original/clipped/$c/*.jp2 --weights "$WEIGHTS" \
    --profile "$d/shift_consistency/border_profile.npy" --outdir "$d/blend_extra"; done
```
> O `blend_extra.py` **sempre sobrescreve** sua saída (sem skip). Para o poisson corrigido
> (Dirichlet) entrar no pooled, rode o loop com o `blend_extra.py` **já corrigido**.

### `pool_blend_extra.py` — ranking unificado (clássicos + extras)
Lê (só leitura) `blend_compare/` **e** `blend_extra/` de cada região e gera um ranking
pooled num diretório **novo e separado** (`_pooled_blend_extra/`), sem tocar no `_pooled`:
```bash
python scripts/pool_blend_extra.py
```
Saídas: `blend_all_pooled.csv` (média±dp por estratégia, todas), `fig_blend_all_pooled.png`
(clássicos azul, extras laranja, melhor verde), `summary.txt`. Só numpy + matplotlib.

### `clip_to_aoi.py`
Recorte por **geometria vetorial** (substitui o `clip_to_gt.py`).
```bash
python scripts/clip_to_aoi.py \
    --aoi    dados/bbox/<aoi>.gpkg \
    --input  dados/sentinel/sentinel_original/*.jp2 \
    --outdir dados/sentinel/sentinel_original/clipped/
# --bbox        recorta pelo bounding box da AOI (em vez da geometria exata)
# --layer NOME  escolhe a camada do gpkg (padrão: a primeira)
```
Lê a geometria do `.gpkg` (une múltiplas features), **reprojeta para o CRS de cada cena**
(`pyproj`), recorta (`rasterio.mask`) preservando CRS/resolução nativos. Sem `--outdir`,
salva ao lado da cena com sufixo `_clipped`. Garante exatidão de coordenadas sem depender de GT.

### `infer_geotiff.py`, `clip_to_gt.py` (legado), `degrade_gt.py`, `collect_metadata.py`
Utilitários da fase anterior. `clip_to_gt.py` (corte por raster) fica como legado —
preferir `clip_to_aoi.py`. `degrade_gt.py` só é necessário se for usar a checagem opcional
vs GT. Inalterados exceto o caminho default de `SATLAS_REPO`.

### `sh/run_experiment.sh [N_LR]`
Roda os dois experimentos em sequência **para UMA região** (entrada em `clipped/`).
**Executar no servidor com GPU** (este ambiente não tem torch e não roda inferência).
Para multi-região, use o orquestrador abaixo.

### `run_all_regions.py` — orquestrador multi-região
Processa **todas as regiões de uma vez**, por **casamento geográfico** (não por nome):

```
dados/bbox/<codigo>.gpkg                   ← uma AOI por região
dados/sentinel/sentinel_original/*.jp2     ← POOL FLAT de cenas (nomes originais)
```

Para cada `<codigo>.gpkg`, a rotina **reprojeta a AOI ao CRS de cada cena** e seleciona
as que **contêm a AOI** (cobertura ≥ `--min_cover`, default 0.99). Assim **uma mesma
cena serve a vários `.gpkg` sem ser duplicada**, e gpkg em EPSG:4326 + cena em UTM
funcionam (reprojeção embutida — reaproveita `clip_to_aoi.aoi_in_crs`). Com ≥ `n_lr`
cenas cobrindo a AOI, recorta e roda `shift_consistency` + `blend_compare`, salvando
**por região**:

```
dados/sentinel/sentinel_original/clipped/<codigo>/   ← recorte da região
resultados/<codigo>/shift_consistency/
resultados/<codigo>/blend_compare/
```

É **idempotente e não remove nada**: etapa com CSV de saída presente é **pulada**
(use `--force` para recomputar). Assim, basta **adicionar** novos `<codigo>.gpkg` (e,
se a área for nova, as cenas que a cobrem) e rodar de novo — o que já existe é preservado.
**Sem prefixo, sem duplicar cena.**

```bash
export SATLAS_REPO=/.../satlas-sr
python scripts/run_all_regions.py --weights /.../weights/esrgan_8S2.pth --n_lr 8
# --min_cover 0.99  fração mínima da AOI coberta pela cena (1.0 = contém 100%)
# --only A B        processa só os códigos A e B
# --force           recomputa etapas já feitas
# --skip_blend      roda só o shift_consistency
```

### `pool_regions.py` — sumarização pooled
Lê todas as `resultados/<codigo>/` e gera `resultados/_pooled/`:
`shift_metrics_pooled.csv`, `border_profile_pooled.csv`, `blend_metrics_pooled.csv`
(média±dp entre regiões), `fig_consistency_pooled.png`, `fig_border_pooled.png`,
`fig_blend_pooled.png` (curva por região + pooled) e `summary.txt`. Depende só de
numpy + matplotlib (sem torch); subpastas com prefixo `_` (ex.: `_pooled`) são ignoradas.
```bash
python scripts/pool_regions.py
```

> **Região 2970_3_SE.** Cenas com nomes originais (`T22JDM_*.jp2`) no pool flat; recorte
> em `clipped/2970_3_SE/` e resultados em `resultados/2970_3_SE/{shift_consistency,blend_compare}/`.
> O artigo e a apresentação apontam o `\graphicspath` para essa subpasta. Para uma nova
> região, basta colocar `<codigo>.gpkg` em `bbox/` (e, se a área for nova, as cenas que a
> cobrem no pool) e rodar `run_all_regions.py` — o casamento é geográfico, sem prefixo.

---

## Execução — onde e como

- **Aqui (este ambiente):** apenas edição/planejamento. Sem torch; nada de inferência roda aqui.
- **No servidor:** instalar deps (`torch torchvision` do índice cu12x + `rasterio
  scikit-image matplotlib pyproj fiona shapely kornia basicsr`; `lpips` opcional),
  exportar `SATLAS_REPO`/`WEIGHTS`, garantir os pesos `esrgan_8S2.pth`, e rodar
  `bash scripts/sh/run_experiment.sh` (ver "Ambiente do servidor" para as armadilhas).
- Copiar de volta `resultados/` (CSV + PNG) para preencher a tabela e as figuras do artigo.

### Ordem típica
```
1. clip_to_aoi.py       ← S2 brutas + AOI .gpkg → clipped/
2. shift_consistency.py ← clipped/ → resultados/shift_consistency/
3. blend_compare.py     ← clipped/ + border_profile.npy → resultados/blend_compare/
4. preencher artigo     ← CSV/PNG → relatorio/artigo.tex
```

---

## Status da execução — **zero-shot, multi-região (57 regiões)**

> **Concluída a validação multi-região.** Inferência com os pesos pré-treinados
> `esrgan_8S2.pth`, **sem qualquer retreino/finetuning**, sobre **57 AOIs**
> geograficamente distintas do Sul do Brasil. Os números abaixo são **pooled** (média ±
> dp entre regiões) e são os que constam no artigo e na apresentação. A região-piloto
> `2970_3_SE` (tile MGRS T22JDM) foi a validação inicial; os mosaicos ilustrativos do
> artigo/slides ainda apontam para ela via `\graphicspath`.

**Insumos do run (servidor com GPU Tesla V100-PCIE-32GB):**
- **Regiões:** 57 AOIs (`dados/bbox/<codigo>.gpkg`), articulação cartográfica do Sul do
  Brasil (códigos 2944_*, 2945_*, 2946_*, 2961_*, 2962_*, 2963_*, 2970_3_SE).
- **Cenas:** 8 cenas S2 TCI por região, reprojetadas a grid comum @ 9,555 m (EPSG:3857).
- **Config por região:** `tile=32`, `s=4`, `n_lr=8`, âncoras interiores sem nodata,
  Δ ∈ {0,3,6,10,13,16,19,22,26,29,31} px LR (sobreposição 100 %→3 %).

**Resultados pooled (fonte: `resultados/_pooled/` e `resultados/_pooled_blend_extra/`):**
- **Shift-consistency** (`shift_metrics_pooled.csv`): controle Δ=0 → scPSNR≈100 dB /
  SSIM=1,0 (valida o protocolo). Queda **monotônica**: 90,6 %→scPSNR **30,83** dB / SSIM
  **0,850**; 50 %→27,81 / 0,722; 3,1 %→**24,65 dB / 0,452**. Dispersão entre regiões
  ~±1,2 dB. O gerador **não** é invariante a translação.
- **Perfil de borda** (`border_profile_pooled.csv`): |diferença| média cai de **10,1 DN
  na borda** para ~2,0 DN a ~50 px de profundidade — a não-invariância se **concentra na
  borda** (campo receptivo ≫ tile + zero-padding), confirmando o achado central.
- **Blends clássicos** (`blend_metrics_pooled.csv`): sem blend `seam_excess=1,303`
  (PSNR_ref 25,26); melhor clássico **`gaussian`** (`seam_excess=1,047`, PSNR_ref
  **34,38**, SSIM_ref 0,959). O **data-driven cru** (1,093 / 32,41) **não** superou o
  gaussiano (perfil empírico ruidoso).
- **Blends extras** (`blend_all_pooled.csv`): **`poisson` (Dirichlet)** dá a menor costura
  (`seam_excess=0,979`, SSIM 0,988) e **`smoothed`** (data-driven suavizado) o maior PSNR
  (34,85) — ambos superam o gaussiano. `ensemble` chega a costura quase ideal (0,999) ao
  custo de nitidez (PSNR 28,07).

> **Consistência artigo↔dados verificada (2026-06-21):** todas as tabelas do
> `relatorio/artigo.tex` e da `apresentacao/apresentacao.tex` batem exatamente com os
> CSVs pooled acima. Ao reprocessar/adicionar regiões, regenerar os `_pooled*` e
> **reconferir** os números nas duas fontes LaTeX.

### Ambiente do servidor (reprodutível) — armadilhas resolvidas
- **`.venv` isolado** (servidor não aceita pip global). Torch **deve casar com a CUDA do
  driver**: `pip install torch torchvision --index-url .../whl/cu126` (driver 12.9 → wheel
  cu12x; o wheel default do PyPI vinha `cu130` e quebrava com "driver too old").
- Deps: `rasterio scikit-image matplotlib pyproj fiona shapely kornia basicsr`.
- **Patch do `basicsr`** (torchvision novo removeu `functional_tensor`):
  `sed -i 's/transforms.functional_tensor import rgb_to_grayscale/transforms.functional import rgb_to_grayscale/' .../basicsr/data/degradations.py`.
- Exportar caminhos absolutos: `export SATLAS_REPO=/.../satlas-sr` e
  `export WEIGHTS=/.../weights/esrgan_8S2.pth` (não confiar nos defaults; `HOME` pode
  divergir). O `run_experiment.sh` precisa de fim de linha **LF** (`sed -i 's/\r$//'`).
- Rodar os scripts Python **direto** com `--weights` explícito é o caminho mais robusto.

---

## Relatório — `relatorio/`

`artigo.tex` está **preenchido** (IEEE LaTeX, português, 5 páginas): tabelas e figuras com
os números pooled das 57 regiões; estrutura conforme guidelines (Introdução, Descrição do
Problema, Solução Proposta, Resultados, Conclusões, Bibliografia). Compila limpo com
`pdflatex artigo.tex` (rodar 2× para resolver as citações da `thebibliography` manual).
Classe/estilos: `IEEEtran.cls`, `IEEEtran.bst`, `IEEEtranS.bst`. A apresentação Beamer fica
em `apresentacao/apresentacao.tex` (autor: **Marcel Fernandes Gomes**, forma padronizada nas
duas fontes). Banca/Q&A preparados em `revisao_apresentacao.md`.

---

## Proveniência dos dados

`dados/sentinel/sentinel_original/provenance.json` — Copernicus Data Space (reproduzível pelo
nome do arquivo). `dados/_legacy_multissensor/gt/gt_original/provenance.json` — 1CGEO
(institucional, não público; GT é referência opcional, fora do git).

---

## Contexto Geral

- Disciplina: CMP620 — Prazo: 29/06/2026 23:59 (Moodle); apresentações 22/06 e 24/06/2026.
- Modelo base: Satlas Super-Resolution (allenai), treinado em Sentinel-2.
- Repo do modelo: `~/Desktop/allenai/satlas-super-resolution` (pesos em `pretrained_models/`).
- Templates IEEE em `templates/`; guidelines em `guidelines_padrao.pdf`.
