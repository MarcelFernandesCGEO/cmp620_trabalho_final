# Revisão da apresentação — perguntas e respostas da banca

Trabalho final CMP620 — *Invariância a Translação na Super-Resolução por Tiles:
Análise do Gerador Satlas e Estratégia Ótima de Junção*. Autor: Marcel Fernandes Gomes.

Apresentação ~10 min + 5 min de perguntas. Abaixo, perguntas prováveis da banca
agrupadas por tema, com respostas curtas (o que dizer) e o respaldo nos dados.
Números conferidos contra `resultados/_pooled/` e `resultados/_pooled_blend_extra/`
(57 regiões) em 2026-06-21.

---

## 1. Conceito e teoria

**P1. Qual a diferença entre equivariância e invariância à translação, e qual delas você mede?**
Equivariância: deslocar a entrada desloca a saída do mesmo jeito (`f(T_Δ x) = T_{sΔ} f(x)`).
Invariância: a saída não muda quando a entrada se desloca. As *camadas* convolucionais são
formalmente equivariantes; a CNN completa, por causa de padding e subamostragem, não é
estritamente nem uma coisa nem outra. Eu meço o desvio da **equivariância**: se ela valesse,
as duas inferências do mesmo conteúdo físico (com o tile deslocado) coincidiriam exatamente
na sobreposição. A discordância que mido é o quanto essa equivariância formal se quebra.

**P2. Por que uma CNN deixa de ser equivariante se as convoluções são equivariantes?**
Três motivos clássicos: (i) **zero-padding** nas bordas quebra a simetria de translação —
um pixel perto da borda "vê" zeros que um pixel central não vê; (ii) **subamostragem/aliasing**
(no nosso caso o upsampling); (iii) operações que dependem de posição absoluta. No Satlas, a
causa dominante é o padding combinado com um **campo receptivo maior que o tile**.

**P3. Como isso se conecta com o que foi visto em aula?**
É verificação empírica direta da Aula 06, §8.3 ("Equivariância à translação"). Mobiliza
ainda: padding "same"/zero-padding (§6), campo receptivo efetivo ampliado pela profundidade
(§5, §7), e weight sharing/equivariância do upsampling *nearest* (§3, §8.3). Secundariamente,
GANs/SR como modelo generativo (Aula 23) e perda perceptual (Aula 22), por o ESRGAN usá-las.

---

## 2. Metodologia e protocolo

**P4. Por que os deslocamentos são inteiros em pixels LR?**
Porque o upsampling é `F.interpolate(mode='nearest')` ×2 duas vezes (×4). Nearest é
equivariante para deslocamentos **inteiros**: Δ px LR → 4·Δ px HR exatos. Com Δ inteiro, o
grid HR alinha sem reamostragem, então o upsampling **não** entra como fonte de erro. Se eu
usasse deslocamento fracionário, precisaria reamostrar e introduziria um confound. Assim,
qualquer discordância que sobra vem do modelo, não do alinhamento.

**P5. Como você garante que está comparando o mesmo conteúdo físico?**
As 8 cenas são reprojetadas **uma vez** a um grid comum (EPSG:3857 @ 9,555 m). Para um
deslocamento Δ, rodo SR no offset 0 e no offset Δ; a região sobreposta Ω contém exatamente
os mesmos pixels do terreno (porque Δ px LR = 4Δ px HR exatos). Comparo as duas saídas SR
só dentro de Ω.

**P6. Por que essa métrica dispensa verdade de campo (GT)?**
É **auto-consistência**: comparo duas saídas do *próprio* modelo entre si, não contra uma
referência externa. Se o modelo fosse equivariante, elas seriam idênticas em Ω
independentemente de qualquer GT. O scPSNR/SSIM mede a discordância entre as duas saídas.

**P7. O que valida que o protocolo está correto?**
O **controle Δ=0** (sobreposição 100%): aí Ω é o tile inteiro e as duas saídas são a mesma
inferência, então scPSNR ≈ 100 dB e SSIM = 1,000. Isso confirma que a queda observada para
Δ>0 é efeito real do deslocamento, não bug de alinhamento ou de métrica.

**P8. O Satlas é multi-frame — como as 8 cenas entram?**
O gerador recebe as 8 cenas empilhadas (8×3 = 24 canais) numa janela 32×32 e produz **uma**
saída SR 128×128, explorando informação sub-pixel entre as datas. No experimento as 8 cenas
são uma **entrada fixa**; o que varia é só a **posição do tile**. A seleção de frames é
fixada globalmente (determinística, por menor nodata) para que a diferença entre offsets
venha do modelo, não de troca de input.

**P9. Por que restringir ao Sentinel-2?**
É o domínio de treino do Satlas-SR. Usar outro sensor introduziria *domain shift*, que se
confundiria com o efeito de translação que quero isolar. Mantendo S2, o que mido é
puramente (não-)equivariância, sem viés multissensor.

---

## 3. Resultados — consistência e perfil de borda

**P10. Qual é o resultado principal, em uma frase?**
O gerador **não** é invariante a translação: a auto-consistência cai monotonicamente de
**30,8 dB (90% de sobreposição) para 24,7 dB (3%)**, e o SSIM de 0,85 para 0,45, com baixa
dispersão (~±1,2 dB) entre as 57 regiões.

**P11. Por que a discordância cresce quando a sobreposição diminui?**
Porque sobreposição menor = região comparada mais próxima das **bordas** dos tiles. O erro
de equivariância se concentra na borda (perfil cai de 10,1 DN na borda para ~2,0 DN no
interior). Menos sobreposição → a comparação cai justamente onde o erro é maior.

**P12. Por que mesmo a 90% de sobreposição não há coincidência perfeita?**
Porque **não existe interior verdadeiramente limpo**: o campo receptivo (~350 convoluções
3×3, centenas de px LR) excede o tile de 32×32, então *todo* pixel de saída integra, em
algum grau, o zero-padding da borda. O erro só diminui, não zera.

**P13. O perfil de borda é a "prova" da causa? Como?**
Sim. Ele localiza onde nasce a não-invariância: máxima exatamente em d=0 (a borda) e
decaimento monotônico para dentro. Isso casa com o mecanismo previsto pela teoria —
o padding entra pelo campo receptivo e sua influência relativa diminui com a profundidade.
O perfil é ao mesmo tempo a *explicação* e o *insumo* da máscara de blend.

**P14. Os números são robustos ou de uma área sortuda?**
Robustos: 57 regiões geograficamente distintas, agregadas em pooled. O desvio-padrão entre
regiões fica ~±1,2 dB no scPSNR em todas as faixas e ~±0,04 no seam_excess dos blends. O
ranking não muda entre áreas, o que qualifica o efeito como propriedade do **gerador**, não
da paisagem.

---

## 4. Resultados — blends e mosaico

**P15. Por que o blend gaussiano vence entre os clássicos?**
Porque seu decaimento suave casa com a forma do perfil de erro (alto na borda, baixo no
interior): ele dá pouco peso à borda ruim e muito ao interior confiável, sem introduzir o
ruído de amostragem do perfil empírico. Resultado pooled: seam_excess 1,047 (ideal 1,0),
PSNR_ref 34,38, SSIM_ref 0,959 — melhor em todos os eixos entre none/linear/cosine/gaussian.

**P16. Por que o data-driven cru NÃO superou o gaussiano, se é "guiado pelos dados"?**
Achado honesto: o perfil empírico `P(d)` carrega **ruído de amostragem**, e usar `1/P(d)`
direto como peso propaga esse ruído para a máscara (seam 1,093, PSNR 32,41 < gaussiano). A
desvantagem era de *ruído*, não de *princípio* — ao suavizar/parametrizar o perfil
("smoothed"), a máscara guiada por dados passa a superar a gaussiana (PSNR 34,85, o maior de
todos). Ou seja: o princípio estava certo, faltava regularizar.

**P17. O que é o seam_excess e por que o ideal é ≈ 1?**
É a razão entre o gradiente médio nas linhas de junção e o gradiente médio global. Se a
costura é invisível, a junção tem o mesmo "nível de detalhe" do resto → razão ≈ 1. Acima de
1 há excesso de gradiente na costura (degrau visível); `none` dá 1,303 (≈ +30%).

**P18. O Poisson dá seam_excess 0,979, abaixo de 1. Isso é bom ou ruim?**
Abaixo de 1 significa que a junção ficou **mais suave** que o conteúdo médio — não há
degrau, e a reconstrução em domínio de gradiente integra a transição. É o melhor em costura
e em SSIM (0,988). O detalhe técnico importante: usei condição de **Dirichlet** (ancorada ao
composto de referência) para fixar a intensidade; a variante de Neumann derivava o tom e
caía a ~26 dB de PSNR.

**P19. E o shift-ensemble?**
Promedia inferências em vários offsets inteiros — cada pixel é coberto por tiles em posições
relativas distintas, e os artefatos de borda (que dependem da posição) **se cancelam**. Usa
a própria não-invariância a favor. Atinge costura quase ideal (0,999), mas perde nitidez
(PSNR 28,07) e é melhor avaliado por shift-consistency do que por fidelidade a uma referência
nítida. Custa N inferências.

**P20. Qual é a recomendação prática de montagem de mosaico?**
Usar **sobreposição não-nula** entre tiles, **feathering gaussiano** na transição e
**descartar a faixa de borda** (~15–20 px HR) onde o erro de equivariância é máximo. Se
quiser o melhor possível, Poisson/Dirichlet ou data-driven suavizado superam o gaussiano.

---

## 5. Limitações e escolhas de projeto

**P21. É zero-shot — você não treinou nada. Isso é suficiente para um trabalho?**
Sim, é uma **avaliação de modelo pré-treinado gerando resultados experimentais** (exatamente
o que os guidelines pedem). O objetivo não é treinar, é *diagnosticar* uma propriedade do
modelo (equivariância) e derivar uma consequência prática (blend). O experimento é controlado
(confounds de upsampling e seleção de frames removidos) e validado em 57 regiões.

**P22. Por que só deslocamento horizontal? E vertical/diagonal?**
Por simetria da arquitetura (convs isotrópicas, padding igual nos quatro lados), o efeito de
borda é o mesmo nas duas direções; o horizontal já cobre o mecanismo. Estender a vertical e
diagonal é trabalho futuro barato, mas não muda a conclusão.

**P23. O modelo é de 35 cm? Qual a escala real?**
Não. Entrada S2 a ~10 m/px; no grid comum, ~9,555 m/px LR; saída ×4 → ~2,4 m/px HR. A GT de
35 cm aparece só como referência secundária *opcional* (degradada), e o coração do trabalho
**dispensa GT**.

**P24. Por que 9,555 m/px e EPSG:3857?**
EPSG:3857 (Web Mercator) é o grid que o Satlas usa; 9,555 m/px é a resolução nativa do nível
de zoom correspondente. Reprojetar tudo para esse grid comum garante alinhamento
pixel-a-pixel entre cenas e entre offsets — pré-requisito para Δ inteiro funcionar.

**P25. As 57 regiões são independentes? Não há autocorrelação espacial?**
São folhas distintas da articulação cartográfica do Sul do Brasil, sem sobreposição entre
AOIs. Pode haver alguma similaridade de bioma, mas a baixa dispersão entre elas *fortalece* a
conclusão (o efeito não depende da paisagem). Generalizar a outros biomas/sensores/escalas é
trabalho futuro declarado.

**P26. Por que não atacar a causa no modelo (anti-aliasing/finetuning)?**
É a direção futura que aponto. Citei BlurPool (Zhang, ICML 2019) para atacar o aliasing na
origem e proponho um finetuning com perda explícita de consistência de borda
`L_borda = ||∇I_A − ∇I_B||` nas junções. Mas isso exige treino; o foco aqui foi diagnosticar
e mitigar em tempo de inferência (blend), que é zero-custo de treino.

---

## 6. Perguntas "difíceis" / de fundamentação

**P27. Como você sabe que o efeito não é da seleção de frames mudando entre offsets?**
Porque a seleção é **fixada globalmente** uma vez (determinística). Toda posição usa
exatamente os mesmos 8 frames. Se eu deixasse a seleção original (por tile, baseada em
nodata), a diferença poderia vir do input — por isso controlei esse confound explicitamente.

**P28. O campo receptivo é "centenas de px LR" — como você estima isso?**
São 23 blocos RRDB, cada um com 3 RDBs × 5 convs 3×3 → ~350 convoluções 3×3 com padding
"same". Cada conv 3×3 soma ±1 px de raio por camada; empilhadas, o campo receptivo teórico
chega à ordem de centenas de px LR — muito acima do tile de 32. O perfil de borda, que
mostra contaminação até dezenas de px de profundidade, é a evidência empírica disso.

**P29. PSNR e SSIM medem a mesma coisa? Por que reportar os dois?**
Não. PSNR é fidelidade ponto-a-ponto (sensível a erro absoluto); SSIM mede similaridade
estrutural (luminância, contraste, estrutura local). Reporto os dois porque a costura pode
ser invisível em PSNR mas visível em estrutura, e vice-versa. Os dois caem juntos aqui, o que
reforça a conclusão.

**P30. Qual é, afinal, a contribuição original?**
Três coisas: (i) um **protocolo de shift-consistency sem GT** específico para SR por tiles;
(ii) um **diagnóstico que localiza** a não-invariância na borda e a liga ao zero-padding sob
campo receptivo excessivo; (iii) uma **receita de junção** validada (gaussiano entre
clássicos; Poisson/Dirichlet e data-driven suavizado superando-o), tudo robusto em 57 regiões.

---

## 7. Checklist rápido antes de apresentar

- [ ] Nome do autor padronizado: **Marcel Fernandes Gomes** (artigo e slides).
- [ ] Tabelas do artigo/slides = CSVs pooled (conferido 2026-06-21).
- [ ] Saber de cor os 3 números-âncora: **30,8→24,7 dB** (consistência), **10,1→2,0 DN**
      (borda), **1,30→1,05** seam (gaussiano).
- [ ] Saber explicar o controle Δ=0 (validação do protocolo).
- [ ] Ter clara a frase-causa: *campo receptivo ≫ tile + zero-padding*.
- [ ] Mosaicos `none` vs `gaussian` lado a lado (slide de recomendação) prontos.
- [ ] Tempo: ~10 min → ~1 min por slide (9 de conteúdo + título).
