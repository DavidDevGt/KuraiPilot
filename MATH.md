# MATH — la matemática detrás de KuraiPilot

Este documento recorre **toda la matemática que el pipeline usa de verdad**, etapa por etapa (mismo orden que [docs/02-pipeline-spec.md](docs/02-pipeline-spec.md)), con las fórmulas exactas tal como están en el código — no la teoría general de cada técnica, sino la variante específica que quedó implementada y por qué. Es descriptivo, no normativo: si este documento y el código difieren, el código gana: actualizar este archivo cuando cambie.

Convención de notación: `luma`, `char_idx`, etc. son los nombres de variable reales del código. Los símbolos griegos (`σ`, `θ`, `π`) se usan cuando el propio código los usa en comentarios. Todo lo que sigue opera sobre arrays NumPy/CuPy completos (regla dura 1, ADR-006) salvo donde se dice explícitamente "secuencial".

## Índice

1. [Geometría de grilla y corrección de aspecto (E2)](#1-geometría-de-grilla-y-corrección-de-aspecto-e2)
2. [Luminancia perceptual (E2)](#2-luminancia-perceptual-e2)
3. [Saliencia — preprocesamiento y postprocesamiento (E3)](#3-saliencia--preprocesamiento-y-postprocesamiento-e3)
4. [Gamma y cuantización, con y sin saliencia (E4)](#4-gamma-y-cuantización-con-y-sin-saliencia-e4)
5. [Dithering: Bayer ordenado y Floyd-Steinberg (E6)](#5-dithering-bayer-ordenado-y-floyd-steinberg-e6)
6. [Refinamiento estructural — Sobel y octantes (E5)](#6-refinamiento-estructural--sobel-y-octantes-e5)
7. [Estabilidad temporal — histéresis y corte de escena (E7)](#7-estabilidad-temporal--histéresis-y-corte-de-escena-e7)
8. [Render: atlas, tintado y fg+bg (E8)](#8-render-atlas-tintado-y-fgbg-e8)
9. [Calibración de rampas y glifos de bloque](#9-calibración-de-rampas-y-glifos-de-bloque)
10. [Encode — color y tiempo exacto (E9)](#10-encode--color-y-tiempo-exacto-e9)
11. [Métricas de calidad: FCR y velocidad](#11-métricas-de-calidad-fcr-y-velocidad)
12. [Estadística del A/B de saliencia](#12-estadística-del-ab-de-saliencia)
13. [Complejidad por etapa](#13-complejidad-por-etapa)

---

## 1. Geometría de grilla y corrección de aspecto (E2)

`src/kurai/engine/grid.py`

La celda de referencia es 8×16 px — relación de aspecto 1:2 (un carácter de terminal es ~el doble de alto que ancho). Para que la grilla de `rows×cols` celdas reproduzca el aspecto del video original sin deformarlo:

```
aspecto_video = video_w / video_h
aspecto_celda = CELL_W / CELL_H = 8/16 = 0.5
```

Se quiere que `cols` celdas de ancho y `rows` celdas de alto cubran ese mismo aspecto en píxeles reales:

```
cols·CELL_W / (rows·CELL_H) ≈ video_w / video_h
```

Despejando `rows`:

```
rows = video_h / video_w · cols · (CELL_W / CELL_H)
```

que es exactamente `grid_shape()`:

```python
rows = round(video_h / video_w * cols * (CELL_W / CELL_H))
```

`terminal_grid()` (`src/kurai/engine/live.py`) reutiliza la misma fórmula porque una celda de terminal real también es ~1:2 — el mismo cálculo geométrico sirve para el export a píxeles y para el modo texto.

## 2. Luminancia perceptual (E2)

`src/kurai/engine/grid.py`, `LUMA_WEIGHTS`

La luminancia percibida no es el promedio RGB — el ojo pesa el verde ~10× más que el azul. Se usan los pesos BT.709 (el estándar del video HD, consistente con el `bt709` que se taguea en el encode, ver §10):

```
luma = 0.2126·R + 0.7152·G + 0.0722·B
```

**Identidad de linealidad** que la implementación explota: como `luma` es una combinación lineal de `(R,G,B)`, promediar los píxeles de una celda y luego calcular la luma del promedio da *exactamente* lo mismo que calcular la luma de cada píxel y promediar:

```
luma(mean(rgb_i)) = w·(1/n · Σ rgb_i) = 1/n · Σ (w·rgb_i) = mean(luma(rgb_i))
```

Por esto `to_grids()` reduce la celda a color promedio primero (una suma entera de una sola pasada, ~7× más rápido que promediar en float) y aplica los pesos de luma *después*, sobre la grilla ya chica — la regla "luma antes del promedio" del pipeline-spec solo prohíbe pesos no lineales (p. ej. una gamma aplicada antes de promediar sí cambiaría el resultado, y por eso la gamma se aplica recién en la Etapa 4, después de esta reducción).

## 3. Saliencia — preprocesamiento y postprocesamiento (E3)

`src/kurai/ai/saliency.py`

### 3.1 Normalización de entrada (ImageNet)

El frame RGB se reescala a `[0,1]` y se normaliza por canal con media/desvío de ImageNet — el dominio con el que U2Net-lite fue entrenado:

```
x' = (x/255 − μ) / σ
μ = (0.485, 0.456, 0.406)   σ = (0.229, 0.224, 0.225)   (orden RGB)
```

### 3.2 Resize bilineal (`_resize_bilinear`)

Convención de **centro de píxel** (la misma que `cv2.INTER_LINEAR`), no esquina de píxel — evita el corrimiento de medio píxel que produce bordes borrosos en direcciones opuestas para upscale/downscale:

```
src_y = (out_y + 0.5)·(in_h/out_h) − 0.5     (clampado a [0, in_h−1])
src_x = (out_x + 0.5)·(in_w/out_w) − 0.5
```

Con `y0=⌊src_y⌋`, `y1=min(y0+1, in_h−1)`, `wy = src_y − y0` (ídem en x), la interpolación bilineal es el lerp 2D estándar:

```
top    = (1−wx)·img[y0,x0] + wx·img[y0,x1]
bottom = (1−wx)·img[y1,x0] + wx·img[y1,x1]
out    = (1−wy)·top + wy·bottom
```

Se usa dos veces: para llevar el frame a 320×320 (entrada del modelo) y para llevar la máscara de salida de vuelta a `rows×cols` (resolución de grilla).

### 3.3 Blur gaussiano separable (`_gaussian_blur`)

Una frontera dura de densidad se ve peor que no tener saliencia — el post-proceso suaviza la máscara con un kernel gaussiano 1D de `σ=2` celdas, aplicado por filas y luego por columnas (la separabilidad de la gaussiana 2D en producto de dos 1D es lo que hace esto barato: `O(n·k)` en vez de `O(n·k²)`):

```
kernel[i] = exp(−i² / 2σ²) / Σ exp(−j² / 2σ²)     para i ∈ [−r, r], r = ⌈3σ⌉
blur(a) = conv1d(conv1d(a, kernel, eje filas), kernel, eje columnas)
```

Padding `reflect` en los bordes (no `zero`, que oscurecería artificialmente el borde de la grilla).

### 3.4 Normalización min-max

```
density = (mask − min(mask)) / (max(mask) − min(mask))
```

con caso especial: si el rango es `< 1e-6` (máscara plana) se clampa a `[0,1]` en vez de dividir por ~cero.

### 3.5 Scheduling temporal

No es una fórmula sino una política de muestreo: se infiere en `frame_idx % 5 == 0` o en corte de escena, y entre corridas se reutiliza (`copy()`) el último mapa — una forma barata de mantener consistencia temporal sin optical flow (eso es Fase 2/flow, todavía no implementado).

## 4. Gamma y cuantización, con y sin saliencia (E4)

`src/kurai/engine/mapping.py`

### 4.1 Gamma

Ley de potencia estándar, aplicada al espacio de luma perceptual:

```
luma_gamma = luma^γ          (γ = 0.8 en todos los presets shipeados)
```

`γ < 1` aclara las sombras (expande el rango bajo) antes de cuantizar — sin esto la rampa de caracteres concentra demasiada resolución en los tonos altos.

### 4.2 Cuantización base (sin saliencia)

```
idx = clip(⌊(luma_gamma + offset)·levels⌋, 0, levels−1)
```

`offset` es el ruido de dithering (Bayer o, con FS, ya absorbido en `luma_gamma` — ver §5).

### 4.3 Cuantización modulada por densidad (con saliencia, E3→E4)

Esta es la pieza matemática central de la Fase 1 (hoy con veredicto **NO-GO** en el A/B, ver §12, pero el mecanismo sigue en el código). La idea: en vez de cuantizar siempre a `levels` niveles, una celda de baja saliencia cuantiza a **menos** niveles (`eff < levels`) y el resultado se re-expande linealmente al rango completo — la rampa visual sigue siendo la misma, pero la celda usa menos caracteres *distintos*.

Niveles efectivos por celda, interpolación lineal entre el piso (`SALIENCY_MIN_LEVELS = 4`) y el techo (`levels`), según `density ∈ [0,1]`:

```
eff_min = min(4, levels)
eff     = round(eff_min + density·(levels − eff_min))     clampado a [eff_min, levels]
```

Cuantización gruesa a `eff` niveles:

```
coarse = clip(⌊(luma_gamma + offset)·eff⌋, 0, eff−1)
```

Re-expansión afín al rango completo `[0, levels−1]`, preservando los extremos (`0→0`, `eff−1→levels−1`) y la monotonía en luma:

```
scale = (levels−1) / max(eff−1, 1)
idx   = round(coarse·scale)
```

**Caso límite que garantiza degradación bit a bit**: con `density ≡ 1.0`, `eff = levels` en todas las celdas ⇒ `coarse = idx` de la fórmula base y `scale = 1` ⇒ el resultado es *exactamente* el de la Etapa 4 sin saliencia. Esto es lo que permite que "sin modelo" (§3, degradación) y "preset sin saliencia" produzcan el mismo golden — no una aproximación, una identidad algebraica (`tests/test_pipeline.py::test_density_uniform_is_noop`).

## 5. Dithering: Bayer ordenado y Floyd-Steinberg (E6)

`src/kurai/engine/dither.py`

### 5.1 Bayer 8×8 ordenado

Se parte de la matriz de Bayer canónica de índices `0..63` (construida por el algoritmo recursivo estándar de bit-reversal, hardcodeada como constante) y se convierte a offsets centrados en cero:

```
offset[i,j] = ((bayer[i,j] + 0.5) / 64 − 0.5) / levels
```

El `+0.5` evita que ningún índice caiga exactamente en 0 o 64 (offset simétrico); dividir por `levels` escala el patrón al ancho de un nivel de cuantización, así el dithering nunca puede saltar un nivel entero (ver `test_bayer_offsets_bounded_and_centered`: `|offset| ≤ 0.5/levels`). La matriz se tilea con `np.tile` a `(rows, cols)`.

Por construcción el patrón es *el mismo en todo frame* — offset determinista en función de `(i,j) mod 8`, nunca del contenido — de ahí "cero flicker inducido": una celda estática siempre recibe el mismo ruido de dithering.

### 5.2 Floyd-Steinberg serpentine

Única excepción a la vectorización estricta (ADR-006): es inherentemente secuencial porque el error de cada celda se difunde a las celdas *todavía no procesadas*. Kernel de difusión estándar (pesos que suman 1, en unidades de `/16`):

```
        celda   7/16
 3/16   5/16   1/16
```

En orden serpentine (fila 0 izquierda→derecha, fila 1 derecha→izquierda, alternando — `step = ±1` según paridad de fila), "adelante" y "atrás" se invierten cada fila, así que el kernel se refleja horizontalmente en las filas impares. Para cada celda, en orden de barrido:

```
k       = ⌊valor · levels⌋              (índice de bin, floor explícito — ver nota)
center  = (k + 0.5) / levels             (centro del bin elegido)
error   = valor − center
valor[adelante]          += 7/16 · error
valor[abajo, atrás]      += 3/16 · error
valor[abajo]              += 5/16 · error
valor[abajo, adelante]    += 1/16 · error
```

**Desviación deliberada del FS de manual**: el algoritmo de texto difunde el error contra el nivel de gris *cuantizado* (para binarización, 0 o 255); acá se difunde contra el **centro del bin**, no contra el valor bruto redondeado. La razón es numérica: `floyd_steinberg()` devuelve `center`, no `valor`, y el `quantize()` posterior vuelve a hacer `⌊center·levels⌋` — devolver el centro exacto garantiza que ese floor reproduzca el mismo `k` sin importar el redondeo de `double→float32` en el camino, cosa que devolver el `valor` ajustado (que podría caer justo en un borde de bin) no garantiza. Verificado por Hypothesis (`test_fs_returns_exact_bin_centers`) sobre miles de formas/niveles aleatorios.

El loop corre en `float64` de Python puro (no NumPy vectorizado — es la excepción admitida), determinista bit a bit porque IEEE-754 double es determinista entre máquinas para las mismas operaciones en el mismo orden.

**Advertencia documentada** (docs/02 E6): FS es sensible al ruido entre frames — al no tener patrón espacial fijo como Bayer, el mismo píxel puede recibir distinto error de difusión en frames consecutivos con ruido de cámara/codec mínimo, induciendo flicker. Por eso FS *siempre* se combina con histéresis (E7, §7) — de hecho la histéresis compara la luma **sin** dithering aplicado (ver `cells_to_charmatrix`), precisamente para no comparar dos difusiones de error distintas entre sí.

## 6. Refinamiento estructural — Sobel y octantes (E5)

`src/kurai/engine/edges.py`

### 6.1 Operador de Sobel 3×3

Kernels estándar aplicados por slicing vectorizado (sin scipy, sin loop por celda):

```
        -1  0  1              -1 -2 -1
Gx =    -2  0  2      Gy =     0  0  0
        -1  0  1               1  2  1
```

Escritos en el código como diferencias de columnas/filas pesadas 1-2-1:

```
gx = (top_right + 2·mid_right + bot_right) − (top_left + 2·mid_left + bot_left)
gy = (bot_left + 2·bot_center + bot_right) − (top_left + 2·top_center + top_right)
```

con padding `edge` (repite el borde) para que la grilla completa tenga vecinos válidos.

### 6.2 Magnitud y orientación

```
magnitude = √(gx² + gy²)            (np.hypot — estable numéricamente)
θ         = atan2(gy, gx)  ∈ (−π, π]
```

Para un borde de contraste pleno (salto 0→1 alineado con el eje), los tres términos de un lado valen 1 y los del otro 0 con pesos `1+2+1=4` ⇒ `magnitude ≈ 4`. El umbral por defecto `DEFAULT_THRESHOLD = 0.5` corresponde entonces a un salto local de luma de `0.5/4 = 0.125` — por debajo es textura tonal, por encima es un borde estructural. Empíricamente marca 5-20% de las celdas en contenido típico.

### 6.3 Cuantización angular a 8 octantes

```
octant = round(θ / (π/4)) mod 8
```

Cada uno de los 8 octantes de 45° mapea a uno de 4 glifos direccionales (`| / - \`), con antipodales compartiendo glifo salvo el horizontal (que distingue `-` de `_` para aprovechar la polaridad del gradiente). Tabla exacta en el código:

```
octante:   0    1    2    3     4    5    6    7
grados:    0   +45  +90  +135  180  -135 -90  -45
glifo:     |    /    -    \     |    /    _    \
```

El carácter final indexa `tonal_levels + offset_del_glifo` en el atlas — un espacio de índices completamente separado de la rampa tonal (§8).

## 7. Estabilidad temporal — histéresis y corte de escena (E7)

`src/kurai/engine/stability.py`

### 7.1 Histéresis por celda

Una celda solo adopta un char_idx nuevo si su luma se movió más que medio nivel de cuantización, ponderado por un factor empírico:

```
h = HYSTERESIS_FACTOR / levels     (HYSTERESIS_FACTOR = 1.5)
adopt(celda) = |luma[t] − luma_committed| > h
```

`luma_committed` es la luma del **momento del último cambio adoptado**, no la del frame anterior — comparar contra el frame anterior permitiría un drift lento (la celda se mueve de a poquito cada frame sin disparar nunca la histéresis, pero acumula un cambio grande a lo largo de muchos frames). Comparar contra el último valor *comprometido* corta ese drift acumulado.

`HYSTERESIS_FACTOR` pasó de 0.6 a 1.5 tras medir sobre metraje real con grano de película — el valor original solo se había validado contra un fixture sintético limpio, que no exponía el problema (ver CHANGELOG).

### 7.2 Detección de corte de escena

Heurística barata por diferencia de luma media global entre frames consecutivos:

```
scene_cut = |mean(luma[t]) − mean(luma[t−1])| > SCENE_CUT_THRESHOLD     (= 0.25)
```

En corte de escena, `luma_committed` se resetea completo (`fill(−1.0)`) — el siguiente frame se trata como si fuera el frame 0 (todas las celdas "frescas", adoptan sin importar histéresis).

## 8. Render: atlas, tintado y fg+bg (E8)

`src/kurai/render/atlas.py`

Todo el render es *indexado*, nunca dibujado: `atlas[char_idx]` es una operación de *gather* — aplicar la función `k ↦ bitmap_k` elemento a elemento sobre toda la grilla de una sola vez, en vez de evaluarla carácter por carácter (regla dura 7).

### 8.1 Tintado (mono y fg)

El atlas es binario, `bitmap ∈ {0, 255}`. Tintar es escalar cada canal de color por la fracción de tinta:

```
tinted[c] = atlas · color[c] // 255        (c ∈ {R,G,B}; división entera)
```

Con `atlas ∈ {0,255}` esto degenera a selección pura (`255·x//255 = x`, `0·x//255 = 0`) — está escrito como producto/escala, no como `where`, porque generaliza sin cambios si en el futuro el atlas deja de ser binario (p. ej. anti-aliasing de glifos).

### 8.2 fg+bg — mezcla de dos colores por celda

La misma fórmula, generalizada a dos fuentes de color en vez de una tinta + negro implícito — es exactamente **alpha compositing** discreto con el mask del glifo como canal alfa:

```
frame = (mask·fg + (255 − mask)·bg) // 255
```

Con `mask ∈ {0,255}` (glifo binario) esto también degenera a selección exacta: `frame = fg` donde hay tinta, `frame = bg` donde no la hay — pero la fórmula es la general de interpolación lineal (`lerp(bg, fg, mask/255)`), lista para admitir anti-aliasing sin cambiar la expresión.

`CharMatrix.bg` se llena en la Etapa 1/2: el decode entrega la grilla a `rows·2` (dos muestras verticales por celda — un "semi-bloque"); `fg` es el promedio de la mitad superior, `bg` el de la inferior (ver `split_half_cells()` en `pipeline.py`); el camino tonal (E4-E7) sigue corriendo sobre el **promedio de ambas mitades**, no sobre una sola.

## 9. Calibración de rampas y glifos de bloque

`src/kurai/render/glyphs.py`, `tools/calibrate_ramp.py`

### 9.1 Cobertura de tinta

```
ink_coverage(char) = Σ (bitmap(char) ≠ 0)      (píxeles encendidos, de 128 posibles en 8×16)
```

La rampa `short` está ordenada por esta métrica medida, no por intuición visual — el gate es que la secuencia sea **estrictamente creciente**:

```
ink_coverage(ramp[i]) < ink_coverage(ramp[i+1])    para todo i
```

Sin esto, dos caracteres consecutivos de la rampa podrían visualmente "empatar" o incluso invertirse en densidad percibida, rompiendo la monotonía luma→carácter que el resto del pipeline asume.

### 9.2 Densidad procedural de los glifos de bloque (`░▒▓█`)

Los 4 bloques Unicode no son bitmaps dibujados a mano — son patrones de dithering ordenado *dentro del propio glifo*, la misma idea de Bayer (§5.1) aplicada a calibrar densidades intermedias sobre un bitmap binario:

```
fill = 0.0  →  todo apagado
fill = 1.0  →  todo encendido
fill = 0.5  →  mask[i,j] = (i + j) mod 2 == 0                    (tablero de ajedrez, 50%)
fill = 0.25 →  mask[i,j] = (i mod 2 == 0) and ((j + i//2) mod 2 == 0)
fill = 0.75 →  ¬mask(0.25)                                       (patrón inverso)
```

## 10. Encode — color y tiempo exacto (E9)

`src/kurai/engine/encode.py`

### 10.1 Espacio de color BT.709

El contenido ASCII renderizado se etiqueta explícitamente como BT.709 en el stream de salida (`setparams=colorspace=bt709:...`), consistente con los mismos pesos de luma usados en la Etapa 2 (§2) — no son dos elecciones independientes, es el mismo estándar de principio a fin. La conversión RGB→YCbCr que hace `ffmpeg` internamente (`scale=out_color_matrix=bt709`) sigue la matriz estándar del estándar:

```
Y  =  0.2126R + 0.7152G + 0.0722B
Cb = (B − Y) / 1.8556
Cr = (R − Y) / 1.5748
```

Sin taguear esto explícitamente, `swscale` usa BT.601 por defecto y los reproductores HD asumen BT.709 al decodificar — el resultado es color visiblemente corrido (verificado y corregido en la auditoría de Fase 0, ver CHANGELOG).

### 10.2 Frame rate racional exacto

El framerate viaja como fracción exacta (`"30000/1001"`), nunca como float formateado. La razón es acumulación de error: si se usara el float redondeado `29.97`,

```
drift(N) = N · (1/29.97 − 1001/30000)
```

crece linealmente con `N` (cantidad de frames) — en un video de 2 horas a ~30 fps (`N ≈ 216 000`) esto es suficiente para desincronizar audio/video de forma perceptible. El racional exacto tiene drift = 0 por construcción.

## 11. Métricas de calidad: FCR y velocidad

### 11.1 Flicker Change Rate (docs/06 §3)

```
FCR = (cambios de char_idx por celda) / (tiempo, en celdas ESTÁTICAS solamente)
```

medido solo sobre celdas de ground-truth conocido (fixture sintético con máscara de movimiento conocida, o región anotada a mano en metraje real). Gate: `FCR ≤ 0.05` (un cambio cada 20 s por celda) con histéresis activa; el baseline sin anti-flicker es típicamente `> 2`. Un componente de estabilización nuevo (p. ej. optical flow, todavía no implementado) solo se acepta si mejora el FCR del fixture de paneo en `≥ 50%` sobre histéresis sola — si no, se descarta (Discernment, IDEA.md).

### 11.2 Speed factor

```
speed_factor = duración_del_video_de_entrada_s / tiempo_de_procesamiento_s
```

Gates por preset (docs/07): `retro ≥ 4×` (medido: 12.9× con NVENC), `detallado ≥ 2×` (medido: 3.6× GPU), `alta-fidelidad ≥ 0.8×`. `kurai bench --check` falla si cualquier preset cae `> 10%` respecto al último resultado aceptado — la regresión de performance es un fallo de test, no una nota aparte (docs/06 §5).

## 12. Estadística del A/B de saliencia

`tools/ab_review.py`, `docs/evaluations/2026-07-18-ab-saliencia.md`

El gate de Fase 1 exige `≥ 60%` de preferencia por `detallado` sobre un mínimo de evaluadores/sesiones:

```
preferencia = votos_detallado / (votos_detallado + votos_retro)      (empates excluidos)
```

El resultado real fue `preferencia = 1/7 ≈ 14.3%`, decidido con **1 sesión ciega** (n=7) en vez de las ≥3 que pide el protocolo — una desviación documentada. La justificación matemática de cerrar ahí: aunque `n=7` es una muestra chica, el error estándar de una proporción es

```
SE = √(p̂·(1−p̂) / n) = √(0.143·0.857 / 7) ≈ 0.132
```

y un intervalo aproximado al 95% (`p̂ ± 1.96·SE`, aproximación normal — imprecisa para `n` tan chico, pero suficiente como cota gruesa) da

```
[0.143 − 0.259, 0.143 + 0.259] ≈ [−0.12, 0.40]   →   clampado a [0, 0.40]
```

Incluso en el extremo superior generoso de esa cota, `0.40 < 0.60`: la distancia al umbral del gate es lo bastante grande como para que 2 sesiones más no fueran a cambiar la conclusión — por eso se cerró ahí en vez de completar el protocolo al pie de la letra. Es una decisión de producto informada por estadística, no la estadística reemplazando al protocolo (que sigue siendo lo correcto para un resultado cerca del umbral).

## 13. Complejidad por etapa

Con `N = rows·cols` celdas por frame y `T` frames:

| Etapa | Costo por frame | Nota |
|---|---|---|
| E1 decode | `O(W·H)` en ffmpeg (fuera de Python) | `W,H` = resolución de trabajo, no de grilla |
| E2 grid+luma | `O(W·H)` (suma entera) + `O(N)` (peso luma) | dominado por la reducción de píxeles a celdas |
| E3 saliencia | `O(320²)` por inferencia, amortizado a `O(320²/5)` por frame | solo cada 5 frames; el resto es `O(N)` (blur+resize) |
| E4 cuantización | `O(N)` | vectorizado, con o sin densidad |
| E6 Bayer | `O(N)` (tile + suma) | offsets pre-computados una vez por job |
| E6 Floyd-Steinberg | `O(N)` pero **secuencial** (constante por celda, sin paralelismo) | única excepción a ADR-006 |
| E5 Sobel | `O(N)` (9 slices + trig por celda) | vectorizado |
| E7 histéresis | `O(N)` | comparación + `copyto` condicional |
| E8 atlas/compose | `O(N·GLYPH_H·GLYPH_W)` = `O(W·H)` en píxeles de salida | gather puro, sin dibujo |
| E9 encode | `O(W·H)` en ffmpeg/NVENC | fuera de Python |

Nada en el pipeline es peor que lineal en el número de píxeles o celdas — es la precondición que hace posible el gate de velocidad (§11.2): cualquier término `O(N²)` o peor en una etapa nueva rompería el presupuesto de tiempo antes de llegar a medirlo.
