# 02 — Pipeline Specification

Especificación normativa de las 9 etapas del pipeline de export. Cada etapa define: contrato de entrada/salida, algoritmo, y si es determinista u opcional-IA. La numeración coincide con el diagrama de [01 §5](./01-architecture-overview.md).

## Convenciones

- Un **frame de trabajo** es un array `float32` normalizado [0,1] en layout `(H, W, C)` RGB, o su equivalente en GPU (CuPy/tensor ONNX).
- La **grilla** es `(rows, cols)`: la resolución en caracteres de la salida. Default: `cols=160`, `rows` derivado del aspect ratio.
- Una **celda** es la región de píxeles del frame que se colapsa a un carácter.
- El **artefacto canónico** por frame es `CharMatrix`: arrays `char_idx: uint8[rows, cols]`, `fg: uint8[rows, cols, 3]`, opcional `bg: uint8[rows, cols, 3]`.

## Etapa 1 — Decode y demux

- **Entrada**: path a archivo de video. Cualquier contenedor/códec que ffmpeg soporte.
- **Salida**: stream de frames + metadatos (`fps`, `duration`, `width`, `height`, `rotation`, `pix_fmt`) + stream de audio apartado sin decodificar.
- **Implementación**: ffmpeg con `-hwaccel cuda` (NVDEC) cuando el códec lo soporta (h264, hevc, vp9, av1); fallback transparente a decode por software. Frames se piden en la resolución de trabajo, no la nativa: si la grilla es 160×90 celdas y cada celda muestrea 8×16 px, la resolución de trabajo es 1280×1440 — **pedir a ffmpeg el scale en GPU (`scale_cuda`) antes de bajar a RAM** para no mover frames 4K por PCIe innecesariamente.
- **Fast path E1+E2 fusionadas** (medido en Fase 0): cuando ninguna etapa del preset necesita píxeles a resolución de trabajo (sin saliencia ni refine, i.e. preset `retro`), ffmpeg escala con `flags=area` directamente a la resolución de grilla — el promedio por celda de E2 ocurre dentro de ffmpeg y el pipe se reduce ~128×. Esto llevó el preset retro de 1.5× a ~13× tiempo real. La reducción en-engine (`to_grids`) sigue siendo el camino canónico para presets con E3/E5 y para los golden files.
- **Metadatos críticos**: respetar el flag de `rotation` (videos de celular vienen rotados por metadata); VFR (frame rate variable) se normaliza a CFR con `fps=` filter para que la relación frame↔carácter-matrix sea 1:1.
- **Audio**: se extrae una sola vez (`-c:a copy` a un contenedor temporal) y no se toca hasta la etapa 9.

## Etapa 2 — Resize a grilla con corrección de aspecto

- **Entrada**: frame RGB en resolución de trabajo.
- **Salida**: dos mapas por frame: `luma_grid: float32[rows, cols]` y `color_grid: float32[rows, cols, 3]`.
- Los glifos monoespaciados no son cuadrados: con la fuente de referencia (ver Etapa 8) la celda es **8×16 px (ratio 1:2)**. El muestreo vertical usa el doble de píxeles fuente que el horizontal; ignorar esto deforma la imagen (error clásico, [INVESTIGATION.md §6](../INVESTIGATION.md)).
- **Algoritmo**: area-average (equivalente a `cv2.INTER_AREA`) — no nearest-neighbor, que aliasa, ni bicúbico, que sobrepasa rangos. Para `luma_grid` se usa luminancia relativa BT.709: `Y = 0.2126R + 0.7152G + 0.0722B` calculada **antes** del promedio por celda (promediar luma, no promediar RGB y luego convertir).
- **Determinista**. Sin parámetros de usuario salvo `cols`.

## Etapa 3 — Saliencia (opcional-IA)

- **Entrada**: frame RGB reducido a 320×320 (entrada del modelo). Implementación Fase 1: se re-muestrea la grilla RGB ya reducida (E1+E2 fusionadas) a 320×320 — el `density_map` es de resolución de grilla de todos modos; la saliencia a resolución de trabajo es un refinamiento futuro.
- **Salida**: `density_map: float32[rows, cols]` en [0,1] — 1.0 = máximo detalle.
- **Modelo y scheduling**: normativo en [04 §2](./04-ai-components.md). Corre cada N frames (default N=5) con propagación del mapa entre corridas.
- **Efecto downstream**: el `density_map` modula (a) la longitud efectiva de la rampa en Etapa 4 — zonas de baja densidad usan una rampa corta de 4 niveles, zonas salientes la rampa completa — y (b) la subdivisión de celda en presets altos (una celda de fondo puede representar 2×2 celdas fusionadas).
- **Apagado** (default en preset `retro`): `density_map ≡ 1.0` y el pipeline es idéntico a no tener la etapa.

## Etapa 4 — Mapeo luminancia → carácter

- **Entrada**: `luma_grid`, `density_map`.
- **Salida**: `char_idx: uint8[rows, cols]`.
- **Rampas** (ordenadas de vacío a denso, calibradas por cobertura de tinta real del glifo en la fuente de referencia, no por intuición):
  - `short` (10): ` .:-=+*#%@`
  - `long` (70): rampa estándar extendida; para presets de fidelidad.
  - `blocks` (Unicode): ` ░▒▓█` + eighth-blocks; máxima fidelidad tonal, mínima estética "texto".
- La calibración de rampa es un paso de build (`tools/calibrate_ramp.py` renderiza cada glifo con la fuente de referencia y mide píxeles encendidos), no una constante copiada de un blog: la cobertura de tinta depende de la fuente.
- **Mapeo**: `idx = quantize(luma, levels=len(ramp_efectiva(density)))`. Con gamma opcional (`luma^0.8` default) porque la percepción de brillo no es lineal.
- **Determinista**.

## Etapa 5 — Refinamiento de carácter por estructura (opcional-IA)

- **Entrada**: parche de píxeles de la celda (8×16) + `char_idx` propuesto.
- **Salida**: `char_idx` refinado, solo en celdas marcadas como "estructurales".
- Dos niveles, activados por preset:
  - **`edges` (determinista, preset `detallado`)**: Sobel por celda; si la magnitud del gradiente supera umbral, se reemplaza el carácter tonal por el direccional que calza con la orientación (`/ \ | — _ ( )`), estilo AsciiArtist.
  - **`cnn` (IA, preset `alta-fidelidad`)**: clasificador CNN pequeño (spec en [04 §3](./04-ai-components.md)) que elige el mejor glifo del set completo para el parche. Solo corre en celdas donde `edges` detectó estructura — no en celdas tonales planas, que son la mayoría.

## Etapa 6 — Dithering

- **Entrada/Salida**: opera sobre la cuantización de la Etapa 4 (se implementan fusionadas: el dithering ajusta el error de cuantización de luma antes del lookup final).
- Dos algoritmos, elección por preset:
  - **Bayer 8×8 (ordered)**: paralelizable, corre en GPU, default para preview y preset `retro`. Patrón estable entre frames (mismo patrón espacial siempre) → cero flicker inducido.
  - **Floyd-Steinberg serpentine**: mejor gradiente, secuencial → corre en CPU (el 9800X3D lo hace sobre una grilla de 160×90 en microsegundos; la secuencialidad no es problema a resolución de grilla, solo lo sería a resolución de píxeles). **Advertencia**: FS es sensible a ruido entre frames y puede inducir flicker; siempre se combina con Etapa 7.
- **Determinista** (ambos).

## Etapa 7 — Estabilidad temporal (anti-flicker)

- **Entrada**: `char_idx[t]`, `char_idx[t-1]`, `luma_grid[t]`, `luma_grid[t-1]`.
- **Salida**: `char_idx[t]` estabilizado.
- **Nivel base (determinista, siempre activo)** — histéresis por celda: una celda solo cambia de carácter si `|luma[t] - luma_committed|` supera `h = 1.5 × (ancho del nivel de cuantización)`. `luma_committed` es la luma del momento del último cambio, no la del frame anterior (evita drift acumulado). En cortes de escena (detectados en Etapa 1 por diferencia global) la histéresis se resetea completa. (Ajustado de 0.6 a 1.5 tras medir FCR≈0.30-0.35 en metraje real con grano/textura — el valor original solo se había validado contra el fixture sintético estático del gate; ver `src/kurai/engine/stability.py`.)
- **Nivel avanzado (IA, preset `alta-fidelidad`)** — optical flow (spec [04 §4](./04-ai-components.md)): el mapa de histéresis se desplaza siguiendo el flow, de modo que en paneos la "memoria" de cada celda sigue al contenido en vez de quedarse fija en coordenadas de pantalla. Sin esto, la histéresis base durante un paneo produce estelas.
- La métrica que valida esta etapa (varianza de carácter en zonas estáticas) está definida en [06 §3](./06-testing-and-evaluation.md).

## Etapa 8 — Render texto → píxeles

- **Entrada**: `CharMatrix`.
- **Salida**: frame RGB `uint8[rows*16, cols*8, 3]`.
- **Fuente de referencia**: una sola fuente monoespaciada bitmap-friendly empaquetada con el proyecto (candidata: Spleen o Cozette 8×16; decidir y fijar en build — la fuente es parte del contrato de reproducibilidad).
- **Implementación**: **atlas de glifos pre-renderizado** una vez al inicio (`uint8[n_glyphs, 16, 8]`); el frame se compone con fancy indexing de NumPy/CuPy: `frame = atlas[char_idx]` reshapeado + tintado por `fg`. Prohibido dibujar carácter por carácter con PIL/Cairo en el hot path — es 100-1000× más lento y fue la causa de muerte de varios proyectos relevados.
- Modos de color: `mono` (verde/ámbar/blanco sobre negro), `fg` (carácter tintado con el color de la celda), `fg+bg` (semi-bloque con dos colores por celda, máxima fidelidad).
- **Contrato `fg+bg`** (Fase 2): la Etapa 1 decodifica la grilla a `rows*2` filas — dos muestras verticales por celda ("semi-bloque"). `fg` = color de la mitad superior, `bg` = el de la inferior; el camino tonal (E4-E7) corre sobre el promedio entero de ambas mitades. En composición, la tinta del glifo lleva `fg` y el resto de la celda `bg` (mezcla por el mask del atlas, solo fancy indexing). `CharMatrix.bg` deja de ser `None` exactamente en este modo.

## Etapa 9 — Encode y mux

- **Entrada**: stream de frames RGB + audio apartado de la Etapa 1.
- **Salida**: `output.mp4` (h264 por compatibilidad) o `output.webm`.
- **Implementación**: ffmpeg NVENC (`h264_nvenc`, preset `p5`, CQ configurable). El contenido ASCII es de alta frecuencia espacial y detalle fino: usar `-tune hq` y CQ ≤ 23 o los glifos se vuelven papilla con la compresión. Audio: `-c:a copy` — el audio original se muxea sin recodificar; solo se recodifica si el contenedor de destino no soporta el códec original.
- **Prácticas obligatorias auditadas contra la industria** (Fase 0):
  - NVENC en calidad constante real exige `-b:v 0` junto a `-cq`; sin eso el bitrate queda capado al default y `-cq` es decorativo.
  - La conversión RGB→YUV se fija a BT.709 (`scale=out_color_matrix=bt709:out_range=tv:flags=full_chroma_int+accurate_rnd`) y se taguea el VUI completo vía `setparams` — swscale usa BT.601 por defecto y los players HD asumen BT.709 (color corrido sin esto).
  - El frame rate viaja como racional exacto (`30000/1001`) por `fps=` y `-r`, nunca como float formateado (drift en videos largos).
  - `stderr` de los subprocess ffmpeg va a archivo temporal, jamás a PIPE sin lector concurrente (deadlock cuando el buffer del pipe se llena de warnings).
- **Sincronía**: la normalización a CFR de la Etapa 1 garantiza que `n_frames_out = n_frames_in`; el PTS del primer frame se preserva.

## 10. Presets (mapeo a etapas)

| Preset | E3 saliencia | E5 refinamiento | E6 dither | E7 anti-flicker | E8 color |
|---|---|---|---|---|---|
| `retro` (default) | off | off | Bayer | histéresis | mono |
| `detallado` | on | edges | Bayer | histéresis | fg |
| `nitido` | off | edges | Floyd-Steinberg | histéresis | fg+bg |
| `alta-fidelidad` | on | edges+cnn | Floyd-Steinberg | histéresis+flow | fg+bg |

`nitido` es el máximo detalle **determinista** (sin IA): la saliencia va apagada tras el NO-GO del A/B de Fase 1 (`docs/evaluations/2026-07-18-ab-saliencia.md`).

## 11. Concurrencia y manejo de errores

- El pipeline es un **DAG de etapas con colas bounded** (backpressure): decode no puede adelantarse más de K frames (default 64) al encode; con 59 GB de RAM el buffer es generoso pero acotado — un video de 2 h no debe poder llenar la memoria.
- Etapas GPU (2, 3, 6-Bayer, 8) se agrupan para minimizar transferencias host↔device; el frame baja a CPU una sola vez si el preset usa FS, cero veces si no.
- **Fallo de una etapa opcional-IA** (modelo no carga, OOM de VRAM): warning + degradación al equivalente determinista del preset inferior; el job nunca aborta por un componente opcional. **Fallo de etapa obligatoria** (decode corrupto a mitad de archivo): el job termina con exit code ≠ 0 y reporta el timestamp del frame que falló; lo ya encodeado se descarta (no se entregan outputs parciales silenciosos).
- Progreso: cada etapa reporta frames procesados; la CLI muestra un solo progreso agregado con ETA basado en throughput medido, no estimado.
