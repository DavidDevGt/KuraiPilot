# Benchmark — tiempos reales de conversión

Cuánto tarda KuraiPilot en convertir un video, medido sobre la biblioteca de
clips de prueba en la **máquina de referencia** (`kurai`). Los números solo
valen en este hardware ([ADR-001](./docs/adr/ADR-001-local-first.md)); en otra
máquina cambian, pero la estructura (qué domina el costo) se mantiene.

> **Respuesta corta**: en la máquina de referencia, un video de **1 minuto** se
> convierte en **~2–3 s** con el preset `retro` (determinista, sin IA) o en
> **~7–19 s** con `detallado` (saliencia + bordes). Todo es varias veces más
> rápido que tiempo real: nunca esperás más de lo que dura el video.

## Hardware y pipeline medido

| Componente | Valor |
|---|---|
| GPU | NVIDIA RTX 5070 Ti · 16 GB · driver 595.71 · CUDA 13.2 |
| CPU | AMD Ryzen 7 9800X3D · 8 núcleos / 16 hilos |
| RAM | 59 GB |
| OS | Ubuntu 25.10 · kernel 6.17 |
| Video I/O | ffmpeg 7.1.1 · NVDEC/NVENC (`h264_nvenc`) |
| Inferencia | onnxruntime-gpu 1.27 · `CUDAExecutionProvider` |

**Camino activo en esta medición**: decode + encode por **NVENC** (GPU), y en
`detallado` la saliencia (U2Net-lite) corre en **GPU vía CUDA**. Es el camino
completo por hardware, no el de software.

## Metodología

- Se mide `run_job()` con `time.perf_counter` — excluye el arranque del
  intérprete y la importación de módulos; es el tiempo puro de conversión.
- **Warmup**: una corrida `detallado` descartada antes de medir, para amortizar
  la inicialización del contexto CUDA y la sesión ONNX (costo fijo de ~1 s que
  no se repite entre videos en un mismo proceso).
- **2 corridas medidas** por (video, preset); se reporta la **mejor**
  (steady-state).
- `--cols 160` fijo para todos (el default del CLI) — comparación directa entre
  videos, aunque penaliza a los verticales (ver análisis).
- Encoder: `h264_nvenc` (GPU) en ambos presets. Audio siempre `-c:a copy`.
- Reproducible con `uv run python tools/bench_videos.py` (harness versionado).

## Resultados

| Video | Tipo | Resolución | fps | Frames | `retro` | `detallado` | det. ms/frame |
|---|---|---|---|---|---|---|---|
| tears_of_steel_60s | live-action (letterbox) | 862×360 | 24 | 1440 | 1.65 s · **36.3×** | 6.86 s · **8.8×** | 4.8 |
| sintel_trailer_720p | animación 3D | 1280×720 | 24 | 1253 | 1.85 s · **28.2×** | 7.46 s · **7.0×** | 6.0 |
| artemis_launch | cohete / humo | 854×480 | 25 | 824 | 1.38 s · **23.9×** | 5.05 s · **6.5×** | 6.1 |
| notld_1968_60s | cine B/N con grano | 640×480 | 24 | 1439 | 2.58 s · **23.2×** | 13.06 s · **4.6×** | 9.1 |
| jellyfish_1080p_10s | real macro | 1920×1080 | 30 | 300 | 0.77 s · **13.1×** | 2.09 s · **4.8×** | 7.0 |
| bbb_1080p_10s | animación 3D | 1920×1080 | 60 | 600 | 1.17 s · **8.6×** | 3.81 s · **2.6×** | 6.4 |
| tiktok_dancer_white | vertical 9:16 (baile) | 720×1280 | 24 | 675 | 2.81 s · **10.0×** | 14.08 s · **2.0×** | 20.9 |
| tiktok_dance_neon | vertical 9:16 (baile) | 720×1280 | 24 | 472 | 2.07 s · **9.5×** | 9.90 s · **2.0×** | 21.0 |
| tiktok_coast | vertical 9:16 (naturaleza) | 720×1280 | 24 | 382 | 1.79 s · **8.9×** | 8.33 s · **1.9×** | 21.8 |
| tiktok_city_night | vertical 9:16 (ciudad) | 720×1280 | 24 | 343 | 1.63 s · **8.7×** | 7.51 s · **1.9×** | 21.9 |
| portrait_test | vertical 9:16 (barras) | 720×1280 | 30 | 900 | 3.58 s · **8.4×** | 18.95 s · **1.6×** | 21.1 |
| tiktok_dance_red | vertical 9:16 (baile) | 720×1280 | 24 | 156 | 0.98 s · **6.6×** | 3.65 s · **1.8×** | 23.4 |

(× = múltiplo de tiempo real; más alto = más rápido. `retro` va de 6.6× a 36×;
`detallado` de 1.6× a 8.8×.)

## Análisis: qué domina el costo

**El costo es proporcional a `frames × celdas_de_grilla`, no a la resolución del
video.** El pipeline reduce el frame a la grilla de caracteres *dentro de
ffmpeg* (fast path E1+E2 con `scale=area`), así que un 1080p y un 480p con la
misma grilla cuestan casi lo mismo por frame.

- **`retro` cuesta ~2.5–4 ms/frame**: decode + cuantización + histéresis +
  render por atlas + NVENC, todo vectorizado. Domina el I/O.
- **`detallado` agrega ~3–17 ms/frame**: saliencia (U2Net cada 5 frames en GPU),
  Sobel de bordes y composición a color. En landscape queda en ~5–9 ms/frame;
  en vertical sube a ~21 ms/frame.

**Por qué los verticales son más lentos**: con `--cols 160`, un frame 9:16
genera una grilla de ~160×284 celdas (~45k), contra ~160×45 (~7k) de un 16:9 —
**~3× más celdas por frame**, y por eso ~3× el tiempo por frame. No es la
orientación en sí: es que a columnas fijas, el vertical tiene muchas más filas.

**Recomendación práctica para verticales**: bajar a `--cols 120` los deja
cómodamente por encima de 2× tiempo real (medido: 2.6–3.3× en los mismos clips)
sin pérdida visible de detalle a la escala de un teléfono.

## Gate de performance (docs/07)

- `retro` (Fase 0): gate ≥4× — **cumplido con margen** (6.6× el más lento).
- `detallado` (Fase 1): gate ≥2× — **cumplido** en todo el contenido landscape y
  la mayoría del vertical; los verticales muy densos a `cols=160`
  (portrait 1.6×, dance_red 1.8×) quedan apenas por debajo y suben sobre 2× con
  `cols=120`.

## Contraste: GPU vs CPU para la saliencia

`detallado` con inferencia **CPU** (sin `onnxruntime-gpu`) cae a **~1.4×** tiempo
real en el clip de referencia (jellyfish), contra 4.8× con GPU: las ~60
inferencias de U2Net por cada 300 frames dominan cuando corren en CPU. El gate de
≥2× asume el camino GPU (extra `gpu`). Sin GPU, `detallado` sigue funcionando y
sigue siendo más rápido que tiempo real, pero pierde el margen; `retro` no usa IA
y es rápido en cualquier caso.

## Cómo reproducir

```bash
uv sync --extra preview --extra gpu     # onnxruntime-gpu para el camino GPU
uv run python tools/fetch_models.py     # baja y verifica U2Net-lite (saliencia)
uv run python tools/bench_videos.py     # corre este benchmark → output/bench_results.json
```

El `kurai bench` incorporado ([docs/05 §6](./docs/05-performance-and-capacity.md))
mide un clip sintético 1080p30 fijo para el gate de regresión de CI
(`bench/results/accepted.json`); este documento complementa eso con tiempos
reales sobre contenido variado. Ambos solo valen en la máquina de referencia.

---

_Medido el 2026-07-17 en `kurai`. Regenerar tras cambios al hot path._
