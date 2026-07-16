# 05 — Performance & Capacity

Targets de rendimiento sobre el hardware de referencia (`kurai`), presupuestos por etapa y metodología de medición. Los números marcados **[target]** son objetivos de diseño a validar con `kurai bench` durante la Fase 0; los marcados **[medido]** se irán reemplazando con resultados reales — este doc es vivo.

## 1. Definición de throughput

La unidad es el **speed factor**: segundos de video procesados por segundo de wall-clock (`2.0×` = un video de 10 min exporta en 5 min). Se mide end-to-end (`kurai bench --preset X clip.mp4`), no por etapa aislada.

## 2. Targets por preset (input 1080p30, grilla 160×90)

| Preset | Speed factor | Racional |
|---|---|---|
| `retro` | ≥ 4× **[target]** | Sin IA; el techo debería ser NVENC o el pipe de ffmpeg, no nuestro código |
| `detallado` | ≥ 2× | + saliencia cada 5 frames + Sobel |
| `alta-fidelidad` | ≥ 0.8× | Con FS en CPU + CNN + flow; "algo más lento que tiempo real" es aceptable para un modo explícitamente premium |
| Preview (interactivo) | ≥ 30 fps a grilla ≤ 200 cols | Pipeline determinista, WebGL en cliente |
| Terminal live | 30 fps sostenido | Sin IA, sin render a píxeles (solo ANSI out) |

Presupuesto por frame a 30 fps: **33 ms**. A speed factor 4×, el pipeline completo tiene ~8 ms/frame de presupuesto — de ahí la obsesión por evitar transferencias PCIe y loops Python ([02 §11](./02-pipeline-spec.md), [ADR-006](./adr/ADR-006-python-core.md)).

## 3. Presupuesto por etapa (preset `alta-fidelidad`, el peor caso)

| Etapa | Presupuesto/frame | Dónde | Riesgo |
|---|---|---|---|
| 1 decode NVDEC + scale_cuda | 3 ms | GPU (ASIC) | Bajo — hardware dedicado |
| 2 resize+luma | 1 ms | GPU | Bajo |
| 3 saliencia (amortizada /5) | 2 ms | GPU | Medio — medir U2Net real en ORT |
| 4 mapeo | <0.5 ms | GPU | Bajo |
| 5 Sobel + CNN batch | 3 ms | GPU | **Alto** — gate de aceptación en [04 §3](./04-ai-components.md) |
| 6 FS dithering | 1 ms | CPU | Bajo a resolución de grilla (14.4k celdas) |
| 7 histéresis + Farneback | 2 ms | GPU | Medio |
| 8 render atlas | 2 ms | GPU | Bajo con fancy indexing |
| 9 NVENC + mux | 3 ms | GPU (ASIC) | Bajo |
| Transferencias + overhead Python | 5 ms | — | **El presupuesto real a vigilar** |
| **Total** | ~22 ms ≈ 1.5× | | |

Nota: las etapas GPU se solapan con decode/encode (pipeline asíncrono con colas), así que el total wall-clock es menor que la suma — la tabla es presupuesto de no-regresión por etapa, no una predicción de latencia serial.

## 4. Memoria

- **RAM (59 GB)**: cola bounded de 64 frames de trabajo. A resolución de trabajo 1280×1440 RGB float32 ≈ 22 MB/frame → ~1.4 GB por cola; con 3 colas activas <5 GB. Amplio margen; el límite existe para que un job no crezca sin techo, no porque falte RAM.
- **VRAM (16 GB)**: presupuesto detallado en [04 §5](./04-ai-components.md). Regla operativa: el pipeline reserva su working set al inicio del job y falla rápido si no hay VRAM suficiente, en vez de fragmentar y morir por OOM a mitad de un export de 2 horas.

## 5. Escenarios de capacidad

| Escenario | Comportamiento esperado |
|---|---|
| Video 4K de entrada | El costo extra es solo del NVDEC + scale_cuda (GPU); el resto del pipeline trabaja a resolución de trabajo, invariante al input. Penalización esperada <20% |
| Video de 2 horas | Streaming puro: RAM constante (colas bounded), sin archivos intermedios de frames. Solo crece el output |
| Grilla gigante (400 cols) | El costo de E2-E8 escala ~O(celdas); 400×225 = 6× las celdas del default. `alta-fidelidad` puede caer bajo 0.5× — documentar en la CLI, no impedir |
| VRAM ocupada por otro modelo Ollama | Scene Analyst se auto-desactiva ([04 §5](./04-ai-components.md)); hot path verifica su reserva al inicio |
| `KURAI_DISABLE_GPU=1` | Todo el camino CPU: decode sw, NumPy, encode `libx264`. Target: ≥ 0.5× en `retro` con los 16 threads del 9800X3D. Existe para CI y para validar la degradación, no como modo soportado de uso |

## 6. Metodología de benchmark

- `kurai bench` corre los 4 clips del set de fixtures ([06 §4](./06-testing-and-evaluation.md)) × 3 presets y emite JSON: speed factor, ms/frame por etapa (p50/p95), pico de VRAM y RAM, % de tiempo en transferencias host↔device.
- Los resultados se versionan en `bench/results/` con hash de commit — la regresión de performance se detecta por diff contra el último resultado aceptado, con umbral de ruido ±10%.
- Perfilado puntual: `nsys` (Nsight Systems) para ver solapamiento GPU real y detectar sincronizaciones accidentales (`cudaStreamSynchronize` implícitos de CuPy↔ORT son el sospechoso habitual).

## 7. Optimizaciones diferidas (no hacer todavía)

Listadas para que no se hagan prematuramente; cada una entra solo si `bench` muestra que su etapa rompe presupuesto:

1. TensorRT EP para U2Net/CNN (vs. CUDA EP) — típicamente 1.5-3× en modelos chicos.
2. Kernel CUDA propio para histéresis+mapeo fusionados (hoy: operaciones CuPy separadas).
3. Migrar el hot loop a Rust/PyO3 — solo si el overhead Python de §3 supera su presupuesto de forma no arreglable con vectorización ([ADR-006](./adr/ADR-006-python-core.md) define el criterio de salida).
