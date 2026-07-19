# KuraiPilot

**Conversor local-first de video a video ASCII renderizado.** Entra cualquier video que `ffmpeg` sepa decodificar, sale un `.mp4` reproducible en cualquier player — cada frame convertido a caracteres, glifos renderizados de verdad (no texto plano), audio original bit a bit intacto. Todo corre en tu máquina: sin backend, sin nube, sin telemetría.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![tests](https://img.shields.io/badge/tests-113%20passing-brightgreen)](./tests)

```bash
make setup                     # uv sync, entorno base + preview
make doctor                    # verifica ffmpeg / NVDEC / NVENC / GPU / Ollama
kurai convert video.mp4        # → video_ascii.mp4, audio intacto
kurai preview video.mp4        # UI en el navegador: sliders en vivo, WebGL
kurai live video.mp4           # reproduce ASCII directo en la terminal
```

---

## ¿Cómo funciona esto?

La idea central: **no hay "conversión a ASCII" como un solo paso mágico** — es un pipeline de 9 etapas deterministas que transforma cada frame de video en una matriz de caracteres, y esa matriz es lo único que importa. Todo lo demás (el mp4 de salida, la vista previa en el navegador, la reproducción en terminal) es una *proyección* de la misma matriz.

```
video.mp4
   │  ffmpeg decodifica (NVDEC si hay GPU) y separa el audio sin tocarlo
   ▼
frame RGB completo
   │  se reduce a una grilla de celdas (p.ej. 160×90), corrigiendo el aspecto
   │  1:2 de un carácter de terminal — si no corrigieras esto, los círculos
   │  saldrían ovalados
   ▼
luminancia (brillo) por celda
   │  se le aplica una curva gamma, y se cuantiza contra una "rampa" de
   │  caracteres calibrada por cobertura de tinta real:
   │       " .:-=+*#%@"
   │  el espacio es 0% de tinta, la @ es ~85% — no es un orden arbitrario,
   │  se midió cuántos píxeles negros pinta cada glifo
   ▼
dithering (Bayer 8×8)
   │  sin esto, un degradado suave se ve a bandas duras entre niveles; el
   │  patrón Bayer es FIJO entre frames (no depende del contenido), así que
   │  no induce parpadeo por sí mismo
   ▼
anti-flicker (histéresis)
   │  aquí está el problema real de cualquier conversor ingenuo: si cada
   │  frame cuantiza de forma independiente, el ruido de compresión de video
   │  hace que una celda "tiemble" entre dos caracteres 30 veces por segundo.
   │  la histéresis exige que el brillo cambie más de un umbral antes de
   │  aceptar un carácter nuevo — una celda solo cambia si el cambio es real
   ▼
CharMatrix: (char_idx, color) por celda   ◀── EL ARTEFACTO CANÓNICO
   │
   ├──▶ Renderer: atlas de glifos pre-renderizado (fancy indexing de NumPy,
   │    nunca se dibuja texto carácter por carácter — eso es 100-1000× más
   │    lento) → frame RGB → ffmpeg NVENC + mux del audio original
   │
   ├──▶ Preview: la misma CharMatrix viaja por WebSocket al navegador
   │    (no píxeles — ~72 KB/frame sin comprimir) y un shader WebGL2 de un
   │    solo pass la pinta con el mismo atlas
   │
   └──▶ Live: la misma CharMatrix se proyecta a texto ANSI con color 24-bit
        y se imprime directo en la terminal a 30 fps
```

**Por qué esto importa**: como los tres modos de salida (export, preview, terminal) consumen exactamente la misma función (`cells_to_charmatrix`), lo que ves en el preview interactivo *es* lo que vas a obtener en el video final — bit a bit, no "aproximadamente". Eso se verifica con un test (`test_preview_charmatrix_identical_to_export_path`), no es una promesa de la documentación.

**Reproducibilidad real**: mismo video + misma configuración → misma salida, byte por byte. Nada de `random` sin semilla, nada de orden de iteración de sets/dicts filtrándose al resultado, nada de timestamps del sistema en el cálculo. Esto se prueba con *golden files*: una CharMatrix de referencia versionada en el repo contra la que cada corrida se compara con igualdad exacta.

### Los 9 stages, en una tabla

| # | Etapa | Qué hace | Determinista |
|---|---|---|---|
| E1 | Decode/demux | ffmpeg separa video y audio; el audio nunca se re-codifica | Sí |
| E2 | Grilla | Reduce el frame a `rows×cols` celdas, corrige aspecto 1:2 | Sí |
| E3 | Saliencia *(Fase 1)* | Modelo ONNX marca qué región merece más detalle | Con modelo |
| E4 | Mapeo | Luminancia → índice de carácter en la rampa calibrada | Sí |
| E5 | Refinamiento *(Fase 2)* | CNN ajusta el glifo por textura local, no solo brillo | Con modelo |
| E6 | Dithering | Bayer 8×8 (GPU) o Floyd-Steinberg (Fase 2) | Sí |
| E7 | Anti-flicker | Histéresis por celda contra el último valor comprometido | Sí |
| E8 | Render | Atlas de glifos pre-renderizado, composición vectorizada | Sí |
| E9 | Encode/mux | ffmpeg NVENC, audio original con `-c:a copy` | Sí |

Hoy (Fase 0 + 0.5) el pipeline corre **completamente determinista**: cero componentes de IA en el camino obligatorio. E3 y E5 son los únicos puntos donde un modelo entra, y son opt-in por preset — si el modelo no está disponible, el sistema degrada al equivalente determinista con un warning, nunca aborta por eso.

### La regla que gobierna todo el motor: vectorización estricta

En las etapas 2-8 está prohibido iterar celda por celda en Python — todo es NumPy (o CuPy en GPU) operando sobre el array completo de una sola vez. Un `for` sobre 14,400 celdas (160×90) treinta veces por segundo en Python puro es, literalmente, cientos de veces más lento que la misma operación vectorizada. La única excepción documentada es Floyd-Steinberg, que es secuencial por naturaleza del algoritmo (cada píxel depende del error difundido por el anterior).

---

## Instalación

Requiere Python 3.12+ y `ffmpeg` en el PATH (con NVDEC/NVENC si querés aceleración GPU — no es obligatorio, el sistema corre por software si no hay GPU).

```bash
git clone https://github.com/DavidDevGt/KuraiPilot.git
cd KuraiPilot
make setup      # uv sync --extra preview
make doctor     # confirma ffmpeg / GPU / Ollama antes de convertir nada
```

`make setup-gpu` agrega los extras pesados de GPU (`onnxruntime-gpu`, `cupy`) — no hacen falta hasta la Fase 1.

## Uso

```bash
kurai convert video.mp4                          # preset retro, 160 cols
kurai convert video.mp4 --preset retro --cols 220 -o salida.mp4
kurai preview video.mp4 --port 8420               # navegador, sliders en vivo
kurai live video.mp4                              # ANSI en terminal, 30 fps
kurai bench --check                               # regresión de performance
kurai doctor                                      # diagnóstico del entorno
```

| Comando | Qué produce |
|---|---|
| `convert` | El `.mp4` ASCII final, con audio, encode NVENC/libx264 |
| `preview` | Server local (`127.0.0.1` solo) + cliente WebGL2 interactivo |
| `live` | Reproducción ANSI directa en stdout, sin encode ni archivo de salida |
| `bench` | Passthrough (techo de I/O) + retro (gate de velocidad), versionado |
| `doctor` | Verifica ffmpeg, NVDEC/NVENC, GPU, Ollama — primer comando en máquina nueva |

## Presets

| Preset | Saliencia | Refinamiento | Dither | Color | Estado |
|---|---|---|---|---|---|
| `retro` (default) | — | — | Bayer | mono | ✅ Fase 0 |
| `detallado` | U2Net-lite | edges | Bayer | fg | ✅ Fase 1 |
| `nitido` | — | edges | Floyd-Steinberg | fg+bg | ✅ Fase 2 parcial |
| `alta-fidelidad` | U2Net-lite | edges+CNN | Floyd-Steinberg | fg+bg | 🔜 Fase 2 |

Pedir un preset con componentes de una fase futura falla con un mensaje claro (`El preset necesita componentes de Fase N`), nunca con un traceback.

## Rendimiento

Medido en la máquina de referencia (RTX 5070 Ti 16 GB, Ryzen 7 9800X3D, ver `docs/05`) sobre un clip 1080p30 de 10 s:

| Modo | Factor sobre tiempo real |
|---|---|
| `bench passthrough` (decode→encode sin procesar, techo de I/O) | 9.27× |
| `bench retro` (pipeline completo) | **12.95×** |

Los benchmarks solo son válidos en la máquina de referencia ([ADR-001](./docs/adr/ADR-001-local-first.md)) — `kurai bench --check` falla si hay regresión >10% contra el baseline versionado en `bench/results/accepted.json`.

## Arquitectura

```
CLI ──▶ Preview ──▶ Engine ──▶ AI Sidecar | Renderer ──▶ Config | Types | Probe
```

Cada capa solo puede importar de las que están a su derecha — lo valida `import-linter` en cada `make check` y en CI; si un cambio necesita romper esta regla, el diseño del cambio está mal, no el linter. El detalle completo (vista de contenedores C4, flujo de datos, presupuesto de VRAM) vive en [docs/01-architecture-overview.md](./docs/01-architecture-overview.md).

Seis ADRs registran las decisiones estructurales y por qué se tomaron — no se editan, se supersede con uno nuevo si cambian:

| Decisión | ADR |
|---|---|
| Todo local, sin backend, sin telemetría | [ADR-001](./docs/adr/ADR-001-local-first.md) |
| Core determinista; IA opcional y desactivable | [ADR-002](./docs/adr/ADR-002-deterministic-core.md) |
| ffmpeg como única frontera de video, NVDEC/NVENC | [ADR-003](./docs/adr/ADR-003-ffmpeg-nvenc.md) |
| U2Net/ISNet ONNX para saliencia, no SAM en el hot path | [ADR-004](./docs/adr/ADR-004-saliency-model.md) |
| Ollama solo por escena, jamás por frame | [ADR-005](./docs/adr/ADR-005-ollama-role.md) |
| Python como lenguaje del core, vectorización estricta | [ADR-006](./docs/adr/ADR-006-python-core.md) |

## Estado del proyecto

```
Fase 0 ──▶ Fase 0.5 ──▶ Fase 1 ──▶ Fase 2 ──▶ Fase 3
  ✅          ✅         saliencia   alta       Scene
 core       preview      (A/B       fidelidad   Analyst
 determ.    + live       decide)   (CNN+flow)   (Ollama)
```

- **Fase 0** — pipeline determinista completo, preset `retro`, gate de velocidad ≥4× cumplido (12.95×).
- **Fase 0.5** — preview WebGL interactivo (<100 ms por ajuste) y `kurai live` en terminal, ambos verificados bit a bit idénticos al export.
- **Fase 1+** — pendiente: saliencia con gate A/B ≥60% de preferencia (la apuesta central del producto: densidad de detalle consciente del sujeto en vez de uniforme — se valida ahí o se abandona), CNN de glifos, Scene Analyst vía Ollama.

Detalle completo de criterios de aceptación por fase en [docs/07-roadmap.md](./docs/07-roadmap.md).

## Testing y calidad

```bash
make check      # gate local = CI: ruff + mypy strict + import-linter + pytest
make test-cpu   # la suite exactamente como la corre CI (KURAI_DISABLE_GPU=1)
make cov        # con reporte de cobertura
```

113 tests, 86%+ de cobertura, mypy en modo strict sobre todo `src/kurai` (sin `Any`, todo tipado con `npt.NDArray[np.uint8]` y compañía). Property-based testing con Hypothesis para las funciones puras de geometría/cuantización. Los tests golden comparan la CharMatrix con igualdad exacta — nunca video encodeado, nunca con tolerancia. CI corre solo en CPU; el camino CUDA y los benchmarks de performance se validan a mano en la máquina de referencia.

No hace falta GPU para contribuir.

## Documentación

| Pregunta | Dónde está la respuesta |
|---|---|
| Visión de producto, por qué existe esto | [IDEA.md](./IDEA.md) |
| Estado del arte investigado | [INVESTIGATION.md](./INVESTIGATION.md) |
| Arquitectura completa (normativa) | [docs/](./docs/README.md) |
| Contrato de cada etapa del pipeline | [docs/02-pipeline-spec.md](./docs/02-pipeline-spec.md) |
| Qué modelo de IA corre dónde y por qué | [docs/04-ai-components.md](./docs/04-ai-components.md) |
| Roadmap y gates de cada fase | [docs/07-roadmap.md](./docs/07-roadmap.md) |
| Cómo contribuir | [CONTRIBUTING.md](./CONTRIBUTING.md) |
| Reglas para agentes de IA trabajando en el repo | [CLAUDE.md](./CLAUDE.md) |
| Historial de cambios | [CHANGELOG.md](./CHANGELOG.md) |

## Principio rector

**Determinista por defecto, IA por elección.** El pipeline completo funciona con todos los componentes de IA apagados — no hay una versión "degradada sin IA", hay una versión *base* sobre la que cada componente de IA tiene que ganarse su lugar con métrica o A/B ciego, nunca por asunción ([ADR-002](./docs/adr/ADR-002-deterministic-core.md)).

## Licencia

[MIT](./LICENSE) — Copyright © 2026 KuraiPilot contributors.
