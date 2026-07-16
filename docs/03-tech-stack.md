# 03 — Tech Stack

Stack elegido y su mapeo al entorno de referencia (`kurai`: Ubuntu 25.10, Ryzen 7 9800X3D, RTX 5070 Ti 16 GB, CUDA 13.2, driver 595.71, 59 GB RAM). Las decisiones estructurales tienen ADR; esto es el inventario operativo.

## 1. Runtime del core

| Componente | Elección | Versión / nota |
|---|---|---|
| Lenguaje | Python | 3.12+ (venv propio, no el Python del sistema). [ADR-006](./adr/ADR-006-python-core.md) |
| Arrays CPU | NumPy | Vectorización estricta: ningún loop por-píxel/por-celda en Python en el hot path |
| Arrays GPU | CuPy | Build para CUDA 13.x. Comparte puntero de device memory con ONNX Runtime vía DLPack |
| Inferencia | ONNX Runtime GPU | Execution provider CUDA; TensorRT EP como optimización posterior si hace falta ([05](./05-performance-and-capacity.md)) |
| Video I/O | ffmpeg (subprocess, rawvideo pipes) | Build del sistema con `--enable-nvdec --enable-nvenc`; verificar con `ffmpeg -hwaccels`. [ADR-003](./adr/ADR-003-ffmpeg-nvenc.md) |
| Detección de escena | PySceneDetect (ContentDetector) | Alimenta el reset de histéresis (E7) y el Scene Analyst |
| CLI | Typer | Subcomandos: `convert`, `preview`, `live`, `bench` |
| Config | Pydantic (settings + presets en TOML) | Presets de [02 §10](./02-pipeline-spec.md) como archivos versionados en `presets/` |

**Por qué ffmpeg por subprocess y no PyAV**: PyAV amarra la versión de libav a la wheel y complica NVDEC; el subprocess con pipes rawvideo es el patrón más robusto, se benchmarkea igual, y deja a ffmpeg actualizarse con el sistema. Trade-off aceptado: parsing de metadatos vía `ffprobe -print_format json`.

## 2. Componentes de IA (inventario; spec normativa en [04](./04-ai-components.md))

| Rol | Modelo | Formato | Dónde corre |
|---|---|---|---|
| Saliencia (E3) | U2Net-lite / ISNet | ONNX | AI Sidecar, GPU, hot path cada N frames |
| Clasificador de glifo (E5) | CNN propia ~200k params | ONNX (entrenada en PyTorch) | AI Sidecar, GPU, solo celdas estructurales |
| Optical flow (E7) | OpenCV CUDA Farneback → RAFT-small si no alcanza | OpenCV / ONNX | AI Sidecar, GPU |
| Scene Analyst | `minicpm-v4.5` vía Ollama | GGUF (ya instalado) | Daemon Ollama, fuera del hot path, opcional |

Modelos del inventario de Ollama local que **no** se usan y por qué: los LLM de texto (`devstral`, `qwen*-coder`, `glm-4.7-flash`, etc.) son herramientas de desarrollo, no componentes del producto; `nomic-embed-text` no tiene rol en v1 (candidato futuro: búsqueda semántica sobre biblioteca de conversiones — fuera de alcance).

## 3. Preview

| Componente | Elección |
|---|---|
| Server local | FastAPI + Uvicorn, bind solo a `127.0.0.1` |
| Transporte | WebSocket: la CharMatrix (no píxeles) viaja al cliente — 160×90×5 bytes ≈ 72 KB/frame sin comprimir, trivial en localhost |
| Render cliente | WebGL2, atlas de glifos como textura, un quad instanciado por celda (patrón sprite-sheet de [INVESTIGATION.md §3](../INVESTIGATION.md)) |
| Build frontend | Vite + TypeScript vanilla (sin framework; la UI es un canvas y ocho sliders) |

Enviar la CharMatrix en vez de video renderizado mantiene el principio de artefacto canónico: el cliente WebGL y el Renderer de export son dos proyecciones del mismo dato.

## 4. Testing y calidad

| Área | Herramienta |
|---|---|
| Tests | pytest; golden files de CharMatrix en `tests/golden/` ([06](./06-testing-and-evaluation.md)) |
| Bench | `kurai bench` (subcomando propio) + pytest-benchmark para microbenchmarks |
| Lint/format | ruff (lint+format), mypy en el core del Engine |
| Entrenamiento CNN E5 | PyTorch + export ONNX; script en `training/`, corre local en la 5070 Ti |

## 5. Layout de repositorio propuesto

```
KuraiPilot/
├── IDEA.md / INVESTIGATION.md
├── docs/                    # este directorio
├── presets/                 # retro.toml, detallado.toml, alta-fidelidad.toml
├── src/kurai/
│   ├── cli.py
│   ├── engine/              # etapas 1-9, una módulo por etapa
│   ├── ai/                  # sidecar: saliencia, cnn, flow + scene_analyst.py
│   ├── render/              # atlas de glifos, composición
│   └── preview/             # FastAPI + static frontend build
├── frontend/                # fuente TS del preview
├── training/                # entrenamiento del clasificador de glifos
├── tools/                   # calibrate_ramp.py, fetch_models.py
├── models/                  # ONNX descargados (gitignored, hash-pinned)
└── tests/
    ├── golden/
    └── fixtures/            # clips de prueba cortos (retrato, deporte, paisaje, animación)
```

## 6. Gestión de modelos y entorno

- Los ONNX no van al repo: `tools/fetch_models.py` los descarga y verifica por SHA-256 pineado. Sin red, el sistema arranca en modo determinista puro y lo dice.
- `uv` para el venv y lockfile (`uv.lock` versionado) — reproducibilidad del entorno, coherente con G4.
- Variables de entorno relevantes: `KURAI_MODELS_DIR`, `KURAI_OLLAMA_URL` (default `http://127.0.0.1:11434`), `KURAI_DISABLE_GPU` (fuerza el camino CPU completo, usado en CI y para verificar la degradación de [02 §11](./02-pipeline-spec.md)).
