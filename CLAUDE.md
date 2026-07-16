# KuraiPilot — Guía para agentes

Conversor local-first de video a video ASCII renderizado. **Toda la documentación de diseño es normativa y vive en `docs/`** — este archivo es el mapa de reglas, no las reemplaza.

## Comandos

```bash
make setup      # uv sync (entorno base, sin GPU extras)
make check      # gate pre-commit: ruff + mypy strict + pytest
make doctor     # verifica ffmpeg/NVDEC/NVENC/GPU/Ollama (correr en máquina nueva)
make setup-gpu  # extras GPU (onnxruntime-gpu, cupy) — solo desde Fase 1
```

Correr un solo test: `uv run pytest tests/test_smoke.py::test_nombre -v`

## Dónde está cada verdad

| Pregunta | Fuente |
|---|---|
| Qué hace cada etapa del pipeline, contratos E/S | `docs/02-pipeline-spec.md` (normativo) |
| Qué modelo de IA corre dónde, presupuestos | `docs/04-ai-components.md` (normativo) |
| Por qué se decidió X | `docs/adr/` — los ADRs no se editan; se supersede con uno nuevo |
| Qué se hace ahora y con qué gate | `docs/07-roadmap.md` — Fase 0 primero: bench passthrough ANTES que etapas |
| Targets de performance | `docs/05` — regresión >10% en `kurai bench --check` = fallo |

## Reglas duras (violarlas = rechazar el cambio)

1. **Vectorización estricta (ADR-006)**: en etapas 2-8 del pipeline, prohibido iterar por píxel o celda en Python. Todo es NumPy/CuPy sobre arrays completos o batch ONNX. Única excepción: Floyd-Steinberg (`engine/dither.py`), que es secuencial por naturaleza.
2. **Determinismo (ADR-002 / G4)**: mismo input + misma config ⇒ CharMatrix bit a bit idéntica. Nada de `random` sin semilla fija, nada de depender de orden de dict/set, nada de tiempo del sistema en el pipeline.
3. **La CharMatrix es el artefacto canónico** (`src/kurai/types.py`): los tests golden comparan CharMatrix con igualdad exacta, nunca video encodeado ni con tolerancia.
4. **IA solo donde docs/04 lo permite**: nuevo modelo o cambio de scheduling requiere actualizar `docs/04` + ADR. Ollama/VLM jamás en el hot path (ADR-005). Modelos generativos generando output ASCII: prohibido (ADR-002).
5. **Degradación, no aborto**: fallo de componente opcional-IA ⇒ warning + equivalente determinista. El job solo aborta por fallo de etapa obligatoria, y reporta el timestamp.
6. **ffmpeg es la única frontera de video (ADR-003)**: nada más abre archivos de video. Audio siempre `-c:a copy`.
7. **Render por atlas**: prohibido dibujar texto carácter por carácter (PIL/Cairo) en el hot path — solo fancy indexing sobre el atlas pre-renderizado.
8. **Golden files**: actualizar un golden requiere justificar en el commit el cambio de algoritmo que lo motiva. Un golden que cambia "solo" es una regresión.

## Convenciones

- Python 3.12+, mypy strict en `src/kurai` — todo firmado con tipos; `npt.NDArray[np.uint8]` etc., no `Any`.
- Los stubs `NotImplementedError("Fase N")` marcan trabajo pendiente por fase del roadmap; al implementar uno, borrar el stub-marker y cubrirlo en tests.
- Docstrings de módulo referencian la sección de docs que implementan — mantener esas referencias al mover código.
- Presets en `presets/*.toml`, validados por `kurai/config.py` al cargar. Un preset nuevo se agrega a la tabla de `docs/02 §10` primero.
- Mensajes de usuario del CLI en español; código e identificadores en inglés.
- Commits: convencionales (`feat:`, `fix:`, `docs:`, `test:`), cuerpo en español si hace falta.

## Contexto de hardware (máquina de referencia `kurai`)

RTX 5070 Ti 16 GB (CUDA 13.2), Ryzen 7 9800X3D, 59 GB RAM, Ubuntu 25.10, ffmpeg 7.1 con NVDEC/NVENC verificado. Los benchmarks solo valen en esta máquina (ADR-001). Presupuesto VRAM detallado en `docs/04 §5` — ojo: Ollama con un modelo grande cargado compite por VRAM con el pipeline.

## Estado actual

Fase de diseño completada; esqueleto con contratos tipados y stubs por fase. **Siguiente paso: Fase 0 (`docs/07`), empezando por `kurai bench` con pipeline passthrough** (decode→encode sin procesar) para medir el techo de la infraestructura antes de escribir etapas.
