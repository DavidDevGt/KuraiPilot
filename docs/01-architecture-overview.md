# 01 — Architecture Overview

## 1. Contexto y alcance

KuraiPilot convierte cualquier video de entrada en un video renderizado enteramente en caracteres ASCII/Unicode, con el audio original intacto, ejecutándose por completo en la máquina local. No hay backend, no hay nube, no hay telemetría: el sistema es una aplicación local con dos superficies (CLI y preview web servido localmente).

### Objetivos

- **G1**: Aceptar "cualquier video" que ffmpeg pueda decodificar (contenedores y códecs arbitrarios) y producir un mp4/webm reproducible en cualquier player.
- **G2**: Preview interactivo casi instantáneo antes del export pesado.
- **G3**: Calidad visual superior al conversor promedio: densidad de detalle consciente del sujeto (saliencia) y ausencia de flicker temporal.
- **G4**: Reproducibilidad: mismo input + misma config → misma salida.

### No-objetivos (explícitos)

- **NG1**: No es un servicio multi-tenant ni una web pública. No hay colas, auth, ni billing.
- **NG2**: No genera arte con modelos generativos end-to-end ([ADR-002](./adr/ADR-002-deterministic-core.md)).
- **NG3**: No es un reproductor de terminal en vivo como objetivo primario (existe como modo secundario barato, ver §4).
- **NG4**: No edición de video (cortes, transiciones): entra un video, sale ese mismo video en ASCII.

## 2. Vista de contexto (C4 nivel 1)

```
┌────────────┐   video file    ┌──────────────────────────┐   mp4/webm    ┌────────────┐
│   Usuario   ├───────────────▶│        KuraiPilot         ├──────────────▶│  Player /   │
│  (rou)      │◀───────────────┤  (proceso local, kurai)   │               │  redes      │
└────────────┘  preview WebGL  └───────┬──────────┬───────┘               └────────────┘
                                        │          │
                              ┌─────────▼──┐  ┌────▼─────────┐
                              │  ffmpeg     │  │  Ollama       │
                              │ (NVDEC/     │  │ (minicpm-v4.5 │
                              │  NVENC)     │  │  análisis de  │
                              └────────────┘  │  escena, opt.)│
                                              └──────────────┘
```

Dependencias externas: **ffmpeg** (decode/encode, obligatoria) y **Ollama** (análisis de escena, opcional — el sistema funciona sin el daemon corriendo).

## 3. Vista de contenedores (C4 nivel 2)

| Contenedor | Responsabilidad | Tecnología | Proceso |
|---|---|---|---|
| **CLI (`kurai`)** | Punto de entrada: parsea config, orquesta el job, reporta progreso | Python (Typer/argparse) | Proceso principal |
| **Engine** | El pipeline de 9 etapas (decode → … → encode). Dueño de la GPU para frames | Python + NumPy/CuPy + ONNX Runtime | Mismo proceso, workers |
| **Renderer** | Composición texto→imagen vía atlas de glifos; nunca dibuja carácter por carácter | NumPy tile composition / CUDA | Dentro del Engine |
| **Preview server** | Sirve la UI de preview y streamea frames de baja resolución al navegador | FastAPI + WebSocket, WebGL en cliente | Proceso hijo opcional |
| **AI Sidecar** | Modelos por-frame: saliencia, clasificador de carácter, optical flow | ONNX Runtime (CUDA EP / TensorRT EP) | Dentro del Engine, sesión dedicada |
| **Scene Analyst** | Análisis semántico por escena (no por frame): sugiere preset/rampa | Ollama HTTP API → `minicpm-v4.5` | Daemon externo, best-effort |

Separación clave: el **AI Sidecar** vive en el hot path (corre por frame o cada N frames, latencia en milisegundos, modelos <1 GB); el **Scene Analyst** vive fuera de él (corre una vez por escena detectada, latencia en segundos, tolerante a fallos — si Ollama no responde, se usa el preset por defecto y se sigue). El detalle normativo está en [04-ai-components.md](./04-ai-components.md) y [ADR-005](./adr/ADR-005-ollama-role.md).

## 4. Modos de operación

| Modo | Superficie | Pipeline | Latencia objetivo |
|---|---|---|---|
| **Export** (principal) | CLI | Completo, con etapas de IA según preset | Batch; ver targets en [05](./05-performance-and-capacity.md) |
| **Preview** | Navegador local | Determinista, resolución de grilla reducida, WebGL en cliente | Interactivo (<100 ms por ajuste de slider) |
| **Terminal live** (secundario) | stdout ANSI | Determinista puro, sin IA | Tiempo real 30 fps |

Los tres modos comparten el mismo core de mapeo (mismo código de luminancia→carácter, misma rampa) para que lo que se ve en preview sea lo que sale en export.

## 5. Flujo de datos (export)

```
input.mp4
  │  ffmpeg NVDEC ─ demux audio aparte (sin tocar)
  ▼
frames NV12/RGB en GPU ──▶ [scene detect] ──▶ Scene Analyst (async, opcional)
  ▼
resize a grilla (corrige aspecto 1:2 del glifo)
  ▼
[saliencia cada N frames + propagación] → mapa de densidad
  ▼
luminancia → índice de carácter (+ refinamiento CNN si preset alta-fidelidad)
  ▼
dithering (Bayer en GPU | Floyd-Steinberg en CPU según preset)
  ▼
anti-flicker (histéresis por celda; optical flow si alta-fidelidad)
  ▼
matriz de (char_idx, fg_color[, bg_color]) por frame   ◀── ARTEFACTO CANÓNICO
  ▼
Renderer: atlas de glifos → frame RGB
  ▼
ffmpeg NVENC (h264/hevc) + mux audio original (-c:a copy)
  ▼
output.mp4
```

La **matriz de caracteres** es el artefacto canónico del sistema: es lo que se testea con golden files, lo que garantiza reproducibilidad bit a bit, y lo que permite exportar a otros formatos (`.cast`, HTML) sin re-procesar. El render a píxeles es una proyección de ese artefacto.

## 6. Restricciones del entorno de referencia

El hardware de referencia (y de desarrollo) es la workstation `kurai`; los presupuestos de [05](./05-performance-and-capacity.md) están calibrados contra ella:

- **CPU**: AMD Ryzen 7 9800X3D (16 threads) — Floyd-Steinberg secuencial y el mux corren aquí.
- **GPU**: RTX 5070 Ti, **16 GB VRAM**, CUDA 13.2, driver 595 — NVDEC/NVENC + inferencia ONNX. El presupuesto de VRAM es el recurso más escaso: Ollama con un modelo de visión cargado puede tomar 6-9 GB, por eso el Scene Analyst es opcional y el AI Sidecar usa modelos pequeños ([04 §5](./04-ai-components.md)).
- **RAM**: 59 GB — permite buffering generoso de frames entre etapas (colas bounded, ver [02 §11](./02-pipeline-spec.md)).
- **OS**: Ubuntu 25.10, Python 3 del sistema; el proyecto usa venv propio.

## 7. Decisiones estructurales y sus ADRs

| Decisión | ADR |
|---|---|
| Todo local, sin backend | [ADR-001](./adr/ADR-001-local-first.md) |
| Core determinista; IA opcional y desactivable | [ADR-002](./adr/ADR-002-deterministic-core.md) |
| ffmpeg como única frontera de decode/encode, con NVDEC/NVENC | [ADR-003](./adr/ADR-003-ffmpeg-nvenc.md) |
| U2Net/ISNet ONNX para saliencia (no SAM en el hot path) | [ADR-004](./adr/ADR-004-saliency-model.md) |
| Ollama solo para análisis por escena, nunca por frame | [ADR-005](./adr/ADR-005-ollama-role.md) |
| Python como lenguaje del core, con vectorización estricta | [ADR-006](./adr/ADR-006-python-core.md) |
