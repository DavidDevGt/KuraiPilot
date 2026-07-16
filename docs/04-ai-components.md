# 04 — AI Components

Documento **normativo** para todo componente con modelo del sistema. Regla de gobernanza: no se agrega ni cambia un modelo del hot path sin actualizar este doc y el ADR correspondiente. La justificación de *por qué* estos componentes y no otros está en [IDEA.md](../IDEA.md) (sección Delegation) e [INVESTIGATION.md §4](../INVESTIGATION.md).

## 1. Taxonomía: hot path vs. análisis

| Clase | Presupuesto de latencia | Presupuesto VRAM | Fallo |
|---|---|---|---|
| **Hot path** (por frame o cada N) | ≤ 10 ms por invocación | ≤ 2 GB agregado entre todos | Degrada a determinista, nunca aborta |
| **Análisis** (por escena) | segundos, asíncrono | El que Ollama administre | Se ignora; preset default |

## 2. Saliencia (Etapa 3)

- **Modelo**: U2Net-lite (~4.7 MB) como baseline; ISNet si la calidad de máscara en el set de evaluación lo justifica ([ADR-004](./adr/ADR-004-saliency-model.md) documenta por qué no SAM/MobileSAM en el hot path).
- **Entrada**: frame RGB 320×320, normalización ImageNet. **Salida**: máscara [0,1] re-muestreada a `(rows, cols)` → `density_map`.
- **Scheduling**: inferencia cada **N=5 frames**. Entre inferencias, el mapa se propaga: sin flow disponible, se mantiene (la saliencia cambia lento); con flow (preset alta-fidelidad), se warpea. En corte de escena se fuerza inferencia inmediata.
- **Post-proceso**: blur gaussiano (σ=2 celdas) sobre la máscara antes de usarla como densidad — una frontera dura de densidad se ve peor que no tener saliencia.
- **Criterio de aceptación** (gate de la Fase 1, [07](./07-roadmap.md)): en el set curado de [06 §4](./06-testing-and-evaluation.md), preferencia A/B ≥ 60% contra el pipeline sin saliencia. Si no lo pasa, el componente no se promociona a los presets.

## 3. Clasificador de glifos (Etapa 5, nivel `cnn`)

- **Modelo**: CNN propia, arquitectura mínima (~3 conv layers, ~200k params), entrada el parche de celda 8×16 en gris, salida softmax sobre el set de glifos de la fuente de referencia (~100 clases).
- **Entrenamiento** (en `training/`, corre en la 5070 Ti): dataset sintético auto-generado — cada glifo del atlas renderizado con distorsiones (blur, ruido, shift sub-celda, contraste) como en el enfoque de [arXiv 2503.14375](https://arxiv.org/pdf/2503.14375). No requiere datos etiquetados a mano; el generador es parte del repo.
- **Invocación selectiva**: solo celdas donde el Sobel de la Etapa 5-`edges` marcó estructura (típicamente 5-20% de las celdas). Se procesan en batch por frame (una llamada ONNX, no una por celda).
- **Criterio de aceptación**: mejora perceptible en el subset "textura" (pelo/follaje) del set curado **y** ≤ 3 ms por frame en el batch p95. Si empata con `edges`, se elimina — es el componente con mayor riesgo de ser rendimiento decreciente ([INVESTIGATION.md §4.1](../INVESTIGATION.md)).

## 4. Optical flow (Etapa 7, nivel avanzado)

- **Escalera de complejidad** (subir un peldaño solo si la métrica de flicker de [06 §3](./06-testing-and-evaluation.md) lo exige):
  1. Sin flow: histéresis fija (default, resuelve escenas de cámara estática).
  2. **Farneback denso en OpenCV CUDA** sobre el frame ya reducido a grilla — barato, sin modelo, suficiente para paneos suaves. Este es el nivel esperado para el preset alta-fidelidad.
  3. RAFT-small ONNX — solo si Farneback falla en cámara en mano; hoy es hipótesis, no plan.
- **Uso**: el flow desplaza el mapa de histéresis (`luma_committed`), no los píxeles. Resolución de flow = resolución de grilla; no se calcula flow a resolución de video.

## 5. Presupuesto de VRAM (16 GB, RTX 5070 Ti)

| Consumidor | Reserva | Nota |
|---|---|---|
| Desktop (GNOME/Xwayland) | ~0.5 GB | Medido en el entorno de referencia |
| NVDEC + NVENC + colas de frames GPU | ~2 GB | Depende de resolución de trabajo |
| AI Sidecar (U2Net + CNN + Farneback) | ~1.5 GB | Sesiones ONNX residentes todo el job |
| **Ollama con `minicpm-v4.5` cargado** | **~6-7 GB** | Solo si el Scene Analyst está activo |
| Margen libre | ~5 GB | |

Consecuencias operativas: (a) el export con Scene Analyst activo cabe con margen; (b) si el usuario tiene **otro** modelo grande cargado en Ollama (`glm-4.7-flash` ≈ 19 GB no cabe junto al pipeline), el Scene Analyst debe detectar presión de VRAM (consulta a `/api/ps` + `nvidia-smi`) y **auto-desactivarse con un warning** en vez de provocar OOM del sidecar. El hot path tiene prioridad sobre el análisis, siempre.

## 6. Scene Analyst (`minicpm-v4.5` vía Ollama)

El único componente que usa el stack Ollama local, y el único con un modelo grande. [ADR-005](./adr/ADR-005-ollama-role.md) fija sus límites.

- **Qué hace**: al detectar una escena nueva (PySceneDetect), toma **un** keyframe, lo manda a `minicpm-v4.5` y pide un JSON estructurado: `{tipo_de_escena, sujeto_principal, iluminación, sugerencia: {rampa, color_mode, saliencia_on}}`.
- **Qué se hace con eso**: en modo interactivo, se muestra como sugerencia ("esta escena es un retrato con poca luz — sugerido: rampa larga, mono ámbar"); en modo `--auto`, ajusta el preset por escena. Nunca toca parámetros por frame.
- **Contrato de fallo**: timeout 30 s por keyframe; sin respuesta o sin Ollama corriendo → se sigue con el preset del usuario, un solo warning por job. El export **jamás** espera al Scene Analyst: corre en paralelo y sus sugerencias aplican solo a escenas aún no procesadas (o en una segunda pasada si el usuario la pide).
- **Por qué este modelo**: es el único modelo de visión del inventario local, ya está instalado (6.1 GB), y la tarea (describir una escena, una vez por escena) es exactamente el perfil de carga que un VLM local tolera bien. Per-frame sería ~100-500 ms/frame — 30-150× el presupuesto del hot path; por eso la regla de la clase "análisis" existe.

## 7. Lo que está prohibido (y dónde está argumentado)

- Modelos generativos (difusión/GAN/VLM) generando o "mejorando" el output ASCII: [ADR-002](./adr/ADR-002-deterministic-core.md).
- Cualquier modelo por-frame fuera del presupuesto de la clase hot path: este doc, §1.
- Superresolución en v1: valor acotado ([INVESTIGATION.md §4.4](../INVESTIGATION.md)); candidato para después de Fase 2 solo si aparecen casos reales de input de baja resolución.
