# ADR-006 — Python como lenguaje del core, con vectorización estricta y criterio de salida

**Estado**: Aceptado — 2026-07-15

## Contexto

Candidatos para el Engine: Rust (performance predecible, pero el ecosistema de inferencia/CUDA/video es más áspero y el proyecto es unipersonal), Go (precedente en AlexEidt/ASCII-Video, pero sin historia buena con ONNX/CUDA), y Python (ecosistema completo: NumPy/CuPy, ONNX Runtime, OpenCV, PyTorch para entrenar la CNN — todo de primera clase). El riesgo de Python es conocido: un loop por-celda en el intérprete mata el presupuesto de ~8 ms/frame ([05 §2](../05-performance-and-capacity.md)).

La observación que decide: en este pipeline el trabajo pesado ya vive fuera del intérprete (ffmpeg, NVDEC/NVENC, kernels de CuPy, sesiones ONNX). El rol de Python es orquestar y expresar las etapas como operaciones de array — si se mantiene esa disciplina, el overhead del intérprete es marginal.

## Decisión

- Python 3.12+ para todo el sistema (Engine, CLI, preview server); TypeScript solo en el frontend WebGL.
- **Regla de vectorización estricta**: en el hot path (etapas 2-8) está prohibido iterar por píxel o por celda en Python. Toda etapa se expresa como operaciones NumPy/CuPy sobre arrays completos o llamadas batch a ONNX. Un `for` sobre celdas en el hot path se rechaza en revisión aunque "funcione".
- La única excepción tolerada: Floyd-Steinberg (inherentemente secuencial) — se implementa con Numba `@njit` o como extensión si el fallback puro resulta lento, pero su presupuesto (1 ms a resolución de grilla) probablemente ni lo exija.
- **Criterio de salida** (cuándo re-abrir esta decisión): si tras vectorizar correctamente, el renglón "transferencias + overhead Python" de [05 §3](../05-performance-and-capacity.md) excede su presupuesto de 5 ms/frame de forma sostenida y `nsys` muestra que el tiempo está en el intérprete/GIL y no en nuestros kernels, se migra el hot loop a Rust/PyO3 manteniendo Python como shell. Ese cambio requiere un ADR que supersede a este.

## Consecuencias

- (+) Velocidad de desarrollo máxima con el ecosistema ML/video completo; entrenar la CNN de E5 y consumirla comparten lenguaje.
- (+) El criterio de salida convierte "¿y si Python es lento?" de debate recurrente en una condición medible.
- (−) La regla de vectorización exige diseño más cuidadoso por etapa (histéresis y dithering como operaciones de máscara, no condicionales por celda).
- (−) GIL limita el paralelismo del orquestador; mitigado porque las colas entre etapas mueven arrays grandes (el trabajo está en C/CUDA) y ffmpeg/ORT liberan el GIL.
