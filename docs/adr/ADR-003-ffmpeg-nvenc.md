# ADR-003 — ffmpeg como única frontera de video, con NVDEC/NVENC

**Estado**: Aceptado — 2026-07-15

## Contexto

"Cualquier video" como contrato de entrada (G1) exige la matriz de compatibilidad de códecs/contenedores más amplia que exista — que es ffmpeg y nada más. Para el I/O de video había tres opciones: OpenCV `VideoCapture` (compatibilidad limitada, poco control de hwaccel), PyAV (bindings nativos de libav, pero amarra la versión de libav a la wheel y el camino NVDEC es frágil), o ffmpeg como subprocess con pipes rawvideo. La máquina de referencia tiene NVDEC/NVENC (RTX 5070 Ti) y el driver 595 con CUDA 13.2.

## Decisión

- Todo decode y encode pasa por **ffmpeg como subprocess**, con pipes rawvideo hacia/desde el Engine. Nada más del sistema abre archivos de video.
- Hardware acceleration por defecto: `-hwaccel cuda` + `scale_cuda` en decode (el downscale a resolución de trabajo ocurre en GPU antes de cruzar PCIe), `h264_nvenc` en encode. Fallback automático a software si el códec no tiene soporte de hardware, sin cambio de comportamiento observable.
- Metadatos vía `ffprobe -print_format json`. Audio extraído con `-c:a copy` y muxeado sin recodificar ([02 E1/E9](../02-pipeline-spec.md)).
- VFR se normaliza a CFR en decode para mantener la relación 1:1 frame↔CharMatrix.

## Consecuencias

- (+) La matriz de compatibilidad de inputs es la de ffmpeg del sistema y mejora con `apt upgrade`, sin re-release nuestro.
- (+) Decode 4K y encode salen del presupuesto de CPU/PCIe (ASICs dedicados); es la base del target 4× de [05](../05-performance-and-capacity.md).
- (+) El subprocess aísla crashes de demuxer (input corrupto tira el ffmpeg, no el Engine; se reporta limpio).
- (−) Dependencia de runtime que hay que verificar al arranque (`ffmpeg -hwaccels`, presencia de nvenc) con mensajes de error accionables.
- (−) Los pipes rawvideo cuestan una copia de memoria por frame; aceptado — medido contra el resto del presupuesto es ruido a resolución de trabajo.
- (−) Sin control frame-exacto de seeking fino (aceptable: el pipeline es streaming secuencial, no editor).
