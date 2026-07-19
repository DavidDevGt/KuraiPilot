# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.1.0/). Versionado: SemVer una vez que exista la primera release; hasta entonces, `0.1.0-dev` en `main`.

## [Unreleased]

### Added

- **Fase 2 parcial — Floyd-Steinberg (E6) + color fg+bg (E8)**, preset `nitido`. FS serpentine determinista (secuencial en CPU, la excepción sancionada por ADR-006; ~3 ms/frame en grilla 160×90, sin Numba) reemplaza a Bayer: devuelve centros de bin exactos, así `quantize` reproduce el nivel elegido sin riesgo de borde por redondeo double→float32. `fg+bg`: el decode entrega la grilla a `rows*2` (dos muestras verticales por celda, "semi-bloque"); `fg` = mitad superior, `bg` = inferior, el camino tonal corre sobre el promedio y la composición mezcla tinta/fondo por el mask del atlas (contrato nuevo en docs/02 E8; `CharMatrix.bg` deja de ser `None` en este modo). `nitido` = máximo detalle determinista (edges on, saliencia off tras el NO-GO del A/B): **1.8× tiempo real** en la máquina de referencia. `live` degrada fg+bg→fg con warning (regla 5); el golden de `retro` quedó byte-idéntico; golden nuevo `gradient_nitido_40x10.npz`. Restan de Fase 2: CNN de glifos y Farneback flow.

- **Fase 1 — Saliencia (E3) + bordes (E5)**, preset `detallado`. La densidad de detalle deja de ser uniforme: U2Net-lite (ONNX, ~4.6 MB, pineado por SHA-256 en `models/manifest.toml`) infiere un mapa de saliencia cada 5 frames (con propagación e inferencia forzada en corte de escena) que modula la rampa efectiva por celda — el sujeto usa la rampa completa, el fondo colapsa a 4 niveles. E5-`edges` (Sobel determinista, vectorizado) reemplaza el carácter tonal por un glifo direccional (`/ \ | - _`, bitmaps embebidos nuevos) en los bordes estructurales. `detallado` corre a **3.6× tiempo real** con GPU en la máquina de referencia. Degradación limpia: sin modelo u onnxruntime, `density ≡ 1.0` con warning y la salida es la determinista, bit a bit (los golden de `retro` no se tocan). El A/B ≥60% (docs/07) resultó NO-GO: 14% de preferencia por `detallado` (1/7, `tools/ab_review.py`) — veredicto en `docs/evaluations/2026-07-18-ab-saliencia.md`.

- **Fase 0.5 completa** (gate en verde, docs/07): `kurai preview` — server FastAPI+WebSocket en localhost que streamea la CharMatrix (no píxeles) a un cliente WebGL2 de un solo pass, con sliders de columnas/rampa/gamma/color, play/pause/seek; ajustes de rampa/gamma/color reflejan en <100 ms y la CharMatrix del preview es bit a bit la del export (mismo código, test de igualdad). `kurai live` — reproducción ANSI en terminal con run-length de color 24-bit, pacing a fps con drop de frames y restore de terminal garantizado. `iter_frames` gana seek (`start_s`).

- **Fase 0 completa** (gate en verde, docs/07): `kurai convert` produce video ASCII real con preset `retro` — decode/demux (E1), grilla con corrección de aspecto 1:2 (E2), rampa calibrada por cobertura de tinta con glifos bitmap 8×16 embebidos, mapeo+Bayer fusionados (E4/E6), anti-flicker por histéresis con FCR=0 medido (E7), render por atlas (E8), encode NVENC con audio bit-idéntico (E9). 12.9× tiempo real en la máquina de referencia (fast path E1+E2 fusionadas vía scale=area).
- `kurai bench` con modos passthrough y retro, baselines versionados y `--check` de regresión.

- Documentación de arquitectura completa (`docs/` + 6 ADRs) e investigación de estado del arte.
- Esqueleto tipado del pipeline con contratos por etapa y stubs por fase del roadmap.
- CLI `kurai` con `doctor` funcional (verifica ffmpeg/NVDEC/NVENC/GPU/Ollama).
- Presets `retro`, `detallado`, `alta-fidelidad` validados por Pydantic.
- CI en GitHub Actions (CPU-only) y suite por módulo con Hypothesis y fixtures lavfi.
- Fronteras de arquitectura ejecutables con import-linter.
- Infraestructura multi-equipo: CONTRIBUTING, CODEOWNERS, plantillas de PR/issues.

### Changed

- **Umbral de histéresis (E7) de 0.6 a 1.5** (`kurai/engine/stability.py`): medido sobre metraje real con grano/textura (película de 1968, ruido de codec), el umbral viejo dejaba ~30-35% de la grilla cambiando de carácter cada frame — perceptible como parpadeo/distorsión incómoda en escenas alejadas o con mucho detalle, pese a que el gate sintético (docs/06 §3) daba FCR=0. El fixture del gate era demasiado limpio para exponerlo. El nuevo valor corta ese ruido a la mitad (~0.16 FCR) sin costo medible en contenido limpio con movimiento real de cámara (Sintel: FCR 0.025→0.016). No toca `cols` ni ningún otro default de preset — mejora de calidad "gratis" en el mismo ancho de grilla.

### Fixed

- `kurai --version` no funcionaba sin subcomando.
- Auditoría del motor de video contra mejores prácticas (docs/02 E1/E9): deadlock latente de `stderr=PIPE` sin lector en decode/encode (ahora a tempfile), NVENC con bitrate capado por falta de `-b:v 0` en modo CQ, colores corridos por matriz BT.601 default de swscale (ahora BT.709 explícito + VUI tagueado vía `setparams`), y drift de frame rate por pasar float en vez del racional exacto.
