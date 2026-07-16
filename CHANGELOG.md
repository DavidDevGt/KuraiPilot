# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.1.0/). Versionado: SemVer una vez que exista la primera release; hasta entonces, `0.1.0-dev` en `main`.

## [Unreleased]

### Added

- **Fase 0 completa** (gate en verde, docs/07): `kurai convert` produce video ASCII real con preset `retro` — decode/demux (E1), grilla con corrección de aspecto 1:2 (E2), rampa calibrada por cobertura de tinta con glifos bitmap 8×16 embebidos, mapeo+Bayer fusionados (E4/E6), anti-flicker por histéresis con FCR=0 medido (E7), render por atlas (E8), encode NVENC con audio bit-idéntico (E9). 12.9× tiempo real en la máquina de referencia (fast path E1+E2 fusionadas vía scale=area).
- `kurai bench` con modos passthrough y retro, baselines versionados y `--check` de regresión.

- Documentación de arquitectura completa (`docs/` + 6 ADRs) e investigación de estado del arte.
- Esqueleto tipado del pipeline con contratos por etapa y stubs por fase del roadmap.
- CLI `kurai` con `doctor` funcional (verifica ffmpeg/NVDEC/NVENC/GPU/Ollama).
- Presets `retro`, `detallado`, `alta-fidelidad` validados por Pydantic.
- CI en GitHub Actions (CPU-only) y suite por módulo con Hypothesis y fixtures lavfi.
- Fronteras de arquitectura ejecutables con import-linter.
- Infraestructura multi-equipo: CONTRIBUTING, CODEOWNERS, plantillas de PR/issues.

### Fixed

- `kurai --version` no funcionaba sin subcomando.
