# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/es/1.1.0/). Versionado: SemVer una vez que exista la primera release; hasta entonces, `0.1.0-dev` en `main`.

## [Unreleased]

### Added

- Documentación de arquitectura completa (`docs/` + 6 ADRs) e investigación de estado del arte.
- Esqueleto tipado del pipeline con contratos por etapa y stubs por fase del roadmap.
- CLI `kurai` con `doctor` funcional (verifica ffmpeg/NVDEC/NVENC/GPU/Ollama).
- Presets `retro`, `detallado`, `alta-fidelidad` validados por Pydantic.
- CI en GitHub Actions (CPU-only) y suite por módulo con Hypothesis y fixtures lavfi.
- Fronteras de arquitectura ejecutables con import-linter.
- Infraestructura multi-equipo: CONTRIBUTING, CODEOWNERS, plantillas de PR/issues.

### Fixed

- `kurai --version` no funcionaba sin subcomando.
