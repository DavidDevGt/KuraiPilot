# KuraiPilot — Documentación de arquitectura

Sistema local-first de conversión de video a video ASCII renderizado. Este directorio contiene la documentación técnica de nivel arquitectura; la visión de producto y la investigación de estado del arte viven en la raíz del repo ([IDEA.md](../IDEA.md), [INVESTIGATION.md](../INVESTIGATION.md)).

## Mapa de documentos

| Doc | Qué responde |
|---|---|
| [01-architecture-overview.md](./01-architecture-overview.md) | Qué es el sistema, sus límites, componentes y cómo fluyen los datos. |
| [02-pipeline-spec.md](./02-pipeline-spec.md) | Especificación etapa por etapa del pipeline de conversión: contratos, algoritmos, garantías. |
| [03-tech-stack.md](./03-tech-stack.md) | Stack elegido y su mapeo al hardware/entorno de desarrollo real. |
| [04-ai-components.md](./04-ai-components.md) | Los componentes con IA: modelos, presupuesto de VRAM, scheduling, fallbacks. |
| [05-performance-and-capacity.md](./05-performance-and-capacity.md) | Targets de rendimiento, presupuestos por etapa, metodología de benchmark. |
| [06-testing-and-evaluation.md](./06-testing-and-evaluation.md) | Estrategia de testing: golden frames, métrica de flicker, evaluación A/B, regresiones. |
| [07-roadmap.md](./07-roadmap.md) | Fases con criterios de aceptación verificables. |
| [adr/](./adr/) | Architecture Decision Records — el porqué de cada decisión estructural. |

## Cómo leer esta documentación

- Si sos nuevo en el proyecto: `01 → 02 → 07`.
- Si vas a implementar una etapa del pipeline: `02` + el ADR que la cubre.
- Si vas a tocar cualquier componente con IA: `04` es normativo — define qué modelo corre dónde y con qué presupuesto; no se agrega un modelo al hot path sin actualizar ese doc y el ADR correspondiente.
- Los ADRs son inmutables una vez aceptados; una decisión se revierte con un ADR nuevo que supersede al anterior, no editando el viejo.

## Principios rectores (resumen)

1. **Determinista por defecto, IA por elección** — el pipeline completo funciona y produce salida de calidad con todos los componentes de IA apagados ([ADR-002](./adr/ADR-002-deterministic-core.md)).
2. **Local-first** — todo corre en la máquina del usuario; ningún frame sale de la máquina ([ADR-001](./adr/ADR-001-local-first.md)).
3. **La GPU es para frames, no para chat** — inferencia por-frame solo para modelos pequeños optimizados; los modelos de Ollama quedan fuera del hot path ([ADR-005](./adr/ADR-005-ollama-role.md)).
4. **Mismo input + misma config ⇒ mismo output**, bit a bit en la capa de texto. La reproducibilidad es un feature, no un accidente.
