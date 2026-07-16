# ADR-001 — Local-first: sin backend, sin nube

**Estado**: Aceptado — 2026-07-15

## Contexto

IDEA.md dejaba abierta la pregunta "¿server-side, client-side o híbrido?" pensando en un producto web público. El entorno real del proyecto es una workstation con RTX 5070 Ti (16 GB), 59 GB de RAM y un stack Ollama local ya operativo. No hay un segundo usuario, no hay infraestructura de servidor, y los costos de cómputo de los modos con IA a escala eran uno de los riesgos abiertos.

## Decisión

Todo el sistema corre en la máquina local. El "híbrido" de IDEA.md se reinterpreta dentro de la misma máquina: preview ligero en el navegador (servido desde `127.0.0.1`), export pesado en el proceso local con la GPU. Ningún frame, video ni metadato sale de la máquina.

## Consecuencias

- (+) El riesgo "costo de infraestructura / límites de tier gratuito" de IDEA.md desaparece entero.
- (+) Diligence simplificada: no alojamos ni redistribuimos contenido de terceros — el video nunca abandona la máquina de quien tiene el archivo.
- (+) La GPU local es mejor que cualquier tier gratuito de nube que el proyecto podría pagar.
- (−) Los benchmarks y gates de performance solo son válidos en la máquina de referencia; no hay CI con GPU ([06 §5](../06-testing-and-evaluation.md) lo asume).
- (−) Distribuir a otros usuarios exigirá empaquetado y tolerancia a hardware diverso — explícitamente fuera de roadmap ([07](../07-roadmap.md)).
- Convertir esto en servicio público requeriría un ADR que supersede a este; no es una evolución incremental.
