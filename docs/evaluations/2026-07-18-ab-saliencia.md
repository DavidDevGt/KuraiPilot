# A/B saliencia — 2026-07-18

- Commit: `3e6beb6`
- Votos decididos: 7 (empates: 0, saltados: 0)
- Evaluadores: ana (1)
- Preferencia por `detallado` (saliencia): **14%** (1/7)
- Gate (≥60%, ≥3 evaluadores o sesiones ciegas): ❌ NO PASA

## Por clip

| Clip | detallado | retro | empate |
|---|---|---|---|
| artemis_launch | 0 | 1 | 0 |
| bbb_1080p_10s | 0 | 1 | 0 |
| jellyfish_1080p_10s | 0 | 1 | 0 |
| notld_1968_60s | 0 | 1 | 0 |
| portrait_test | 0 | 1 | 0 |
| sintel_trailer_720p | 0 | 1 | 0 |
| tears_of_steel_60s | 1 | 0 | 0 |

## Decisión

**NO-GO.** 14% de preferencia por `detallado` (1/7), muy por debajo del 60% del
gate ([docs/07](../07-roadmap.md) Fase 1) — la hipótesis central de IDEA.md
(densidad no uniforme por saliencia mejora la percepción) no se sostiene sobre
este set curado.

Cierre deliberado con **1 sesión ciega (n=7)** en vez de las ≥3 que pide el
protocolo ([docs/06 §4](../06-testing-and-evaluation.md)): la señal es lo
bastante lejos del umbral (14% vs. 60%) como para no justificar el tiempo de
2 sesiones más solo para confirmar la misma dirección. Se documenta como
desviación explícita del protocolo, no como su cumplimiento.

Saliencia (E3) y edges (E5) quedan en el código como estaban — el gate fallado
no dispara por sí solo un revert; decidir si se abandona el componente, se
ajusta (p. ej. modular la intensidad de la modulación por densidad en vez de
todo-o-nada) o se re-evalúa con más evaluadores es una decisión de producto
aparte, pendiente de ADR si implica remover el componente (regla dura 4).
