# ADR-002 — Core determinista; IA opcional, acotada y desactivable

**Estado**: Aceptado — 2026-07-15

## Contexto

La investigación ([INVESTIGATION.md §4](../../INVESTIGATION.md)) mostró que el mapeo video→ASCII es una función bien definida de la entrada: no hay ambigüedad creativa que un modelo generativo deba resolver. A la vez, dos mejoras con modelo (saliencia, estabilidad temporal avanzada) tienen impacto visual real que ningún método determinista alcanza.

## Decisión

1. El pipeline completo funciona y produce salida de calidad competitiva con **todos** los componentes de IA apagados; ese es el preset por defecto.
2. Cada componente de IA es opt-in por preset, tiene un equivalente determinista de degradación, y un criterio de aceptación medible contra el baseline sin IA ([04](../04-ai-components.md), [06](../06-testing-and-evaluation.md)). Si no supera su gate, se elimina.
3. Prohibido: modelos generativos (difusión/GAN/VLM) generando, retocando o "mejorando" el output ASCII, en cualquier etapa.
4. Reproducibilidad como contrato: mismo input + misma config ⇒ misma CharMatrix, bit a bit, entre corridas.

## Consecuencias

- (+) El producto nunca depende de que un modelo cargue para funcionar; el fallo de IA degrada, no aborta.
- (+) Cada componente de IA carga con la obligación de demostrar su valor con métrica o A/B — aplicación directa de Discernment ([IDEA.md](../../IDEA.md)): no se agrega IA para poder decir que hay IA.
- (+) La reproducibilidad habilita golden-file testing exacto ([06 §1](../06-testing-and-evaluation.md)).
- (−) Los componentes de IA con no-determinismo numérico inherente (orden de reducción en GPU) quedan fuera de la garantía bit a bit; se acepta y se testean por métrica, no por golden.
- (−) Renunciamos de antemano a explorar estilización generativa (p.ej. "ASCII estilo pintura") — sería otro producto.
