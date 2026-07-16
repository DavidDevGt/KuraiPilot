# ADR-004 — U2Net-lite para saliencia; SAM queda fuera del hot path

**Estado**: Aceptado — 2026-07-15

## Contexto

La densidad de detalle no uniforme (Etapa 3) es la apuesta visual central del producto ([IDEA.md](../../IDEA.md), Fase 1). Necesita un mapa de saliencia por frame (amortizado cada N frames) dentro del presupuesto del hot path: ≤ 10 ms por invocación y ≤ 2 GB de VRAM agregada para todo el AI Sidecar ([04 §1](../04-ai-components.md)). INVESTIGATION.md mencionaba MobileSAM/U2Net como candidatos.

Lo que la etapa necesita es **un mapa continuo de importancia visual**, no una segmentación de instancias con clases ni máscaras editables:
- **U2Net-lite**: ~4.7 MB, salida directamente un mapa de saliencia [0,1], inferencia de un solo paso, ONNX maduro. Es exactamente la forma del dato que la etapa consume.
- **ISNet (DIS)**: sucesor con bordes más finos, mismo perfil de uso, algo más pesado.
- **SAM/MobileSAM**: produce máscaras de instancia binarias y necesita prompts o modo "everything" (caro); habría que post-procesar máscaras a mapa continuo. Más VRAM, más latencia, más piezas — para tirar la mayor parte de lo que computa.

## Decisión

U2Net-lite en ONNX Runtime (CUDA EP) como modelo de saliencia del hot path, entrada 320×320, cada N=5 frames con propagación entre corridas. ISNet es el upgrade pre-aprobado si la calidad de máscara falla el A/B de Fase 1 con evidencia de que el problema es el modelo (no el post-proceso). SAM en cualquier variante queda fuera del hot path; si algún día se quiere edición interactiva de la máscara en el preview ("clic para marcar el sujeto"), eso es un feature nuevo con su propio ADR.

## Consecuencias

- (+) Presupuesto sobrado: ~5 MB de pesos y una inferencia de un paso vs. el pipeline prompt+decoder de SAM.
- (+) Salida continua = el `density_map` sale casi directo (solo resize + blur, [04 §2](../04-ai-components.md)).
- (−) U2Net (2020) no es estado del arte en fineza de bordes; mitigado porque la salida se consume a resolución de grilla (160×90) — la fineza de borde a nivel píxel es irrelevante tras el downsample.
- (−) Sin conciencia semántica ("es un rostro"): saliencia pura puede elegir mal en escenas con varios focos. Si el A/B de Fase 1 falla por esto, la señal del Scene Analyst (que sí sabe qué es el sujeto) puede sesgar el mapa — extensión compatible con esta decisión.
