# 07 — Roadmap

Fases con criterios de aceptación verificables. Cada fase termina con un gate explícito; no se abre la siguiente con el gate anterior en rojo. Las fases refinan las de [IDEA.md](../IDEA.md) con el detalle de arquitectura ya decidido.

## Fase 0 — Core determinista (el producto mínimo honesto)

**Alcance**: etapas 1, 2, 4, 6-Bayer, 7-histéresis, 8, 9 ([02](./02-pipeline-spec.md)). CLI `kurai convert`. Preset `retro` completo. Calibración de rampa (`tools/calibrate_ramp.py`) y elección/fijado de la fuente de referencia.

**Gate de salida** (✅ completado 2026-07-16):
- [x] Un mp4 1080p30 arbitrario produce un mp4 ASCII con audio intacto (hash de stream de audio idéntico — `test_convert_e2e_with_audio`).
- [x] Golden files + garantía de reproducibilidad bit a bit en verde ([06 §1](./06-testing-and-evaluation.md) — `test_gradient_golden`, `test_bit_exact_reproducibility`).
- [x] FCR ≤ 0.05 en el fixture estático ([06 §3](./06-testing-and-evaluation.md) — FCR medido: 0 exacto, `test_fcr_zero_on_noisy_static`).
- [x] Speed factor ≥ 4× en `retro` sobre la máquina de referencia ([05 §2](./05-performance-and-capacity.md)): **12.9×** con NVENC (`bench/results/accepted.json`). Nota: exigió fusionar E1+E2 (scale=area en ffmpeg cuando ninguna etapa necesita píxeles a resolución de trabajo) — la versión con reducción en NumPy daba 1.5×.
- [x] Círculo de prueba sale circular (corrección de aspecto — `test_circle_stays_circular`).

**Riesgo principal**: el overhead de transferencias y del intérprete ([05 §3](./05-performance-and-capacity.md)). Mitigación: el bench se construye *primero*, con el pipeline vacío (decode→encode passthrough), para conocer el techo de la infraestructura antes de escribir etapas.

## Fase 0.5 — Preview y terminal live

**Alcance**: preview server + frontend WebGL ([03 §3](./03-tech-stack.md)); modo `kurai live` (ANSI a stdout). Ambos consumen el mismo core de mapeo que el export.

**Gate** (✅ completado 2026-07-16):
- [x] La CharMatrix del preview es idéntica a la del export con la misma config — por construcción (`PreviewSession.compute` usa `cells_to_charmatrix`, el mismo código de `run_job`) y verificado frame a frame en `test_preview_charmatrix_identical_to_export_path`.
- [x] Cambio de rampa/gamma/color refleja en <100 ms: recompute desde el cell-frame cacheado, medido en el peor caso (grilla 200 cols) por `test_config_change_under_100ms`. El cambio de columnas re-decodifica con seek (`-ss`), único ajuste que excede el presupuesto — documentado, no ocultado.
- [x] `kurai live`: ANSI con run-length de color, pacing con drop de frames, alt-screen con restore garantizado incluso ante fallo a mitad de stream.

## Fase 1 — Saliencia

**Alcance**: Etapa 3 con U2Net-lite ONNX ([04 §2](./04-ai-components.md)), scheduling cada 5 frames, preset `detallado` (incluye E5-`edges`, que es determinista y entra aquí). `tools/fetch_models.py` con hashes pineados. Degradación limpia sin modelos.

**Gate** (implementación completa; el A/B es de juicio humano por diseño):
- [ ] A/B ≥ 60% de preferencia con vs. sin saliencia en el set curado ([06 §4](./06-testing-and-evaluation.md)) — **el gate falló**: 14% de preferencia por `detallado` (1/7) con `tools/ab_review.py`, muy por debajo del 60%. Veredicto NO-GO registrado en [docs/evaluations/2026-07-18-ab-saliencia.md](./evaluations/2026-07-18-ab-saliencia.md) — cerrado con 1 sesión ciega (n=7) en vez de las ≥3 del protocolo, desviación explícita justificada por la distancia al umbral, no oculta. La apuesta central de IDEA.md (densidad no uniforme mejora la percepción) no se sostuvo sobre el set curado; qué hacer con saliencia/edges de acá en más (abandonar, ajustar, re-evaluar) queda como decisión de producto separada.
- [x] Speed factor ≥ 2× en `detallado`: **3.6×** con inferencia GPU (CUDAExecutionProvider) en la máquina de referencia (jellyfish 1080p; U2Net-lite cada 5 frames domina el costo). Con inferencia CPU baja a 1.4× — el gate exige el path GPU (extra `gpu`, `onnxruntime-gpu`). Clips muy densos (1080p60, vertical 30fps) quedan en ~2.0×.
- [x] Job completa con warning si los modelos no están: `test_detallado_degrades_without_model` en verde — sin ONNX u onnxruntime, `density ≡ 1.0` y la salida es la determinista, con `UserWarning` (regla 5).

## Fase 2 — Alta fidelidad

**Alcance**: CNN de glifos (entrenamiento + export ONNX, `training/`), FS dithering, Farneback flow para histéresis warpeada, color `fg+bg`. Preset `alta-fidelidad` completo.

**Gate**:
- [ ] CNN: mejora perceptible en subset textura **y** ≤ 3 ms/frame p95, o se elimina ([04 §3](./04-ai-components.md) — la eliminación es un resultado aceptable de la fase, no un fracaso).
- [ ] Flow: FCR en fixture de paneo mejora ≥ 50% sobre histéresis sola, o el flow no entra.
- [ ] Speed factor ≥ 0.8×.

## Fase 3 — Scene Analyst y export enriquecido

**Alcance**: integración Ollama/`minicpm-v4.5` por escena con contrato de fallo completo ([04 §6](./04-ai-components.md)); flag `--auto`. Exports adicionales según demanda real: asciinema `.cast`, HTML embebible (reutiliza el frontend del preview), SVG de frame.

**Gate**: los tests de contrato del Analyst (timeout, JSON inválido, VRAM presionada, Ollama caído) en verde; el export nunca se bloquea esperando al Analyst.

## Fuera de roadmap (decisión, no olvido)

- Superresolución de entrada — hasta tener casos reales de input de baja resolución ([04 §7](./04-ai-components.md)).
- RAFT-small para flow — hasta que Farneback falle una métrica.
- Servicio web público, multi-tenant — contradice [ADR-001](./adr/ADR-001-local-first.md); requeriría re-abrir esa decisión con un ADR nuevo.
- Empaquetado/distribución (PyPI, AppImage) — cuando exista un segundo usuario.

## Secuencia crítica

```
Fase 0 ──▶ Fase 0.5 ──▶ Fase 1 ──▶ Fase 2 ──▶ Fase 3
  │            │           │
  bench      preview    el A/B decide si la apuesta
  primero    valida UX  central del producto es real
```

La dependencia dura es Fase 0 → todo lo demás. Fase 0.5 y Fase 1 podrían solaparse (tocan módulos disjuntos), pero el A/B de Fase 1 necesita el preview de 0.5 para revisar pares cómodamente — por eso el orden.
