# ADR-005 — Ollama solo para análisis por escena, nunca en el hot path

**Estado**: Aceptado — 2026-07-15

## Contexto

La máquina de referencia tiene un stack Ollama operativo con `minicpm-v4.5` (VLM, 6.1 GB) entre otros modelos. La tentación obvia: usar el VLM para "entender" los frames y mejorar la conversión. Los números lo prohíben para el caso por-frame: un VLM local responde en cientos de ms a segundos por imagen (~30-150× el presupuesto de 10 ms del hot path, [04 §1](../04-ai-components.md)) y con el modelo cargado consume ~6-7 GB de los 16 GB de VRAM que el pipeline comparte.

A la vez, hay una tarea donde un VLM aporta algo que ningún componente del pipeline tiene: **comprensión semántica de la escena** ("retrato con poca luz", "paisaje de alto contraste") para sugerir preset, rampa y modo de color — la capa de Description de [IDEA.md](../../IDEA.md), donde el sistema comunica decisiones en vocabulario humano.

## Decisión

- Ollama participa en exactamente un componente: el **Scene Analyst** ([04 §6](../04-ai-components.md)) — un keyframe por escena detectada, respuesta JSON estructurada, sugerencia de preset por escena.
- Corre asíncrono y best-effort: timeout de 30 s, cualquier fallo (daemon caído, JSON inválido, VRAM presionada por otro modelo cargado) degrada a "sin sugerencia" con un warning por job. El export nunca lo espera.
- Antes de activarse verifica presión de VRAM (`/api/ps` + `nvidia-smi`) y se auto-desactiva si no hay hueco: el hot path tiene prioridad absoluta sobre el análisis.
- Ningún dato del Analyst entra en la CharMatrix por frame; su granularidad máxima de efecto es "parámetros de preset por escena". Esto preserva la reproducibilidad del core: con `--auto` desactivado, el output es independiente de si Ollama existe.

## Consecuencias

- (+) El VLM local aporta valor donde su perfil de latencia lo permite, sin contaminar el presupuesto del hot path.
- (+) El sistema funciona idéntico sin Ollama instalado — coherente con la degradación de [ADR-002](./ADR-002-deterministic-core.md).
- (−) El modo `--auto` introduce no-determinismo entre corridas (el VLM puede sugerir distinto); se acepta porque es opt-in y opera a granularidad de escena. La garantía bit a bit aplica al modo por defecto.
- (−) Doble fuente de verdad de VRAM (Ollama administra la suya); mitigado con la verificación previa y la prioridad declarada.
