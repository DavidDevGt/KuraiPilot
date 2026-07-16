"""Scene Analyst — minicpm-v4.5 vía Ollama (docs/04 §6, ADR-005). Fase 3.

Límites NO negociables (ADR-005):
- Un keyframe por escena. NUNCA por frame (30-150× el presupuesto del hot path).
- Best-effort: timeout 30 s; cualquier fallo → sin sugerencia + 1 warning por job.
- El export JAMÁS espera al Analyst (corre async en paralelo).
- Verifica presión de VRAM (/api/ps + nvidia-smi) y se auto-desactiva si no hay hueco.
- Su efecto máximo es parámetros de preset POR ESCENA; nada entra a la CharMatrix
  por frame. Sin --auto, el output es independiente de que Ollama exista.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

TIMEOUT_S = 30.0
VISION_MODEL = "minicpm-v4.5"


@dataclass(frozen=True)
class SceneSuggestion:
    scene_type: str
    main_subject: str
    lighting: str
    ramp: str
    color_mode: str
    saliency_on: bool


class SceneAnalyst:
    def __init__(self, ollama_url: str) -> None:
        raise NotImplementedError("Fase 3")

    def vram_headroom_ok(self) -> bool:
        """False → auto-desactivación con warning; el hot path tiene prioridad."""
        raise NotImplementedError("Fase 3")

    async def analyze_keyframe(self, frame_rgb: npt.NDArray[np.uint8]) -> SceneSuggestion | None:
        """None en cualquier fallo (timeout, JSON inválido, daemon caído)."""
        raise NotImplementedError("Fase 3")
