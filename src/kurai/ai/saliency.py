"""Etapa 3 — Saliencia con U2Net-lite ONNX (docs/04 §2, ADR-004). Fase 1.

Entrada 320×320 norm ImageNet; cada N=5 frames con propagación entre corridas;
inferencia forzada en corte de escena. Post: blur gaussiano σ=2 celdas.
Gate de aceptación: A/B ≥ 60% (docs/07 Fase 1) — si no lo pasa, no se promociona.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

INFER_EVERY_N = 5
INPUT_SIZE = 320


class SaliencyModel:
    def __init__(self, model_path: Path) -> None:
        raise NotImplementedError("Fase 1")

    def infer(
        self, frame_rgb: npt.NDArray[np.uint8], rows: int, cols: int
    ) -> npt.NDArray[np.float32]:
        """→ density_map (rows, cols) en [0,1], ya blureado."""
        raise NotImplementedError("Fase 1")
