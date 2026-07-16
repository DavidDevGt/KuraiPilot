"""Etapa 7 — Estabilidad temporal / anti-flicker (docs/02 E7).

Nivel base (siempre activo): histéresis por celda contra luma_committed (la luma
del último cambio, no la del frame anterior — evita drift). Reset en corte de escena.
Nivel avanzado (Fase 2): el mapa de histéresis se warpea con optical flow (ai/flow.py).

La métrica que gobierna esta etapa es FCR (docs/06 §3): target ≤ 0.05 en estático.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

HYSTERESIS_FACTOR = 0.6  # h = 0.6 × ancho del nivel de cuantización (docs/02 E7)


class HysteresisState:
    """Estado por job: luma_committed y char_idx comprometidos por celda."""

    def __init__(self, rows: int, cols: int) -> None:
        self.luma_committed: npt.NDArray[np.float32] = np.full((rows, cols), -1.0, dtype=np.float32)
        self.char_committed: npt.NDArray[np.uint8] = np.zeros((rows, cols), dtype=np.uint8)

    def reset(self) -> None:
        """Corte de escena: el siguiente frame se comporta como frame 0."""
        self.luma_committed.fill(-1.0)

    def apply(
        self,
        luma_grid: npt.NDArray[np.float32],
        char_idx: npt.NDArray[np.uint8],
        levels: int,
    ) -> npt.NDArray[np.uint8]:
        """Devuelve char_idx estabilizado y actualiza el estado. Vectorizado (máscaras)."""
        raise NotImplementedError("Fase 0")
