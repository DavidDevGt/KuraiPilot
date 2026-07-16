"""Etapa 7 nivel avanzado — Optical flow para histéresis warpeada (docs/04 §4). Fase 2.

Escalera: (1) sin flow → (2) Farneback OpenCV CUDA sobre la grilla → (3) RAFT-small
solo si Farneback falla la métrica. Hoy (3) es hipótesis, no plan.
El flow desplaza el mapa de histéresis, no los píxeles. Resolución = grilla.
Gate: FCR en fixture de paneo mejora ≥50% sobre histéresis sola, o no entra.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def farneback_grid_flow(
    prev_luma: npt.NDArray[np.float32], cur_luma: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """→ flow (rows, cols, 2) en unidades de celda."""
    raise NotImplementedError("Fase 2")


def warp_state(
    state: npt.NDArray[np.float32], flow: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Desplaza luma_committed siguiendo el flow (nearest, sin interpolación de luma)."""
    raise NotImplementedError("Fase 2")
