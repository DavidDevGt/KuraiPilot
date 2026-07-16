"""Etapa 2 — Resize a grilla con corrección de aspecto (docs/02 E2).

Celda de referencia: 8×16 px (ratio 1:2). Area-average (INTER_AREA); luminancia
BT.709 calculada ANTES del promedio por celda. Determinista.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

CELL_W = 8
CELL_H = 16


def grid_shape(video_w: int, video_h: int, cols: int) -> tuple[int, int]:
    """(rows, cols) preservando aspecto con celdas 1:2."""
    rows = round(video_h / video_w * cols * (CELL_W / CELL_H))
    return max(rows, 1), cols


def to_grids(
    frame_rgb: npt.NDArray[np.uint8], rows: int, cols: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """→ (luma_grid (rows, cols), color_grid (rows, cols, 3)), ambos float32 en [0,1]."""
    raise NotImplementedError("Fase 0")
