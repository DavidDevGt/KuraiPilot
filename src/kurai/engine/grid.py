"""Etapa 2 — Resize a grilla con corrección de aspecto (docs/02 E2).

Celda de referencia: 8×16 px (ratio 1:2). El decode ya entrega el frame a
resolución de trabajo (rows*16, cols*8); acá cada celda se colapsa a luma y
color promedio. Luminancia BT.709 calculada ANTES del promedio. Determinista.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

CELL_W = 8
CELL_H = 16

# BT.709: el verde pesa ~10× el azul en la percepción de brillo
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def grid_shape(video_w: int, video_h: int, cols: int) -> tuple[int, int]:
    """(rows, cols) preservando aspecto con celdas 1:2."""
    rows = round(video_h / video_w * cols * (CELL_W / CELL_H))
    return max(rows, 1), cols


def work_resolution(rows: int, cols: int) -> tuple[int, int]:
    """(width, height) de trabajo que el decode debe entregar."""
    return cols * CELL_W, rows * CELL_H


def to_grids(
    frame_rgb: npt.NDArray[np.uint8], rows: int, cols: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """→ (luma_grid (rows, cols), color_grid (rows, cols, 3)), float32 en [0,1].

    La reducción corre en suma ENTERA de una pasada (el float toca solo el
    array chico de celdas): ~7× más rápido que promediar en float a resolución
    de trabajo. Promediar RGB y luego pesar es idéntico a pesar y luego
    promediar porque la luma BT.709 es lineal en RGB — la regla "luma antes
    del promedio" (docs/02 E2) solo prohíbe lumas no lineales.
    """
    expected = (rows * CELL_H, cols * CELL_W, 3)
    if frame_rgb.shape != expected:
        raise ValueError(f"Frame {frame_rgb.shape} ≠ resolución de trabajo {expected}")
    sums = frame_rgb.reshape(rows, CELL_H, cols, CELL_W, 3).sum(axis=(1, 3), dtype=np.uint32)
    color_grid = sums.astype(np.float32) * np.float32(1.0 / (CELL_H * CELL_W * 255))
    luma_grid = color_grid @ LUMA_WEIGHTS
    return luma_grid.astype(np.float32), color_grid
