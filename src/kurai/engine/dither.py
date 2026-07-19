"""Etapa 6 — Dithering (docs/02 E6). Determinista.

Bayer 8×8: paralelizable, patrón estable entre frames (cero flicker inducido).
Floyd-Steinberg serpentine: secuencial, CPU; siempre combinado con E7.
Se implementa fusionado con E4 (ajusta el error de luma antes del lookup final).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Matriz de Bayer 8×8 canónica (índices 0..63)
_BAYER_8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float32,
)


def bayer_offsets(rows: int, cols: int, levels: int) -> npt.NDArray[np.float32]:
    """Umbrales Bayer 8×8 tileados a (rows, cols), centrados en cero y escalados
    al ancho de un nivel de cuantización. Estáticos entre frames por construcción
    (cero flicker inducido)."""
    normalized = (_BAYER_8 + 0.5) / 64.0 - 0.5  # (-0.5, 0.5)
    offsets = normalized / levels
    reps_r = -(-rows // 8)  # ceil
    reps_c = -(-cols // 8)
    return np.tile(offsets, (reps_r, reps_c))[:rows, :cols].astype(np.float32)


def floyd_steinberg(luma_grid: npt.NDArray[np.float32], levels: int) -> npt.NDArray[np.float32]:
    """Luma con error difundido (serpentine). Única excepción a la regla de
    vectorización (ADR-006): secuencial por naturaleza; Numba si el puro es lento.

    Devuelve el CENTRO del bin elegido por celda ((k+0.5)/levels), no la luma
    ajustada cruda: así el floor de quantize() reproduce exactamente el nivel
    que el FS eligió, sin riesgo de cruzar un borde de bin al redondear el
    double interno a float32. El loop corre sobre floats de Python (IEEE-754
    double): determinista bit a bit en cualquier máquina.
    """
    rows, cols = int(luma_grid.shape[0]), int(luma_grid.shape[1])
    inv = 1.0 / levels
    top = levels - 1
    grid: list[list[float]] = luma_grid.astype(np.float64).tolist()
    out: list[list[float]] = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        row = grid[r]
        below = grid[r + 1] if r + 1 < rows else None
        step = 1 if r % 2 == 0 else -1
        rng = range(cols) if step == 1 else range(cols - 1, -1, -1)
        for c in rng:
            old = row[c]
            k = int(old * levels)
            if old * levels < k:  # int() trunca hacia 0; corregir a floor
                k -= 1
            if k < 0:
                k = 0
            elif k > top:
                k = top
            center = (k + 0.5) * inv
            out[r][c] = center
            err = old - center
            ahead = c + step
            behind = c - step
            if 0 <= ahead < cols:
                row[ahead] += err * 0.4375  # 7/16
            if below is not None:
                if 0 <= behind < cols:
                    below[behind] += err * 0.1875  # 3/16
                below[c] += err * 0.3125  # 5/16
                if 0 <= ahead < cols:
                    below[ahead] += err * 0.0625  # 1/16
    return np.asarray(out, dtype=np.float32)
