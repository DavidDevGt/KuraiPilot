"""Etapa 6 — Dithering (docs/02 E6). Determinista.

Bayer 8×8: paralelizable, patrón estable entre frames (cero flicker inducido).
Floyd-Steinberg serpentine: secuencial, CPU; siempre combinado con E7.
Se implementa fusionado con E4 (ajusta el error de luma antes del lookup final).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def bayer_offsets(rows: int, cols: int, levels: int) -> npt.NDArray[np.float32]:
    """Matriz de umbrales Bayer 8×8 tileada a (rows, cols), escalada al ancho de nivel."""
    raise NotImplementedError("Fase 0")


def floyd_steinberg(luma_grid: npt.NDArray[np.float32], levels: int) -> npt.NDArray[np.float32]:
    """Luma con error difundido (serpentine). Única excepción a la regla de
    vectorización (ADR-006): secuencial por naturaleza; Numba si el puro es lento."""
    raise NotImplementedError("Fase 2")
