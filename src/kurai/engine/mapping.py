"""Etapa 4 — Mapeo luminancia → carácter (docs/02 E4). Determinista.

Las rampas NO se hardcodean acá: se calibran por cobertura de tinta real del
glifo con tools/calibrate_ramp.py y el resultado se versiona en render/ramps.py.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.config import Ramp


def map_luma(
    luma_grid: npt.NDArray[np.float32],
    ramp: Ramp,
    gamma: float,
    density_map: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.uint8]:
    """→ char_idx (rows, cols). density_map modula la longitud efectiva de rampa (E3)."""
    raise NotImplementedError("Fase 0")
