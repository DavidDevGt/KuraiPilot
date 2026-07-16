"""Etapa 8 — Atlas de glifos y composición (docs/02 E8).

El atlas se pre-renderiza UNA vez al inicio: uint8[n_glyphs, 16, 8]. El frame
se compone con fancy indexing (frame = atlas[char_idx] + tintado por fg).
PROHIBIDO dibujar carácter por carácter con PIL/Cairo en el hot path (100-1000×
más lento; causa de muerte de proyectos relevados en INVESTIGATION.md).

La fuente de referencia es parte del contrato de reproducibilidad (G4):
candidatas Spleen / Cozette 8×16, se fija en Fase 0 y no se cambia sin ADR.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.config import ColorMode
from kurai.types import CharMatrix

GLYPH_W = 8
GLYPH_H = 16


def build_atlas(ramp_chars: str) -> npt.NDArray[np.uint8]:
    """→ (n_glyphs, GLYPH_H, GLYPH_W) en [0,255], un bitmap por glifo de la rampa."""
    raise NotImplementedError("Fase 0")


def compose(
    cm: CharMatrix, atlas: npt.NDArray[np.uint8], color: ColorMode
) -> npt.NDArray[np.uint8]:
    """CharMatrix → frame RGB (rows*GLYPH_H, cols*GLYPH_W, 3). Solo fancy indexing."""
    raise NotImplementedError("Fase 0")
