"""Etapa 8 — Atlas de glifos y composición (docs/02 E8).

El atlas se pre-renderiza UNA vez desde los bitmaps embebidos (glyphs.py);
el frame se compone con fancy indexing — PROHIBIDO dibujar carácter por
carácter en el hot path (100-1000× más lento, INVESTIGATION.md).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.config import ColorMode
from kurai.render.glyphs import GLYPH_H, GLYPH_W, bitmap
from kurai.types import CharMatrix

# Verde fósforo del preset retro (mono)
MONO_COLOR = np.array([102, 255, 102], dtype=np.uint8)


def build_atlas(ramp: str) -> npt.NDArray[np.uint8]:
    """→ (n_glyphs, GLYPH_H, GLYPH_W) uint8 {0,255}, un bitmap por glifo."""
    return np.stack([bitmap(ch) for ch in ramp])


def compose(
    cm: CharMatrix, atlas: npt.NDArray[np.uint8], color: ColorMode
) -> npt.NDArray[np.uint8]:
    """CharMatrix → frame RGB (rows*GLYPH_H, cols*GLYPH_W, 3). Solo indexing."""
    rows, cols = cm.shape

    if color is ColorMode.MONO:
        # El tintado corre sobre el ATLAS (~4k elementos), no sobre el frame
        # (millones): el frame es un gather puro.
        tinted = (
            atlas[:, :, :, None].astype(np.uint16) * MONO_COLOR.astype(np.uint16) // 255
        ).astype(np.uint8)  # (n_glyphs, GH, GW, 3)
        gathered = tinted[cm.char_idx]  # (rows, cols, GH, GW, 3)
        return np.ascontiguousarray(
            gathered.transpose(0, 2, 1, 3, 4).reshape(rows * GLYPH_H, cols * GLYPH_W, 3)
        )

    if color is ColorMode.FG:
        # (rows, cols, GH, GW) → (rows, GH, cols, GW) → (rows*GH, cols*GW)
        mask = atlas[cm.char_idx].transpose(0, 2, 1, 3).reshape(rows * GLYPH_H, cols * GLYPH_W)
        tint = np.repeat(np.repeat(cm.fg, GLYPH_H, axis=0), GLYPH_W, axis=1)
        frame = mask[:, :, None].astype(np.uint16) * tint.astype(np.uint16) // 255
        return frame.astype(np.uint8)

    # fg+bg (docs/02 E8): la tinta del glifo lleva fg y el resto de la celda bg
    # — dos colores por celda. Mismo gather que fg más el término de fondo.
    if cm.bg is None:
        raise ValueError("ColorMode.FG_BG requiere CharMatrix.bg (pipeline fg+bg)")
    mask = atlas[cm.char_idx].transpose(0, 2, 1, 3).reshape(rows * GLYPH_H, cols * GLYPH_W)
    fg_tint = np.repeat(np.repeat(cm.fg, GLYPH_H, axis=0), GLYPH_W, axis=1).astype(np.uint16)
    bg_tint = np.repeat(np.repeat(cm.bg, GLYPH_H, axis=0), GLYPH_W, axis=1).astype(np.uint16)
    m = mask[:, :, None].astype(np.uint16)
    frame = (m * fg_tint + (255 - m) * bg_tint) // 255
    return frame.astype(np.uint8)
