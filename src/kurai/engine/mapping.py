"""Etapa 4 — Mapeo luminancia → carácter (docs/02 E4). Determinista.

Se implementa fusionada con el dithering (E6): los offsets de Bayer se suman
a la luma post-gamma antes de cuantizar. Las rampas calibradas viven en
render/glyphs.py (cobertura de tinta medida, no intuición).

Flujo: luma → apply_gamma → (+offsets Bayer) → quantize. La histéresis (E7)
compara en el espacio post-gamma, el mismo en el que se cuantiza.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def apply_gamma(luma_grid: npt.NDArray[np.float32], gamma: float) -> npt.NDArray[np.float32]:
    """Luma perceptual (post-gamma): la base común de E4/E6/E7."""
    return np.power(luma_grid, gamma, dtype=np.float32)


def quantize(
    luma_gamma: npt.NDArray[np.float32],
    levels: int,
    dither_offsets: npt.NDArray[np.float32] | None = None,
    density_map: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.uint8]:
    """→ char_idx (rows, cols): índice en la rampa, monótono en luma.

    density_map modula la rampa efectiva por celda — llega con la saliencia.
    """
    if density_map is not None:
        raise NotImplementedError("Fase 1")
    adjusted = luma_gamma if dither_offsets is None else luma_gamma + dither_offsets
    idx = np.floor(adjusted * levels).astype(np.int32)
    return np.clip(idx, 0, levels - 1).astype(np.uint8)
