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


SALIENCY_MIN_LEVELS = 4  # docs/02 E3: zonas de baja densidad usan una rampa corta de 4 niveles


def quantize(
    luma_gamma: npt.NDArray[np.float32],
    levels: int,
    dither_offsets: npt.NDArray[np.float32] | None = None,
    density_map: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.uint8]:
    """→ char_idx (rows, cols): índice en la rampa, monótono en luma.

    density_map (saliencia, E3) modula la rampa EFECTIVA por celda: zonas
    salientes (density→1) usan la rampa completa; el fondo (density→0) colapsa
    a SALIENCY_MIN_LEVELS niveles — el presupuesto de detalle se concentra en el
    sujeto (docs/02 E3, docs/04 §2). El char_idx sigue indexando la MISMA rampa
    (el atlas no cambia): una celda de fondo usa menos caracteres distintos, no
    otros glifos. Con density ≡ 1.0 el resultado es bit a bit el de sin saliencia.
    """
    adjusted = luma_gamma if dither_offsets is None else luma_gamma + dither_offsets
    if density_map is None:
        idx = np.floor(adjusted * levels).astype(np.int32)
        return np.clip(idx, 0, levels - 1).astype(np.uint8)

    # Niveles efectivos por celda: lerp(MIN, levels) según densidad. Con
    # density=1 ⇒ eff=levels y todo el bloque colapsa a la cuantización normal.
    eff_min = min(SALIENCY_MIN_LEVELS, levels)
    eff = np.rint(eff_min + density_map * (levels - eff_min)).astype(np.int32)
    eff = np.clip(eff, eff_min, levels)
    coarse = np.clip(np.floor(adjusted * eff).astype(np.int32), 0, eff - 1)
    # Re-expandir el bin grueso al rango completo [0, levels-1] (misma rampa):
    # los extremos se preservan (0→0, eff-1→levels-1) y es monótono en luma.
    scale = np.float32(levels - 1) / np.maximum(eff - 1, 1)
    idx = np.rint(coarse.astype(np.float32) * scale).astype(np.int32)
    return np.clip(idx, 0, levels - 1).astype(np.uint8)
