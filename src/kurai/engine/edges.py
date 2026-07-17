r"""Etapa 5 — Refinamiento estructural determinista, nivel `edges` (docs/02 E5).

Sobel 3×3 por celda sobre la grilla de luma: donde la magnitud del gradiente
supera un umbral, la celda es "estructural" y su carácter tonal se reemplaza por
el glifo direccional que sigue la orientación del borde (`/ \ | - _`, estilo
AsciiArtist). Las celdas planas conservan su carácter tonal.

Vectorización estricta (ADR-006): el Sobel se computa por slicing sobre el array
completo (sin scipy, sin bucles por celda) y la selección glifo-tonal ↔ glifo-
direccional es una máscara booleana con np.where. Determinista (G4): mismo
luma_grid + misma config ⇒ char_idx bit a bit idéntico.

El orden canónico de los direccionales y sus índices los define
render/glyphs.py (`directional_glyph_string`): el índice del direccional en la
posición k es `tonal_levels + k`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.render.glyphs import directional_glyph_string

# Mapeo orientación del gradiente → glifo, por octante de atan2(gy, gx).
#
# Convención: gx = derivada horizontal (columna, derecha positiva), gy =
# derivada vertical (fila, abajo positiva). El BORDE es perpendicular al
# gradiente, así que un gradiente horizontal (octantes 0 y 4) marca un borde
# VERTICAL '|', y uno vertical (octantes 2 y 6) un borde HORIZONTAL. Los
# octantes antipodales comparten glifo (misma recta) salvo el horizontal, que
# distingue polaridad para aprovechar '_': gradiente hacia abajo (brillo debajo)
# → '-' a media altura; hacia arriba → '_' abajo. Las diagonales: gradiente a
# +45° (abajo-derecha) es perpendicular a la anti-diagonal ⇒ '/'; a +135°
# (abajo-izquierda) ⇒ '\'.
#
# Los arcos '(' ')' existen en el set direccional (los usa el refinador `cnn`,
# docs/04 §3, que elige del glifo completo) pero el Sobel por celda sólo aporta
# una orientación recta y no los emite: inventar curvatura desde un detector de
# rectas rompería el determinismo del significado.
#
#      octante:   0    1    2    3     4    5    6    7
#      grados:    0   +45  +90  +135  180  -135 -90  -45
_OCTANT_CHARS = ["|", "/", "-", "\\", "|", "/", "_", "\\"]
_DIRECTIONAL = directional_glyph_string()
_OCTANT_OFFSET: npt.NDArray[np.uint8] = np.array(
    [_DIRECTIONAL.index(ch) for ch in _OCTANT_CHARS], dtype=np.uint8
)

# Umbral por defecto sobre la magnitud Sobel de la luma en [0, 1]. Un borde
# nítido de contraste pleno (paso 0→1) da magnitud ≈ 4 (pesos 1+2+1); 0.5
# corresponde a un salto local de luma ≈ 0.12 (0.5 / 4) — por debajo es textura
# tonal, por encima es estructura. Marca ~5-20% de celdas en contenido típico
# (docs/04 §3): sólo los contornos, no el degradé tonal que domina la grilla.
DEFAULT_THRESHOLD = 0.5


def refine_edges(
    luma_grid: npt.NDArray[np.float32],
    char_idx: npt.NDArray[np.uint8],
    tonal_levels: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> npt.NDArray[np.uint8]:
    """char_idx tonal → refinado: direccional en celdas estructurales (docs/02 E5).

    En las celdas cuya magnitud de gradiente Sobel supera ``threshold`` el índice
    tonal se reemplaza por ``tonal_levels + offset`` del glifo direccional que
    calza con la orientación del borde; el resto conserva su índice tonal.
    """
    rows = luma_grid.shape[0]
    cols = luma_grid.shape[1]
    padded = np.pad(luma_grid, 1, mode="edge").astype(np.float32)

    top_left = padded[0:rows, 0:cols]
    top_center = padded[0:rows, 1 : cols + 1]
    top_right = padded[0:rows, 2 : cols + 2]
    mid_left = padded[1 : rows + 1, 0:cols]
    mid_right = padded[1 : rows + 1, 2 : cols + 2]
    bot_left = padded[2 : rows + 2, 0:cols]
    bot_center = padded[2 : rows + 2, 1 : cols + 1]
    bot_right = padded[2 : rows + 2, 2 : cols + 2]

    gx = (top_right + 2.0 * mid_right + bot_right) - (top_left + 2.0 * mid_left + bot_left)
    gy = (bot_left + 2.0 * bot_center + bot_right) - (top_left + 2.0 * top_center + top_right)

    magnitude = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)  # (-π, π]
    octant = np.rint(theta / (np.pi / 4.0)).astype(np.intp) % 8

    directional = (_OCTANT_OFFSET[octant].astype(np.uint16) + np.uint16(tonal_levels)).astype(
        np.uint8
    )
    mask = magnitude > np.float32(threshold)
    result: npt.NDArray[np.uint8] = np.where(mask, directional, char_idx).astype(np.uint8)
    return result
