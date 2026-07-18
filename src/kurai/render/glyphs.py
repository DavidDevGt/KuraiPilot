"""Glifos bitmap 8×16 embebidos — la fuente de referencia del proyecto (docs/02 E8).

Los bitmaps viven en el repo (no en una fuente del sistema) porque son parte
del contrato de reproducibilidad G4: el mismo char_idx debe producir el mismo
píxel en cualquier máquina, hoy y en dos años. Cambiar un bitmap cambia los
golden files — requiere justificación (docs/06 §1).

La rampa `short` está ordenada por cobertura de tinta medida (píxeles
encendidos), no por intuición: tools/calibrate_ramp.py la verifica y
tests/test_render.py la hace cumplir (monotonía estricta).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.config import Ramp

GLYPH_W = 8
GLYPH_H = 16

# Cada glifo: 16 filas de 8 bits (MSB = columna izquierda). Diseño propio
# inspirado en fuentes bitmap clásicas 8×N, ajustado para cobertura monótona.
_BITMAPS: dict[str, list[int]] = {
    " ": [0x00] * 16,
    ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x18, 0x18, 0, 0, 0, 0],
    ":": [0, 0, 0, 0, 0, 0x18, 0x18, 0, 0, 0, 0x18, 0x18, 0, 0, 0, 0],
    "-": [0, 0, 0, 0, 0, 0, 0, 0x7E, 0x7E, 0, 0, 0, 0, 0, 0, 0],
    "=": [0, 0, 0, 0, 0, 0x7E, 0x7E, 0, 0, 0x7E, 0x7E, 0, 0, 0, 0, 0],
    "+": [0, 0, 0, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x7E, 0x18, 0x18, 0x18, 0x18, 0, 0, 0],
    "*": [0, 0, 0, 0, 0x18, 0xDB, 0x7E, 0x3C, 0x7E, 0xDB, 0x18, 0, 0, 0, 0, 0],
    "#": [0, 0, 0x24, 0x24, 0x24, 0xFF, 0x24, 0x24, 0x24, 0x24, 0xFF, 0x24, 0x24, 0x24, 0, 0],
    "%": [0, 0, 0, 0xC6, 0xC6, 0xCC, 0xCC, 0x18, 0x18, 0x30, 0x30, 0x66, 0x66, 0xC6, 0xC6, 0],
    "@": [0, 0, 0, 0x7C, 0x7C, 0xC6, 0xC6, 0xDE, 0xDE, 0xDE, 0xDE, 0xC0, 0x7C, 0x7C, 0, 0],
}

# Bloques Unicode: procedurales por densidad, patrón espacialmente estable.
_BLOCK_FILL = {" ": 0.0, "░": 0.25, "▒": 0.5, "▓": 0.75, "█": 1.0}

RAMPS: dict[Ramp, str] = {
    Ramp.SHORT: " .:-=+*#%@",
    Ramp.BLOCKS: " ░▒▓█",
    # Ramp.LONG llega con el preset alta-fidelidad (docs/07 Fase 2)
}


def _block_bitmap(fill: float) -> npt.NDArray[np.uint8]:
    if fill == 0.0:
        return np.zeros((GLYPH_H, GLYPH_W), dtype=np.uint8)
    if fill == 1.0:
        return np.full((GLYPH_H, GLYPH_W), 255, dtype=np.uint8)
    rr, cc = np.mgrid[0:GLYPH_H, 0:GLYPH_W]
    if fill == 0.5:
        mask = (rr + cc) % 2 == 0
    elif fill == 0.25:
        mask = (rr % 2 == 0) & ((cc + rr // 2) % 2 == 0)
    else:  # 0.75 = inverso del 0.25
        mask = ~((rr % 2 == 0) & ((cc + rr // 2) % 2 == 0))
    out = np.zeros((GLYPH_H, GLYPH_W), dtype=np.uint8)
    out[mask] = 255
    return out


def bitmap(char: str) -> npt.NDArray[np.uint8]:
    """Bitmap (GLYPH_H, GLYPH_W) uint8 {0, 255} del glifo."""
    rows = _BITMAPS.get(char)
    if rows is not None:
        bits = np.array(rows, dtype=np.uint8)
        return (np.unpackbits(bits[:, None], axis=1) * 255).astype(np.uint8)
    if char in _BLOCK_FILL:
        return _block_bitmap(_BLOCK_FILL[char])
    directional = DIRECTIONAL_GLYPHS.get(char)
    if directional is not None:  # E5 `edges`: trazos direccionales (definidos abajo)
        return directional.copy()
    raise KeyError(f"Glifo '{char}' no está en la fuente embebida")


def ink_coverage(char: str) -> int:
    """Píxeles encendidos del glifo — la métrica que ordena las rampas."""
    return int(np.count_nonzero(bitmap(char)))


def ramp_chars(ramp: Ramp) -> str:
    chars = RAMPS.get(ramp)
    if chars is None:
        raise NotImplementedError("Fase 2")  # Ramp.LONG, docs/07
    return chars


# ── Glifos direccionales (E5 nivel `edges`, docs/02 E5) ──────────────────────
# La rampa tonal `short` no incluye trazos direccionales; el refinamiento
# estructural de la Etapa 5 los necesita para reemplazar el carácter tonal por
# uno que siga la orientación del borde (estilo AsciiArtist). Se generan de
# forma procedural y determinista (mismo dtype (uint8), mismo layout (GLYPH_H,
# GLYPH_W) y misma escala de tinta {0, 255} que los tonales) para no depender de
# una fuente del sistema — mismo contrato de reproducibilidad G4. NO forman
# parte de RAMPS ni del hash-guard tonal: son un set aparte, indexado en el
# atlas a continuación de la rampa tonal (índice = tonal_levels + posición).
_DIRECTIONAL_ORDER = "/\\|-_()"


def _dir_canvas() -> npt.NDArray[np.uint8]:
    return np.zeros((GLYPH_H, GLYPH_W), dtype=np.uint8)


def _draw_diagonal(canvas: npt.NDArray[np.uint8], *, rising: bool) -> None:
    """Diagonal de 2 px. rising=True → '/' (abajo-izquierda a arriba-derecha)."""
    rows = np.arange(GLYPH_H)
    frac = rows / (GLYPH_H - 1)  # 0.0 arriba → 1.0 abajo
    span = np.float64(GLYPH_W - 1)
    cols = np.rint((1.0 - frac) * span if rising else frac * span).astype(np.intp)
    canvas[rows, cols] = 255
    canvas[rows, np.clip(cols - 1, 0, GLYPH_W - 1)] = 255


def _draw_vertical(canvas: npt.NDArray[np.uint8]) -> None:
    """'|' — barra vertical de 2 px centrada."""
    mid = GLYPH_W // 2
    canvas[:, mid - 1 : mid + 1] = 255


def _draw_horizontal(canvas: npt.NDArray[np.uint8], top_row: int) -> None:
    """'-' / '_' — barra horizontal de 2 px a la altura `top_row`."""
    canvas[top_row : top_row + 2, 1 : GLYPH_W - 1] = 255


def _draw_arc(canvas: npt.NDArray[np.uint8], *, bulge_left: bool) -> None:
    """'(' (panza a la izquierda) / ')' (panza a la derecha) — arco vertical."""
    rows = np.arange(GLYPH_H)
    bulge = np.sin(np.pi * rows / (GLYPH_H - 1))  # 0 en extremos, 1 al medio
    cols = np.rint(4.0 - 3.0 * bulge if bulge_left else 3.0 + 3.0 * bulge)
    cols = np.clip(cols.astype(np.intp), 0, GLYPH_W - 1)
    canvas[rows, cols] = 255


def _build_directional() -> dict[str, npt.NDArray[np.uint8]]:
    glyphs: dict[str, npt.NDArray[np.uint8]] = {}

    slash = _dir_canvas()
    _draw_diagonal(slash, rising=True)
    glyphs["/"] = slash

    backslash = _dir_canvas()
    _draw_diagonal(backslash, rising=False)
    glyphs["\\"] = backslash

    pipe = _dir_canvas()
    _draw_vertical(pipe)
    glyphs["|"] = pipe

    dash = _dir_canvas()
    _draw_horizontal(dash, GLYPH_H // 2 - 1)
    glyphs["-"] = dash

    underscore = _dir_canvas()
    _draw_horizontal(underscore, GLYPH_H - 2)
    glyphs["_"] = underscore

    lparen = _dir_canvas()
    _draw_arc(lparen, bulge_left=True)
    glyphs["("] = lparen

    rparen = _dir_canvas()
    _draw_arc(rparen, bulge_left=False)
    glyphs[")"] = rparen

    return glyphs


# char → bitmap (GLYPH_H, GLYPH_W) uint8 {0, 255}, en el orden canónico.
DIRECTIONAL_GLYPHS: dict[str, npt.NDArray[np.uint8]] = _build_directional()


def directional_glyph_string() -> str:
    """Orden canónico de los glifos direccionales de E5 `edges` (docs/02 E5).

    El índice de cada direccional en el atlas es `tonal_levels + posición` en
    esta cadena. Determinista y estable — mismo contrato de reproducibilidad G4
    que las rampas tonales.
    """
    return _DIRECTIONAL_ORDER
