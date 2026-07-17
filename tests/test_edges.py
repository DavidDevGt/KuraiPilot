"""E5 nivel `edges`: glifos direccionales + refinamiento estructural (docs/02 E5).

Verifica el contrato de forma/tinta de los bitmaps direccionales, el orden
canónico determinista y el mapeo orientación→glifo del Sobel vectorizado.
"""

from __future__ import annotations

import numpy as np

from kurai.engine.edges import refine_edges
from kurai.render.glyphs import (
    DIRECTIONAL_GLYPHS,
    GLYPH_H,
    GLYPH_W,
    directional_glyph_string,
)

TONAL_LEVELS = 10  # rampa `short` (docs/02 §10)


def _dir_index(char: str) -> int:
    """Índice de atlas del glifo direccional `char` (docs/02 E5)."""
    return TONAL_LEVELS + directional_glyph_string().index(char)


def test_directional_bitmaps_shape_and_ink() -> None:
    """Cada direccional es (GLYPH_H, GLYPH_W) uint8 {0,255} con tinta > 0."""
    for char, bm in DIRECTIONAL_GLYPHS.items():
        assert bm.shape == (GLYPH_H, GLYPH_W), char
        assert bm.dtype == np.uint8, char
        assert set(np.unique(bm)) <= {0, 255}, char
        assert np.count_nonzero(bm) > 0, char  # no está vacío


def test_directional_glyph_string_is_deterministic_and_complete() -> None:
    assert directional_glyph_string() == directional_glyph_string()
    assert directional_glyph_string() == "/\\|-_()"
    for char in directional_glyph_string():
        assert char in DIRECTIONAL_GLYPHS
    assert len(directional_glyph_string()) == len(DIRECTIONAL_GLYPHS)


def test_flat_grid_leaves_char_idx_untouched() -> None:
    """Grilla plana: ningún gradiente supera el umbral ⇒ char_idx idéntico."""
    luma = np.full((8, 12), 0.5, dtype=np.float32)
    char_idx = (np.arange(8 * 12, dtype=np.uint8) % TONAL_LEVELS).reshape(8, 12)
    out = refine_edges(luma, char_idx, TONAL_LEVELS)
    assert np.array_equal(out, char_idx)


def test_vertical_edge_marks_vertical_glyph() -> None:
    """Borde vertical (mitad negra | mitad blanca): las columnas del borde se
    marcan con '|' (≥ tonal_levels); las planas quedan intactas."""
    luma = np.zeros((6, 8), dtype=np.float32)
    luma[:, 4:] = 1.0  # cols 0-3 negro, 4-7 blanco → borde entre col 3 y 4
    char_idx = np.zeros((6, 8), dtype=np.uint8)
    out = refine_edges(luma, char_idx, TONAL_LEVELS)

    pipe = _dir_index("|")
    assert np.all(out[:, 3] == pipe)
    assert np.all(out[:, 4] == pipe)
    # Las columnas planas conservan el tonal de entrada.
    for col in (0, 1, 2, 5, 6, 7):
        assert np.all(out[:, col] == 0), col


def test_horizontal_edge_marks_horizontal_glyph() -> None:
    """Borde horizontal (negro arriba, blanco abajo) → '-' a media altura."""
    luma = np.zeros((8, 6), dtype=np.float32)
    luma[4:, :] = 1.0  # filas 0-3 negro, 4-7 blanco → borde entre fila 3 y 4
    char_idx = np.zeros((8, 6), dtype=np.uint8)
    out = refine_edges(luma, char_idx, TONAL_LEVELS)

    dash = _dir_index("-")
    assert np.all(out[3, :] == dash)
    assert np.all(out[4, :] == dash)
    for row in (0, 1, 2, 5, 6, 7):
        assert np.all(out[row, :] == 0), row


def test_diagonal_edges_map_to_matching_glyph() -> None:
    """Un degradé diagonal recto marca con la diagonal que le corresponde.

    Gradiente hacia abajo-derecha (luma crece con fila y columna) ⇒ borde en la
    anti-diagonal ⇒ '/'. Hacia abajo-izquierda ⇒ '\\'.
    """
    n = 8
    rr, cc = np.mgrid[0:n, 0:n].astype(np.float32)
    denom = np.float32(2 * (n - 1))
    char_idx = np.zeros((n, n), dtype=np.uint8)

    slash_luma = ((rr + cc) / denom).astype(np.float32)
    slash = refine_edges(slash_luma, char_idx, TONAL_LEVELS, threshold=0.1)
    assert slash[4, 4] == _dir_index("/")

    backslash_luma = ((rr + (n - 1 - cc)) / denom).astype(np.float32)
    backslash = refine_edges(backslash_luma, char_idx, TONAL_LEVELS, threshold=0.1)
    assert backslash[4, 4] == _dir_index("\\")


def test_directional_indices_start_at_tonal_levels() -> None:
    """Todo carácter reemplazado por un borde vive en [tonal_levels, ...)."""
    luma = np.zeros((6, 8), dtype=np.float32)
    luma[:, 4:] = 1.0
    char_idx = np.zeros((6, 8), dtype=np.uint8)
    out = refine_edges(luma, char_idx, TONAL_LEVELS)
    marked = out[out != char_idx]
    assert marked.size > 0
    assert np.all(marked >= TONAL_LEVELS)


def test_refine_edges_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    luma = rng.random((16, 24), dtype=np.float32)
    char_idx = (np.arange(16 * 24, dtype=np.uint8) % TONAL_LEVELS).reshape(16, 24)
    a = refine_edges(luma, char_idx, TONAL_LEVELS)
    b = refine_edges(luma, char_idx, TONAL_LEVELS)
    assert np.array_equal(a, b)


def test_result_is_uint8_ndarray_of_grid_shape() -> None:
    luma = np.zeros((5, 7), dtype=np.float32)
    char_idx = np.zeros((5, 7), dtype=np.uint8)
    out = refine_edges(luma, char_idx, TONAL_LEVELS)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (5, 7)
