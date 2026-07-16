"""E8 + glifos: calibración de rampa, identidad del atlas, composición."""

from __future__ import annotations

import numpy as np

from kurai.config import ColorMode, Ramp
from kurai.render.atlas import MONO_COLOR, build_atlas, compose
from kurai.render.glyphs import GLYPH_H, GLYPH_W, RAMPS, bitmap, ink_coverage, ramp_chars
from kurai.types import CharMatrix


def test_all_ramps_strictly_monotonic_by_ink_coverage() -> None:
    """La calibración de docs/02 E4: más índice de rampa = más tinta, siempre."""
    for ramp, chars in RAMPS.items():
        coverages = [ink_coverage(ch) for ch in chars]
        assert coverages == sorted(set(coverages)), f"rampa {ramp}: {coverages}"


def test_ramp_extremes() -> None:
    for chars in RAMPS.values():
        assert ink_coverage(chars[0]) == 0  # el nivel 0 es espacio (negro puro)
        assert ink_coverage(chars[-1]) > GLYPH_H * GLYPH_W * 0.25


def test_bitmaps_are_binary_and_right_shape() -> None:
    for chars in RAMPS.values():
        for ch in chars:
            bm = bitmap(ch)
            assert bm.shape == (GLYPH_H, GLYPH_W)
            assert set(np.unique(bm)) <= {0, 255}


def test_atlas_identity_roundtrip() -> None:
    """Componer una CharMatrix con todos los glifos reproduce el atlas exacto
    (docs/06 §2 E8): la composición es indexing puro, sin resampling."""
    chars = ramp_chars(Ramp.SHORT)
    atlas = build_atlas(chars)
    n = len(chars)
    cm = CharMatrix(
        char_idx=np.arange(n, dtype=np.uint8).reshape(1, n),
        fg=np.full((1, n, 3), 255, dtype=np.uint8),
    )
    frame = compose(cm, atlas, ColorMode.FG)
    for i in range(n):
        cell = frame[:, i * GLYPH_W : (i + 1) * GLYPH_W]
        assert np.array_equal(cell[:, :, 0], atlas[i]), f"glifo {i} ('{chars[i]}')"


def test_mono_uses_phosphor_color() -> None:
    atlas = build_atlas(ramp_chars(Ramp.SHORT))
    cm = CharMatrix(
        char_idx=np.full((2, 2), 9, dtype=np.uint8),  # '@', el más denso
        fg=np.full((2, 2, 3), 255, dtype=np.uint8),
    )
    frame = compose(cm, atlas, ColorMode.MONO)
    lit = frame[frame[:, :, 1] > 0]
    assert len(lit) > 0
    assert np.array_equal(np.unique(lit.reshape(-1, 3), axis=0)[-1], MONO_COLOR)


def test_fg_tints_with_cell_color() -> None:
    atlas = build_atlas(ramp_chars(Ramp.SHORT))
    red = np.zeros((1, 1, 3), dtype=np.uint8)
    red[0, 0] = (255, 0, 0)
    cm = CharMatrix(char_idx=np.full((1, 1), 9, dtype=np.uint8), fg=red)
    frame = compose(cm, atlas, ColorMode.FG)
    assert frame[:, :, 0].max() == 255  # canal rojo encendido
    assert frame[:, :, 1].max() == 0  # verde/azul apagados
    assert frame[:, :, 2].max() == 0
