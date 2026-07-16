"""CharMatrix: igualdad exacta — la base del golden-file testing (docs/06 §1)."""

from __future__ import annotations

import numpy as np

from kurai.types import CharMatrix


def _cm(bg: bool = False) -> CharMatrix:
    return CharMatrix(
        char_idx=np.arange(32, dtype=np.uint8).reshape(4, 8),
        fg=np.full((4, 8, 3), 128, dtype=np.uint8),
        bg=np.zeros((4, 8, 3), dtype=np.uint8) if bg else None,
    )


def test_equal_matrices() -> None:
    assert _cm().equals(_cm())
    assert _cm(bg=True).equals(_cm(bg=True))


def test_single_cell_difference_detected() -> None:
    a, b = _cm(), _cm()
    b.char_idx[3, 7] += 1
    assert not a.equals(b)


def test_single_color_channel_difference_detected() -> None:
    a, b = _cm(bg=True), _cm(bg=True)
    assert b.bg is not None
    b.bg[0, 0, 2] = 1
    assert not a.equals(b)


def test_bg_presence_mismatch_is_inequality() -> None:
    assert not _cm().equals(_cm(bg=True))
    assert not _cm(bg=True).equals(_cm())


def test_shape_property() -> None:
    assert _cm().shape == (4, 8)
