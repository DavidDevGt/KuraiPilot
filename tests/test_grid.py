"""Etapa 2 — geometría de grilla. Property-based: la corrección de aspecto 1:2
debe cumplirse para cualquier resolución razonable, no solo los casos que se nos
ocurrieron (docs/02 E2)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from kurai.engine.grid import CELL_H, CELL_W, grid_shape


def test_reference_case_1080p() -> None:
    assert grid_shape(1920, 1080, 160) == (45, 160)


def test_vertical_video() -> None:
    rows, cols = grid_shape(1080, 1920, 90)
    assert cols == 90
    assert rows == 80  # 1920/1080 * 90 * 0.5


@given(
    w=st.integers(min_value=160, max_value=7680),
    h=st.integers(min_value=90, max_value=4320),
    cols=st.integers(min_value=20, max_value=600),
)
def test_grid_shape_properties(w: int, h: int, cols: int) -> None:
    rows, out_cols = grid_shape(w, h, cols)
    assert out_cols == cols
    assert rows >= 1
    # Aspecto: rows*CELL_H / cols*CELL_W ≈ h/w (el render no deforma), con
    # tolerancia de 1 fila por el redondeo. Bajo 1 fila manda el clamp a 1.
    ideal_rows = h / w * cols * (CELL_W / CELL_H)
    if ideal_rows < 1:
        assert rows == 1
    else:
        assert abs(rows - ideal_rows) <= 0.5 + 1e-9


@given(
    w=st.integers(min_value=160, max_value=3840),
    h=st.integers(min_value=90, max_value=2160),
)
def test_more_cols_never_fewer_rows(w: int, h: int) -> None:
    """Monotonía: subir la resolución de columnas nunca reduce filas."""
    prev_rows = 0
    for cols in (40, 80, 160, 320):
        rows, _ = grid_shape(w, h, cols)
        assert rows >= prev_rows
        prev_rows = rows
