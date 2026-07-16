"""Smoke tests de la carretera: el CLI arranca, los presets validan, la grilla
respeta el aspecto, la probe corre. Nada de pipeline todavía (Fase 0)."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from kurai.cli import app
from kurai.config import ColorMode, DitherMode, RefineMode, load_preset
from kurai.engine.grid import grid_shape
from kurai.probe import probe
from kurai.types import CharMatrix

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("convert", "preview", "live", "bench", "doctor"):
        assert cmd in result.output


def test_presets_load_and_match_spec() -> None:
    """Los tres presets validan contra Pydantic y coinciden con docs/02 §10."""
    retro = load_preset("retro")
    assert not retro.saliency and retro.color is ColorMode.MONO

    detallado = load_preset("detallado")
    assert detallado.saliency and detallado.refine is RefineMode.EDGES

    alta = load_preset("alta-fidelidad")
    assert alta.dither is DitherMode.FLOYD_STEINBERG
    assert alta.refine is RefineMode.EDGES_CNN


def test_unknown_preset_lists_available() -> None:
    with pytest.raises(FileNotFoundError, match="retro"):
        load_preset("inexistente")


def test_grid_shape_corrects_glyph_aspect() -> None:
    """Celdas 1:2 — un video 16:9 a 160 cols da ~45 filas, no 90 (docs/02 E2)."""
    rows, cols = grid_shape(1920, 1080, 160)
    assert cols == 160
    assert rows == 45


def test_charmatrix_exact_equality() -> None:
    a = CharMatrix(
        char_idx=np.zeros((4, 8), dtype=np.uint8),
        fg=np.zeros((4, 8, 3), dtype=np.uint8),
    )
    b = CharMatrix(
        char_idx=np.zeros((4, 8), dtype=np.uint8),
        fg=np.zeros((4, 8, 3), dtype=np.uint8),
    )
    assert a.equals(b)
    b.char_idx[0, 0] = 1
    assert not a.equals(b)


def test_probe_runs_and_finds_ffmpeg() -> None:
    r = probe()
    assert r.can_convert, f"ffmpeg debería existir en la máquina de referencia: {r.errors}"


def test_convert_reports_pending_phase(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "x.mp4"
    f.write_bytes(b"")
    result = runner.invoke(app, ["convert", str(f)])
    assert result.exit_code == 2  # fase pendiente, no crash
