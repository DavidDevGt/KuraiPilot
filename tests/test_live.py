"""Modo live: proyección ANSI de la CharMatrix y pacing con drop de frames."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from kurai.config import ColorMode, JobConfig, load_preset
from kurai.render.ansi import ENTER_ALT_SCREEN, EXIT_ALT_SCREEN, HOME, charmatrix_to_ansi
from kurai.types import CharMatrix


def _cm() -> CharMatrix:
    return CharMatrix(
        char_idx=np.array([[0, 9], [5, 0]], dtype=np.uint8),
        fg=np.array([[[255, 0, 0], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
    )


def test_ansi_mono_maps_ramp_chars() -> None:
    out = charmatrix_to_ansi(_cm(), " .:-=+*#%@", ColorMode.MONO)
    assert out.startswith(HOME)
    assert " @\n+ " in out  # idx 0→' ', 9→'@', 5→'+'
    assert out.count("\x1b[38;2;") == 1  # mono: UN solo código de color


def test_ansi_fg_run_length() -> None:
    """Celdas contiguas del mismo color comparten UN código SGR; el cambio
    de color emite uno nuevo. La fila 1 tiene 2 colores; la 0, uno."""
    out = charmatrix_to_ansi(_cm(), " .:-=+*#%@", ColorMode.FG)
    assert out.count("\x1b[38;2;255;0;0m") == 1  # fila roja: un solo run
    assert "\x1b[38;2;0;255;0m" in out and "\x1b[38;2;0;0;255m" in out


def test_ansi_charmatrix_is_the_canonical_artifact() -> None:
    """El texto ANSI contiene EXACTAMENTE los caracteres de la rampa indexados
    por la CharMatrix — misma proyección canónica que el atlas (docs/01 §5)."""
    ramp = " ░▒▓█"
    cm = CharMatrix(
        char_idx=np.array([[4, 3, 2, 1, 0]], dtype=np.uint8),
        fg=np.zeros((1, 5, 3), dtype=np.uint8),
    )
    out = charmatrix_to_ansi(cm, ramp, ColorMode.MONO)
    assert "█▓▒░ " in out


@pytest.mark.ffmpeg
def test_live_writes_frames_and_restores_terminal(
    clip_testsrc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kurai.engine import live as live_mod

    monkeypatch.setattr(
        live_mod.shutil, "get_terminal_size", lambda: __import__("os").terminal_size((80, 24))
    )
    buf = io.StringIO()
    cfg = JobConfig(preset=load_preset("retro"))
    shown, skipped = live_mod.run_live(clip_testsrc, cfg, out=buf, max_frames=5)
    output = buf.getvalue()
    assert output.startswith(ENTER_ALT_SCREEN)
    assert output.endswith(EXIT_ALT_SCREEN)  # SIEMPRE restaura, pase lo que pase
    assert shown + skipped == 5
    assert shown >= 1


class _ExplodingSink:
    """Falla EXACTAMENTE una vez a mitad de la reproducción (terminal roto,
    pipe cerrado…) y registra todo lo demás."""

    def __init__(self, explode_at: int) -> None:
        self.chunks: list[str] = []
        self._writes = 0
        self._explode_at = explode_at
        self._exploded = False

    def write(self, s: str) -> None:
        self._writes += 1
        if self._writes == self._explode_at and not self._exploded:
            self._exploded = True
            raise RuntimeError("terminal roto a mitad del stream")
        self.chunks.append(s)

    def flush(self) -> None:
        pass


@pytest.mark.ffmpeg
def test_live_restores_terminal_on_error(
    clip_testsrc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si algo explota a mitad de la reproducción, el terminal NO queda en el
    alt buffer (el finally SIEMPRE emite el restore)."""
    from kurai.engine import live as live_mod

    monkeypatch.setattr(
        live_mod.shutil, "get_terminal_size", lambda: __import__("os").terminal_size((80, 24))
    )
    sink = _ExplodingSink(explode_at=3)  # tras ENTER + primer frame
    with pytest.raises(RuntimeError, match="terminal roto"):
        live_mod.run_live(clip_testsrc, JobConfig(preset=load_preset("retro")), out=sink)
    assert sink.chunks[0] == ENTER_ALT_SCREEN
    assert sink.chunks[-1] == EXIT_ALT_SCREEN


def test_terminal_grid_fits_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from kurai.engine import live as live_mod

    monkeypatch.setattr(live_mod.shutil, "get_terminal_size", lambda: os.terminal_size((100, 30)))
    rows, cols = live_mod.terminal_grid(1920, 1080, None)
    assert cols <= 100
    assert rows <= 29  # deja una línea libre
