"""CLI: arranque, versión, doctor, manejo de errores de usuario."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from kurai import __version__
from kurai.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("convert", "preview", "live", "bench", "doctor"):
        assert cmd in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_doctor_exits_zero_with_ffmpeg_present() -> None:
    """En cualquier máquina de desarrollo/CI ffmpeg existe → doctor OK."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffmpeg" in result.output


def test_convert_missing_file_is_usage_error() -> None:
    result = runner.invoke(app, ["convert", "no-existe.mp4"])
    assert result.exit_code != 0


def test_convert_unknown_preset_fails_listing_available(tmp_path: Path) -> None:
    f = tmp_path / "x.mp4"
    f.write_bytes(b"\x00")
    result = runner.invoke(app, ["convert", str(f), "--preset", "inexistente"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)
    assert "retro" in str(result.exception)  # el error lista los disponibles


def test_live_garbage_input_is_clean_error(tmp_path: Path) -> None:
    """live con un input que no es video: error accionable, exit 1, terminal
    intacto (no llega a entrar al alt screen). Desde Fase 0.5 no quedan
    subcomandos pendientes: convert/bench/preview/live están implementados."""
    f = tmp_path / "x.mp4"
    f.write_bytes(b"\x00")
    result = runner.invoke(app, ["live", str(f)])
    assert result.exit_code == 1
    assert "\x1b[?1049h" not in result.output  # jamás entró al alt buffer


@pytest.mark.ffmpeg
def test_convert_cli_happy_path(clip_testsrc: Path, tmp_path: Path) -> None:
    """El camino feliz por la superficie del CLI (progreso incluido), no solo
    por run_job directo."""
    out = tmp_path / "cli_out.mp4"
    result = runner.invoke(app, ["convert", str(clip_testsrc), "--cols", "40", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "tiempo real" in result.output


def test_convert_garbage_input_is_clean_error(tmp_path: Path) -> None:
    """Un input que no es video sale con error accionable, no traceback."""
    f = tmp_path / "garbage.mp4"
    f.write_bytes(b"\x00")
    result = runner.invoke(app, ["convert", str(f)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
