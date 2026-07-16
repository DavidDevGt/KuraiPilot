"""CLI: arranque, versión, doctor, manejo de errores de usuario."""

from __future__ import annotations

from pathlib import Path

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


def test_pending_commands_exit_2_not_crash(tmp_path: Path) -> None:
    """Los subcomandos de fases pendientes salen con código 2, nunca traceback.
    (bench ya no está acá: se implementó en Fase 0.)"""
    f = tmp_path / "x.mp4"
    f.write_bytes(b"\x00")
    for args in (["convert", str(f)], ["preview", str(f)], ["live"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, f"{args}: {result.output}"
