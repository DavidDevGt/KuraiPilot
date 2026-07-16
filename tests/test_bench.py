"""Bench passthrough: mecánica de medición y regresión (no los números —
esos solo valen en la máquina de referencia, docs/05)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kurai.bench import BenchResult, check_regression, run_passthrough


def _result(speed: float, encoder: str = "libx264") -> BenchResult:
    return BenchResult(
        mode="passthrough",
        clip="ref.mp4",
        encoder=encoder,
        video_seconds=10.0,
        wall_seconds=10.0 / speed,
        speed_factor=speed,
        frames=300,
        commit="abc1234",
        timestamp="2026-07-15T00:00:00+00:00",
    )


def test_within_tolerance_passes() -> None:
    assert check_regression(_result(3.8), baseline=_result(4.0)) is None


def test_regression_beyond_10pct_fails() -> None:
    failure = check_regression(_result(3.5), baseline=_result(4.0))
    assert failure is not None and "Regresión" in failure


def test_different_encoder_not_comparable() -> None:
    failure = check_regression(_result(4.0, "libx264"), baseline=_result(20.0, "h264_nvenc"))
    assert failure is not None and "no son comparables" in failure


def test_improvement_passes() -> None:
    assert check_regression(_result(5.0), baseline=_result(4.0)) is None


@pytest.mark.ffmpeg
def test_passthrough_on_small_clip(clip_testsrc: Path, tmp_path: Path) -> None:
    """El passthrough corre de punta a punta sobre el clip chico de la suite
    (el clip 1080p de referencia es solo para la máquina de referencia)."""
    result = run_passthrough(clip_testsrc, tmp_path, use_nvenc=False)
    assert result.frames == 60  # 2 s × 30 fps
    assert result.speed_factor > 0
    assert result.encoder == "libx264"
    assert (tmp_path / "passthrough_out.mp4").exists()


@pytest.mark.ffmpeg
def test_retro_mode_on_small_clip(clip_testsrc: Path, tmp_path: Path) -> None:
    """El modo retro del bench ejercita el pipeline completo."""
    from kurai.bench import run_retro

    result = run_retro(clip_testsrc, tmp_path, use_nvenc=False)
    assert result.mode == "retro"
    assert result.speed_factor > 0
    assert (tmp_path / "retro_out.mp4").exists()
