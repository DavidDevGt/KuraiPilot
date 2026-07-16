"""Auditoría del comando ffmpeg del Encoder: los flags que la investigación
de mejores prácticas marcó como obligatorios (ver docs/02 E9)."""

from __future__ import annotations

from pathlib import Path

from kurai.engine.encode import Encoder
from kurai.types import VideoMeta

META = VideoMeta(
    width=320,
    height=176,
    fps=29.97002997002997,
    n_frames=60,
    duration_s=2.0,
    rotation=0,
    has_audio=False,
    codec="h264",
    fps_rational="30000/1001",
)


def _cmd(use_nvenc: bool) -> list[str]:
    return Encoder(Path("/tmp/x.mp4"), META, None, use_nvenc=use_nvenc)._cmd


def test_nvenc_true_constant_quality_needs_bv0() -> None:
    """Sin -b:v 0, nvenc capa el bitrate al default y -cq es decorativo."""
    cmd = _cmd(use_nvenc=True)
    assert "-cq" in cmd
    bv = cmd.index("-b:v")
    assert cmd[bv + 1] == "0"


def test_bt709_matrix_and_tags() -> None:
    """swscale usa BT.601 por defecto; sin out_color_matrix + tags el color
    queda corrido en players HD."""
    for use_nvenc in (False, True):
        cmd = _cmd(use_nvenc)
        vf = cmd[cmd.index("-vf") + 1]
        assert "out_color_matrix=bt709" in vf
        assert "full_chroma_int" in vf and "accurate_rnd" in vf
        assert "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709" in vf


def test_rational_fps_not_float() -> None:
    """-r recibe el racional exacto (30000/1001), no el float con drift."""
    cmd = _cmd(use_nvenc=False)
    assert cmd[cmd.index("-r") + 1] == "30000/1001"


def test_fps_expr_fallback_without_rational() -> None:
    meta = VideoMeta(
        width=8,
        height=16,
        fps=30.0,
        n_frames=1,
        duration_s=1.0,
        rotation=0,
        has_audio=False,
        codec="h264",
    )
    assert meta.fps_expr == "30.0"
