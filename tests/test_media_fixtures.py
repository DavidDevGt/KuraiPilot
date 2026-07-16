"""Clips sintéticos generados con ffmpeg lavfi: el material de prueba de E1/E9.

Hoy validan que la generación funciona (la carretera); cuando Fase 0 implemente
decode/encode, estos mismos clips alimentan los tests de contrato de E1
(metadatos, VFR→CFR, audio) y E9 (n_frames, hash de audio) — docs/06 §2.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.ffmpeg


def _ffprobe(path: Path) -> dict:  # type: ignore[type-arg]
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def test_testsrc_clip_has_video_and_audio(clip_testsrc: Path) -> None:
    info = _ffprobe(clip_testsrc)
    kinds = {s["codec_type"] for s in info["streams"]}
    assert kinds == {"video", "audio"}
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (320, 180)
    assert video["r_frame_rate"] == "30/1"


def test_silent_clip_has_no_audio_stream(clip_silent: Path) -> None:
    info = _ffprobe(clip_silent)
    kinds = [s["codec_type"] for s in info["streams"]]
    assert kinds == ["video"]
