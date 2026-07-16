"""Etapa 9 — contratos de encode/mux: roundtrip passthrough completo
(docs/06 §2 E9): n_frames_out == n_frames_in, duración ±1 frame, audio
bit-idéntico (hash del stream)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from kurai.engine.decode import extract_audio, iter_frames, probe_video
from kurai.engine.encode import EncodeError, Encoder

pytestmark = pytest.mark.ffmpeg


def _audio_stream_md5(path: Path) -> str:
    """Hash del stream de audio SIN recodificar: detecta cualquier alteración."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "hash",
            "-hash",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture()
def roundtrip_output(clip_testsrc: Path, tmp_path: Path) -> tuple[Path, int]:
    """Passthrough completo: decode → encode con audio muxeado."""
    meta = probe_video(clip_testsrc)
    audio = extract_audio(clip_testsrc, tmp_path / "audio")
    out = tmp_path / "out.mp4"
    with Encoder(out, meta, audio, use_nvenc=False) as enc:
        for frame in iter_frames(clip_testsrc, meta, meta.width, meta.height):
            enc.write(frame)
    return out, enc.frames_written


def test_roundtrip_frame_count(clip_testsrc: Path, roundtrip_output: tuple[Path, int]) -> None:
    out, written = roundtrip_output
    in_meta, out_meta = probe_video(clip_testsrc), probe_video(out)
    assert written == in_meta.n_frames
    assert out_meta.n_frames == in_meta.n_frames


def test_roundtrip_duration(clip_testsrc: Path, roundtrip_output: tuple[Path, int]) -> None:
    out, _ = roundtrip_output
    in_meta, out_meta = probe_video(clip_testsrc), probe_video(out)
    assert abs(out_meta.duration_s - in_meta.duration_s) <= 1.0 / in_meta.fps + 1e-6


def test_roundtrip_audio_bit_identical(
    clip_testsrc: Path, roundtrip_output: tuple[Path, int]
) -> None:
    """La garantía central de docs/02 E9: -c:a copy no altera un solo byte."""
    out, _ = roundtrip_output
    assert _audio_stream_md5(out) == _audio_stream_md5(clip_testsrc)


def test_no_partial_output_on_failure(clip_testsrc: Path, tmp_path: Path) -> None:
    """Si el job aborta a mitad, no queda mp4 parcial silencioso (docs/02 §11)."""
    meta = probe_video(clip_testsrc)
    out = tmp_path / "partial.mp4"
    with (
        pytest.raises(RuntimeError, match="abortado aguas arriba"),
        Encoder(out, meta, None, use_nvenc=False) as enc,
    ):
        enc.write(np.zeros((meta.height, meta.width, 3), dtype=np.uint8))
        raise RuntimeError("abortado aguas arriba")
    assert not out.exists()


def test_wrong_frame_shape_rejected(clip_testsrc: Path, tmp_path: Path) -> None:
    meta = probe_video(clip_testsrc)
    with (
        pytest.raises(EncodeError, match="no coincide"),
        Encoder(tmp_path / "x.mp4", meta, None) as enc,
    ):
        enc.write(np.zeros((10, 10, 3), dtype=np.uint8))


def test_cq_bound_enforced(clip_testsrc: Path, tmp_path: Path) -> None:
    meta = probe_video(clip_testsrc)
    with pytest.raises(ValueError, match="23"):
        Encoder(tmp_path / "x.mp4", meta, None, cq=28)
