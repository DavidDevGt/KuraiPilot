"""Etapa 1 — contratos de decode/demux sobre clips sintéticos (docs/06 §2 E1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kurai.engine.decode import DecodeError, extract_audio, iter_frames, probe_video

pytestmark = pytest.mark.ffmpeg


def test_probe_metadata(clip_testsrc: Path) -> None:
    meta = probe_video(clip_testsrc)
    assert (meta.width, meta.height) == (320, 180)
    assert meta.fps == 30.0
    assert meta.has_audio
    assert meta.codec == "h264"
    assert meta.rotation == 0
    assert abs(meta.duration_s - 2.0) < 0.1
    assert meta.n_frames == round(meta.duration_s * 30)


def test_probe_nonvideo_is_actionable_error(tmp_path: Path) -> None:
    garbage = tmp_path / "not_video.mp4"
    garbage.write_bytes(b"esto no es un mp4")
    with pytest.raises(DecodeError, match="not_video.mp4"):
        probe_video(garbage)


def test_iter_frames_count_and_shape(clip_testsrc: Path) -> None:
    """La relación 1:1 frame↔CharMatrix exige contar exactamente (docs/02 E1)."""
    meta = probe_video(clip_testsrc)
    frames = list(iter_frames(clip_testsrc, meta, work_width=160, work_height=90))
    assert len(frames) == meta.n_frames
    assert all(f.shape == (90, 160, 3) and f.dtype.name == "uint8" for f in frames)


def test_iter_frames_content_not_black(clip_testsrc: Path) -> None:
    meta = probe_video(clip_testsrc)
    first = next(iter(iter_frames(clip_testsrc, meta, 160, 90)))
    assert int(first.max()) > 100  # testsrc2 tiene contenido, no negro


def test_extract_audio_roundtrip(clip_testsrc: Path, tmp_path: Path) -> None:
    dest = extract_audio(clip_testsrc, tmp_path / "audio")
    assert dest is not None
    assert dest.suffix == ".mka"
    assert dest.stat().st_size > 0


def test_extract_audio_none_when_silent(clip_silent: Path, tmp_path: Path) -> None:
    assert extract_audio(clip_silent, tmp_path / "audio") is None
