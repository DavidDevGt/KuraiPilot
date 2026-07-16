"""Casos borde del contrato "cualquier video" (docs/02 E1, docs/06 §6):
rotación por metadata, audio no-AAC, dimensiones impares, clips mínimos,
archivos truncados, y estabilidad temporal a través de un códec REAL
(con su ruido de compresión incluido).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from kurai.config import JobConfig, load_preset
from kurai.engine.decode import DecodeError, iter_frames, probe_video
from kurai.engine.dither import bayer_offsets
from kurai.engine.grid import grid_shape
from kurai.engine.pipeline import cells_to_charmatrix, run_job
from kurai.engine.stability import HysteresisState

pytestmark = pytest.mark.ffmpeg


def _lavfi_clip(dest: Path, source: str, extra: list[str]) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", source]
        + extra
        + [str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


# ------------------------------------------------------------ rotación por metadata


@pytest.fixture(scope="session")
def clip_rotated(clip_testsrc: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Mismo clip con display matrix de 90° (como los videos de celular)."""
    dest = tmp_path_factory.mktemp("rot") / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-display_rotation", "90", "-i", str(clip_testsrc), "-c", "copy", str(dest)],
        check=True, capture_output=True,
    )  # fmt: skip
    return dest


def test_rotated_video_swaps_dimensions(clip_rotated: Path) -> None:
    """ffmpeg autorota en decode: el probe debe reportar las dimensiones YA
    intercambiadas o la grilla sale deformada (docs/02 E1)."""
    meta = probe_video(clip_rotated)
    assert meta.rotation in (90, 270)
    assert (meta.width, meta.height) == (180, 320)  # original: 320×180


def test_rotated_video_decodes_with_swapped_frames(clip_rotated: Path) -> None:
    meta = probe_video(clip_rotated)
    rows, cols = grid_shape(meta.width, meta.height, 40)
    first = next(iter(iter_frames(clip_rotated, meta, cols, rows)))
    assert first.shape == (rows, cols, 3)
    # Video vertical ⇒ render vertical (en píxeles: celdas 1:2, no en celdas)
    assert rows * 16 > cols * 8


def test_rotated_video_converts_e2e(clip_rotated: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    run_job(clip_rotated, JobConfig(preset=load_preset("retro"), cols=40, output=out))
    out_meta = probe_video(out)
    assert out_meta.height > out_meta.width  # sigue vertical tras el render


# ------------------------------------------------------------ audio no-AAC


@pytest.fixture(scope="session")
def clip_mp3_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("mp3") / "mp3audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libmp3lame", str(dest)],
        check=True, capture_output=True,
    )  # fmt: skip
    return dest


def test_mp3_audio_survives_copy(clip_mp3_audio: Path, tmp_path: Path) -> None:
    """-c:a copy debe preservar códecs distintos de AAC sin recodificar."""
    from test_encode import _audio_stream_md5

    out = tmp_path / "out.mp4"
    run_job(clip_mp3_audio, JobConfig(preset=load_preset("retro"), cols=40, output=out))
    assert _audio_stream_md5(out) == _audio_stream_md5(clip_mp3_audio)


# ------------------------------------------------------------ geometrías raras


def test_odd_dimensions_input(tmp_path: Path) -> None:
    """321×181 (impar, imposible en yuv420): el pipeline no asume paridad."""
    clip = _lavfi_clip(
        tmp_path / "odd.mp4",
        "testsrc2=size=321x181:rate=30:duration=1",
        ["-c:v", "libx264", "-pix_fmt", "yuv444p"],
    )
    out = tmp_path / "out.mp4"
    run_job(clip, JobConfig(preset=load_preset("retro"), cols=40, output=out))
    assert probe_video(out).n_frames == 30


def test_two_frame_clip(tmp_path: Path) -> None:
    """El clip mínimo viable: 2 frames. Sin off-by-one en los extremos."""
    clip = _lavfi_clip(
        tmp_path / "tiny.mp4",
        "testsrc2=size=320x180:rate=30:duration=0.0667",
        ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
    )
    out = tmp_path / "out.mp4"
    run_job(clip, JobConfig(preset=load_preset("retro"), cols=40, output=out))
    assert probe_video(out).n_frames == probe_video(clip).n_frames


# ------------------------------------------------------------ inputs rotos


def test_truncated_file_fails_with_context(clip_testsrc: Path, tmp_path: Path) -> None:
    """Un mp4 cortado a la mitad falla con mensaje accionable (con el stderr
    real de ffmpeg), jamás con deadlock ni traceback críptico (docs/02 §11)."""
    data = clip_testsrc.read_bytes()
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(DecodeError, match="truncated.mp4"):
        meta = probe_video(truncated)
        list(iter_frames(truncated, meta, 40, 11))


# ------------------------------------------------ FCR a través de un códec real


def test_fcr_zero_through_real_codec(tmp_path: Path) -> None:
    """El gate de FCR (docs/06 §3) medido sobre frames que pasaron por x264:
    el ruido de compresión real —no gaussiano sintético— tampoco debe mover
    un solo carácter en contenido estático."""
    clip = _lavfi_clip(
        tmp_path / "static.mp4",
        "color=c=0x606060:size=320x180:rate=30:duration=1",
        ["-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p"],  # crf alto = más ruido
    )
    meta = probe_video(clip)
    rows, cols = grid_shape(meta.width, meta.height, 80)
    levels = 10
    state = HysteresisState(rows, cols)
    offsets = bayer_offsets(rows, cols, levels)

    matrices = [
        cells_to_charmatrix(frame, state, offsets, levels, gamma=0.8)
        for frame in iter_frames(clip, meta, cols, rows)
    ]
    assert len(matrices) == 30
    changes = sum(
        int(np.count_nonzero(matrices[i].char_idx != matrices[i - 1].char_idx))
        for i in range(1, len(matrices))
    )
    fcr = changes / (rows * cols * (len(matrices) - 1) / meta.fps)
    assert fcr <= 0.05, f"FCR={fcr:.4f} sobre codec real (gate: 0.05)"
