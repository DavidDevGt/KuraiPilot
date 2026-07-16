"""Etapa 1 — Decode y demux (docs/02 E1, ADR-003).

ffmpeg como subprocess con pipes rawvideo. NVDEC (-hwaccel cuda) cuando el
caller lo pide, fallback transparente a software. VFR→CFR con el filtro fps=
para mantener la relación 1:1 frame↔CharMatrix. La rotación por metadata la
aplica ffmpeg solo (autorotate por displaymatrix); acá solo se registra.
Audio apartado con -c:a copy, jamás recodificado.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kurai.types import VideoMeta


class DecodeError(RuntimeError):
    """Input que ffmpeg/ffprobe no puede procesar; el mensaje es accionable."""


def _ffprobe_json(path: Path) -> dict[str, object]:
    proc = subprocess.run(
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
    )
    if proc.returncode != 0:
        raise DecodeError(f"ffprobe no pudo leer {path.name}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)  # type: ignore[no-any-return]  # json.loads → Any


def _parse_fps(stream: dict[str, object]) -> float:
    """avg_frame_rate primero (VFR honesto); r_frame_rate como fallback."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key)
        if isinstance(raw, str) and raw not in ("0/0", "0/1", ""):
            value = Fraction(raw)
            if value > 0:
                return float(value)
    raise DecodeError("No se pudo determinar el frame rate del video")


def _parse_rotation(stream: dict[str, object]) -> int:
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                return int(entry["rotation"]) % 360
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        return int(tags["rotate"]) % 360
    return 0


def probe_video(path: Path) -> VideoMeta:
    """Metadatos vía ffprobe. n_frames es post-normalización CFR (duration×fps)."""
    info = _ffprobe_json(path)
    streams = info.get("streams")
    if not isinstance(streams, list):
        raise DecodeError(f"{path.name}: sin streams")

    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None
    )
    if video is None:
        raise DecodeError(f"{path.name}: no tiene stream de video")
    has_audio = any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams)

    # Duración del STREAM de video, no del contenedor: el formato incluye el
    # audio, y el padding de AAC lo alarga (rompe n_frames = duración × fps).
    duration_raw = video.get("duration")
    if duration_raw is None:
        fmt = info.get("format")
        duration_raw = fmt.get("duration") if isinstance(fmt, dict) else None
    if duration_raw is None:
        raise DecodeError(f"{path.name}: sin duración en metadata")
    duration_s = float(str(duration_raw))

    fps = _parse_fps(video)
    rotation = _parse_rotation(video)
    width, height = int(str(video["width"])), int(str(video["height"]))
    # ffmpeg autorota en decode: si la rotación es 90/270, el frame que sale
    # del pipe ya viene con las dimensiones intercambiadas.
    if rotation in (90, 270):
        width, height = height, width

    return VideoMeta(
        width=width,
        height=height,
        fps=fps,
        n_frames=round(duration_s * fps),
        duration_s=duration_s,
        rotation=rotation,
        has_audio=has_audio,
        codec=str(video.get("codec_name", "unknown")),
    )


def iter_frames(
    path: Path,
    meta: VideoMeta,
    work_width: int,
    work_height: int,
    hwaccel: bool = False,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Frames RGB (work_height, work_width, 3) a resolución de trabajo, CFR.

    Con hwaccel=True el decode corre en NVDEC; como el consumidor está en RAM
    (pipe rawvideo), ffmpeg baja el frame a sistema tras decodificar — el scale
    corre en software sobre el frame ya decodificado. scale_cuda entra recién
    cuando el consumidor sea GPU-residente (Fase 0 tardía, docs/02 E1).
    """
    frame_bytes = work_width * work_height * 3
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i",
        str(path),
        "-vf",
        f"fps={meta.fps},scale={work_width}:{work_height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    completed = False
    try:
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            if len(chunk) < frame_bytes:
                raise DecodeError(
                    f"{path.name}: frame truncado a los "
                    f"{len(chunk)} de {frame_bytes} bytes (¿archivo corrupto?)"
                )
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(work_height, work_width, 3)
        completed = True
    finally:
        # Consumidor que abandona antes del final (GeneratorExit, excepción
        # aguas arriba): matar ffmpeg sin reportar su broken pipe como error.
        if not completed:
            proc.kill()
        proc.stdout.close()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        code = proc.wait()
        if completed and code != 0:
            raise DecodeError(f"ffmpeg falló decodificando {path.name}: {stderr.strip()}")


def extract_audio(path: Path, dest: Path) -> Path | None:
    """Aparta el audio sin recodificar (-c:a copy) a un contenedor Matroska
    (aguanta cualquier códec). None si el video no tiene audio."""
    if not probe_video(path).has_audio:
        return None
    dest = dest.with_suffix(".mka")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-c:a",
            "copy",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DecodeError(f"No se pudo extraer el audio de {path.name}: {proc.stderr.strip()}")
    return dest
