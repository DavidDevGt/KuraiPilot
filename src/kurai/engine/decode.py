"""Etapa 1 — Decode y demux (docs/02 E1, ADR-003).

Contrato: ffmpeg subprocess con NVDEC + scale_cuda cuando esté disponible,
fallback transparente a software. VFR→CFR. Audio apartado con -c:a copy.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kurai.types import VideoMeta


def probe_video(path: Path) -> VideoMeta:
    """Extrae metadatos vía ffprobe -print_format json. Respeta rotation."""
    raise NotImplementedError("Fase 0")


def iter_frames(
    path: Path, meta: VideoMeta, work_width: int, work_height: int
) -> Iterator[npt.NDArray[np.uint8]]:
    """Frames RGB (work_height, work_width, 3) a resolución de trabajo.

    El scale ocurre en GPU (scale_cuda) antes de cruzar PCIe cuando hay NVDEC.
    """
    raise NotImplementedError("Fase 0")


def extract_audio(path: Path, dest: Path) -> Path | None:
    """Aparta el stream de audio sin recodificar. None si el video no tiene audio."""
    raise NotImplementedError("Fase 0")
