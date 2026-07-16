"""Etapa 9 — Encode y mux (docs/02 E9, ADR-003).

h264_nvenc preset p5 -tune hq, CQ ≤ 23 (el ASCII es alta frecuencia espacial:
más compresión hace papilla los glifos). Audio: -c:a copy, jamás recodificar
salvo incompatibilidad de contenedor. n_frames_out == n_frames_in.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from kurai.types import VideoMeta


class Encoder:
    """Context manager: abre el subprocess ffmpeg, recibe frames RGB, muxea audio al cerrar."""

    def __init__(self, dest: Path, meta: VideoMeta, audio_path: Path | None, cq: int = 21) -> None:
        raise NotImplementedError("Fase 0")

    def write(self, frame_rgb: npt.NDArray[np.uint8]) -> None:
        raise NotImplementedError("Fase 0")

    def __enter__(self) -> Encoder:
        raise NotImplementedError("Fase 0")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError("Fase 0")
