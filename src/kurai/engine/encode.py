"""Etapa 9 — Encode y mux (docs/02 E9, ADR-003).

h264_nvenc preset p5 -tune hq cuando hay NVENC; libx264 CRF como fallback.
CQ/CRF ≤ 23 porque el ASCII es alta frecuencia espacial: más compresión hace
papilla los glifos. Audio: -c:a copy en el mismo comando (mux directo, jamás
recodificar). Sin outputs parciales: si algo falla, el destino se borra.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from kurai.types import VideoMeta


class EncodeError(RuntimeError):
    """Fallo de encode/mux; el output parcial ya fue eliminado."""


class Encoder:
    """Context manager: frames RGB por stdin de ffmpeg, mux de audio al cerrar.

    El caller decide nvenc vs libx264 (típicamente desde probe().hw_pipeline);
    este módulo no toca el entorno.
    """

    def __init__(
        self,
        dest: Path,
        meta: VideoMeta,
        audio_path: Path | None,
        cq: int = 21,
        use_nvenc: bool = False,
    ) -> None:
        if not 1 <= cq <= 23:
            raise ValueError(f"cq={cq} fuera de rango: el máximo aceptable es 23 (docs/02 E9)")
        self.dest = dest
        self.meta = meta
        self.frames_written = 0
        self._frame_shape = (meta.height, meta.width, 3)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{meta.width}x{meta.height}",
            "-r",
            f"{meta.fps}",
            "-i",
            "-",
        ]
        if audio_path is not None:
            cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
        if use_nvenc:
            cmd += [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(cq),
            ]
        else:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq)]
        cmd += ["-pix_fmt", "yuv420p", str(dest)]
        self._cmd = cmd
        self._proc: subprocess.Popen[bytes] | None = None

    def write(self, frame_rgb: npt.NDArray[np.uint8]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise EncodeError("Encoder no está abierto (usar como context manager)")
        if frame_rgb.shape != self._frame_shape:
            raise EncodeError(
                f"Frame {frame_rgb.shape} no coincide con el contrato {self._frame_shape}"
            )
        self._proc.stdin.write(frame_rgb.tobytes())
        self.frames_written += 1

    def __enter__(self) -> Encoder:
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if exc is not None:
            # Job abortado aguas arriba: matar ffmpeg y no dejar output parcial
            proc.kill()
            proc.wait()
            self.dest.unlink(missing_ok=True)
            return

        assert proc.stdin is not None
        proc.stdin.close()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        if proc.wait() != 0:
            self.dest.unlink(missing_ok=True)
            raise EncodeError(f"ffmpeg falló encodeando {self.dest.name}: {stderr.strip()}")
