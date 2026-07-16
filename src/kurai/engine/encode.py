"""Etapa 9 — Encode y mux (docs/02 E9, ADR-003).

h264_nvenc preset p5 -tune hq cuando hay NVENC; libx264 CRF como fallback.
CQ/CRF ≤ 23 porque el ASCII es alta frecuencia espacial: más compresión hace
papilla los glifos. Audio: -c:a copy en el mismo comando (mux directo, jamás
recodificar). Sin outputs parciales: si algo falla, el destino se borra.

Decisiones auditadas contra la práctica de la industria (ver PR de Fase 0):
- NVENC en calidad constante REAL exige `-b:v 0` junto a `-cq` — sin eso el
  bitrate queda capado al default (~2 Mbps) y los glifos se hacen papilla.
- La conversión RGB→YUV se fija a BT.709 (`out_color_matrix`) y se tagea el
  stream: swscale usa BT.601 por defecto y los players HD asumen BT.709 —
  sin esto el color queda corrido.
- stderr va a archivo, nunca a PIPE sin lector (deadlock si ffmpeg escupe
  suficientes warnings mientras nosotros escribimos frames).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import IO

import numpy as np
import numpy.typing as npt

from kurai.types import VideoMeta

# RGB full-range → YUV limitado BT.709 con redondeo preciso, y tagging completo
# vía setparams (las opciones -colorspace/-color_trc del encoder solo escriben
# la matriz; el filtro propaga los tres campos al VUI — verificado con ffprobe).
_COLOR_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv:flags=full_chroma_int+accurate_rnd,"
    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
)


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
            meta.fps_expr,
            "-i",
            "-",
        ]
        if audio_path is not None:
            cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
        cmd += ["-vf", _COLOR_FILTER]
        if use_nvenc:
            # -b:v 0 es obligatorio para calidad constante real con nvenc
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
                "-b:v",
                "0",
            ]
        else:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq)]
        cmd += ["-pix_fmt", "yuv420p", str(dest)]
        self._cmd = cmd
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_file: IO[bytes] | None = None

    def _read_stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        self._stderr_file.seek(0)
        return self._stderr_file.read().decode(errors="replace").strip()

    def write(self, frame_rgb: npt.NDArray[np.uint8]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise EncodeError("Encoder no está abierto (usar como context manager)")
        if frame_rgb.shape != self._frame_shape:
            raise EncodeError(
                f"Frame {frame_rgb.shape} no coincide con el contrato {self._frame_shape}"
            )
        try:
            self._proc.stdin.write(frame_rgb.tobytes())
        except BrokenPipeError:
            # ffmpeg murió a mitad del stream: reportar SU error, no el pipe
            self._proc.wait()
            stderr = self._read_stderr()
            self.dest.unlink(missing_ok=True)
            raise EncodeError(
                f"ffmpeg terminó inesperadamente encodeando {self.dest.name}: {stderr}"
            ) from None
        self.frames_written += 1

    def __enter__(self) -> Encoder:
        self._stderr_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stderr=self._stderr_file)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        proc = self._proc
        self._proc = None
        try:
            if proc is None:
                return
            if exc is not None:
                # Job abortado aguas arriba: matar ffmpeg y no dejar output parcial
                proc.kill()
                proc.wait()
                self.dest.unlink(missing_ok=True)
                return

            assert proc.stdin is not None
            proc.stdin.close()
            if proc.wait() != 0:
                stderr = self._read_stderr()
                self.dest.unlink(missing_ok=True)
                raise EncodeError(f"ffmpeg falló encodeando {self.dest.name}: {stderr}")
        finally:
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None
