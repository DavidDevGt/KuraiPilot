"""Tipos centrales del sistema. La CharMatrix es el artefacto canónico (docs/01 §5):
todo lo que se testea con golden files y todo export es una proyección de ella.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class VideoMeta:
    """Metadatos de entrada tras el probe de la Etapa 1 (docs/02 E1)."""

    width: int
    height: int
    fps: float  # ya normalizado a CFR; solo para cálculo/display
    n_frames: int
    duration_s: float
    rotation: int  # grados; debe estar ya aplicado en decode
    has_audio: bool
    codec: str
    # Racional exacto ("30000/1001") para los filtros fps= y -r de ffmpeg:
    # el float formateado acumula drift en videos largos. Vacío = usar fps.
    fps_rational: str = ""

    @property
    def fps_expr(self) -> str:
        """La expresión de frame rate que se pasa a ffmpeg."""
        return self.fps_rational or f"{self.fps}"


@dataclass
class CharMatrix:
    """Artefacto canónico por frame.

    Garantía G4: mismo input + misma config produce CharMatrix bit a bit idéntica.
    Comparable con == exacto; nunca con tolerancia.
    """

    char_idx: npt.NDArray[np.uint8]  # (rows, cols) índice en la rampa/atlas
    fg: npt.NDArray[np.uint8]  # (rows, cols, 3) RGB
    bg: npt.NDArray[np.uint8] | None = None  # (rows, cols, 3), solo modo fg+bg

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.char_idx.shape
        return rows, cols

    def equals(self, other: CharMatrix) -> bool:
        if not np.array_equal(self.char_idx, other.char_idx):
            return False
        if not np.array_equal(self.fg, other.fg):
            return False
        if self.bg is None and other.bg is None:
            return True
        if self.bg is None or other.bg is None:
            return False
        return np.array_equal(self.bg, other.bg)


@dataclass
class FrameContext:
    """Estado que viaja con cada frame a través del pipeline (docs/02).

    Las etapas leen/escriben campos propios; el orquestador es el dueño del ciclo de vida.
    """

    index: int
    luma_grid: npt.NDArray[np.float32] | None = None  # (rows, cols), E2
    color_grid: npt.NDArray[np.float32] | None = None  # (rows, cols, 3), E2
    density_map: npt.NDArray[np.float32] | None = None  # (rows, cols), E3; None = 1.0
    is_scene_cut: bool = False  # E1/PySceneDetect; resetea histéresis en E7
    char_matrix: CharMatrix | None = None
    warnings: list[str] = field(default_factory=list)
