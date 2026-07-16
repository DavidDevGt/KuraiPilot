"""Sesión de preview: el estado de UN cliente (docs/03 §3, gate en docs/07 0.5).

La garantía central es por construcción: compute() usa exactamente
engine.pipeline.cells_to_charmatrix — el mismo código del export. El test de
igualdad del gate (test_preview.py) lo verifica frame a frame contra run_job.

Cambios de rampa/gamma/color: recomputan el frame actual desde el último
cell_frame cacheado (microsegundos — el gate de <100 ms se cumple sobrado)
y resetean la histéresis (el estado comprometido de una config no vale para
otra). Cambio de cols: exige re-decode con seek (start_s) al frame actual.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kurai.config import ColorMode, Ramp
from kurai.engine.decode import iter_frames, probe_video
from kurai.engine.dither import bayer_offsets
from kurai.engine.grid import grid_shape
from kurai.engine.pipeline import cells_to_charmatrix
from kurai.engine.stability import HysteresisState
from kurai.render.atlas import build_atlas
from kurai.render.glyphs import ramp_chars
from kurai.types import CharMatrix, VideoMeta

MAX_PREVIEW_COLS = 200  # docs/05 §2: preview interactivo hasta 200 cols


@dataclass
class PreviewConfig:
    cols: int = 120
    ramp: Ramp = Ramp.SHORT
    gamma: float = 0.8
    color: ColorMode = ColorMode.MONO

    def clamped(self) -> PreviewConfig:
        return PreviewConfig(
            cols=max(20, min(self.cols, MAX_PREVIEW_COLS)),
            ramp=self.ramp,
            gamma=max(0.1, min(self.gamma, 2.0)),
            color=self.color,
        )


class PreviewSession:
    """Estado por cliente. No es thread-safe: el server la usa desde una sola
    task por WebSocket (el decode corre en thread pero entrega por generador)."""

    def __init__(self, input_file: Path, config: PreviewConfig | None = None) -> None:
        self.input_file = input_file
        self.meta: VideoMeta = probe_video(input_file)
        self.config = (config or PreviewConfig()).clamped()
        self.frame_idx = 0
        self.playing = False
        self._last_cell_frame: npt.NDArray[np.uint8] | None = None
        self._apply_config()

    # ------------------------------------------------------------- geometría

    @property
    def grid(self) -> tuple[int, int]:
        return grid_shape(self.meta.width, self.meta.height, self.config.cols)

    def _apply_config(self) -> None:
        rows, cols = self.grid
        self.ramp_str = ramp_chars(self.config.ramp)
        self.levels = len(self.ramp_str)
        self.atlas = build_atlas(self.ramp_str)
        self._offsets = bayer_offsets(rows, cols, self.levels)
        self._state = HysteresisState(rows, cols)

    # ------------------------------------------------------------- cómputo

    def compute(self, cell_frame: npt.NDArray[np.uint8]) -> CharMatrix:
        """El MISMO camino del export (cells_to_charmatrix) — gate de igualdad."""
        self._last_cell_frame = cell_frame
        return cells_to_charmatrix(
            cell_frame, self._state, self._offsets, self.levels, self.config.gamma
        )

    def frames(self, start_idx: int = 0) -> Iterator[npt.NDArray[np.uint8]]:
        """Cell-frames decodificados desde start_idx (seek por -ss)."""
        rows, cols = self.grid
        start_s = start_idx / self.meta.fps
        yield from iter_frames(self.input_file, self.meta, cols, rows, start_s=start_s)

    # ------------------------------------------------------------- comandos

    def update_config(self, **changes: object) -> tuple[bool, CharMatrix | None]:
        """Aplica cambios de config. → (necesita_re_decode, frame_recomputado).

        rampa/gamma/color: recompute inmediato del frame actual (<100 ms).
        cols: la grilla cambia ⇒ el caller debe reiniciar el decode con seek.
        """
        new = PreviewConfig(
            cols=int(changes.get("cols", self.config.cols)),  # type: ignore[call-overload]
            ramp=Ramp(str(changes.get("ramp", self.config.ramp.value))),
            gamma=float(changes.get("gamma", self.config.gamma)),  # type: ignore[arg-type]
            color=ColorMode(str(changes.get("color", self.config.color.value))),
        ).clamped()

        needs_redecode = new.cols != self.config.cols
        self.config = new
        self._apply_config()  # resetea histéresis: config nueva, estado nuevo

        if needs_redecode or self._last_cell_frame is None:
            return needs_redecode, None
        return False, self.compute(self._last_cell_frame)

    def seek(self, frame_idx: int) -> int:
        """Fija la posición (clampeada). El caller reinicia el decode; la
        histéresis se resetea: un seek es un corte de escena por definición."""
        self.frame_idx = max(0, min(frame_idx, self.meta.n_frames - 1))
        self._state.reset()
        return self.frame_idx
