"""Modo terminal live (docs/01 §4): reproduce el video como ANSI en stdout,
30 fps sostenidos, determinista puro, sin IA y sin render a píxeles.

Pacing por reloj de pared con drop de frames: si el terminal no da abasto,
se saltan frames para no acumular retraso (un live que se atrasa no es live).
Sin audio: es un modo de demostración, no un player (docs/01 §4).
"""

from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from kurai.config import ColorMode, JobConfig
from kurai.engine.decode import iter_frames, probe_video
from kurai.engine.dither import bayer_offsets
from kurai.engine.grid import grid_shape
from kurai.engine.pipeline import cells_to_charmatrix
from kurai.engine.stability import HysteresisState
from kurai.render.ansi import ENTER_ALT_SCREEN, EXIT_ALT_SCREEN, charmatrix_to_ansi
from kurai.render.glyphs import ramp_chars

Clock = Callable[[], float]


class TextSink(Protocol):
    """Lo mínimo que el live necesita de stdout (inyectable en tests)."""

    def write(self, s: str, /) -> object: ...
    def flush(self) -> object: ...


def terminal_grid(input_w: int, input_h: int, max_cols: int | None = None) -> tuple[int, int]:
    """Grilla que cabe en el terminal actual respetando el aspecto del video.

    Las celdas del terminal ya son ~1:2 (como los glifos 8×16), así que
    grid_shape aplica igual; solo se recorta al tamaño de la ventana.
    """
    term = shutil.get_terminal_size()
    cols_fit = max_cols or term.columns
    cols = min(cols_fit, term.columns)
    rows, cols = grid_shape(input_w, input_h, cols)
    if rows > term.lines - 1:  # dejar una línea para no forzar scroll
        scale = (term.lines - 1) / rows
        cols = max(20, int(cols * scale))
        rows, cols = grid_shape(input_w, input_h, cols)
    return rows, cols


def run_live(
    input_file: Path,
    cfg: JobConfig,
    max_cols: int | None = None,
    out: TextSink | None = None,
    clock: Clock = time.monotonic,
    max_frames: int | None = None,
) -> tuple[int, int]:
    """Reproduce el video en ANSI. Devuelve (frames_mostrados, frames_saltados).

    max_cols=None ⇒ decide el ancho del terminal (cfg.cols del export no aplica
    acá: el límite físico es la ventana). `out`/`clock`/`max_frames` existen
    para testear el pacing sin terminal real.
    """
    stream: TextSink = out if out is not None else sys.stdout
    write = stream.write
    flush = stream.flush

    meta = probe_video(input_file)
    rows, cols = terminal_grid(meta.width, meta.height, max_cols)
    ramp = ramp_chars(cfg.preset.ramp)
    levels = len(ramp)
    offsets = bayer_offsets(rows, cols, levels)
    state = HysteresisState(rows, cols)
    color = cfg.preset.color if cfg.preset.color is not ColorMode.FG_BG else ColorMode.FG

    frame_period = 1.0 / meta.fps
    shown = skipped = 0
    write(ENTER_ALT_SCREEN)
    try:
        start = clock()
        for i, cell_frame in enumerate(iter_frames(input_file, meta, cols, rows)):
            if max_frames is not None and i >= max_frames:
                break
            cm = cells_to_charmatrix(cell_frame, state, offsets, levels, cfg.preset.gamma)
            due = start + i * frame_period
            now = clock()
            if now > due + frame_period:
                skipped += 1  # vamos tarde: la histéresis ya vio el frame; no se dibuja
                continue
            if due > now:
                time.sleep(due - now)
            write(charmatrix_to_ansi(cm, ramp, color))
            flush()
            shown += 1
    finally:
        write(EXIT_ALT_SCREEN)
        flush()
    return shown, skipped
