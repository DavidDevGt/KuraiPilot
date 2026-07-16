"""Orquestador del pipeline (docs/01 §5, docs/02 §11).

Fase 0: pipeline determinista streaming — decode → grids → gamma → Bayer+
cuantización → histéresis → atlas → encode+mux. Los presets con componentes
de fases posteriores (saliencia, FS, fg+bg) levantan NotImplementedError con
la fase que los trae. RAM constante: los frames fluyen, nunca se acumulan.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kurai.config import ColorMode, DitherMode, JobConfig, RefineMode
from kurai.engine.decode import extract_audio, iter_frames, probe_video
from kurai.engine.dither import bayer_offsets
from kurai.engine.encode import Encoder
from kurai.engine.grid import LUMA_WEIGHTS, grid_shape, to_grids, work_resolution
from kurai.engine.mapping import apply_gamma, quantize
from kurai.engine.stability import HysteresisState
from kurai.probe import probe
from kurai.render.atlas import build_atlas, compose
from kurai.render.glyphs import ramp_chars
from kurai.types import CharMatrix, VideoMeta

MAX_QUEUE_FRAMES = 64  # docs/02 §11 — hoy el streaming es sincrónico; aplica al paralelizar

ProgressFn = Callable[[int, int], None]  # (frames_done, total)


def _guard_phase(cfg: JobConfig) -> None:
    """Componentes de fases futuras fallan ANTES de decodificar nada."""
    if cfg.preset.saliency:
        raise NotImplementedError("Fase 1")
    if cfg.preset.refine is not RefineMode.OFF:
        raise NotImplementedError("Fase 1")
    if cfg.preset.dither is DitherMode.FLOYD_STEINBERG:
        raise NotImplementedError("Fase 2")
    if cfg.preset.color is ColorMode.FG_BG:
        raise NotImplementedError("Fase 2")
    if cfg.auto_scene:
        raise NotImplementedError("Fase 3")


def cells_to_charmatrix(
    cell_frame: npt.NDArray[np.uint8],
    state: HysteresisState,
    offsets: npt.NDArray[np.float32],
    levels: int,
    gamma: float,
) -> CharMatrix:
    """Etapas 4-7 sobre un frame ya reducido a celdas (rows, cols, 3).

    El núcleo compartido del fast path de run_job y de los tests que alimentan
    frames decodificados reales (con ruido de códec incluido).
    """
    color_grid = cell_frame.astype(np.float32) * np.float32(1.0 / 255.0)
    luma_grid = (color_grid @ LUMA_WEIGHTS).astype(np.float32)
    lg = apply_gamma(luma_grid, gamma)
    state.detect_scene_cut(lg)
    idx = state.apply(lg, quantize(lg, levels, offsets), levels)
    return CharMatrix(char_idx=idx, fg=np.ascontiguousarray(cell_frame))


def run_job(
    input_file: Path,
    cfg: JobConfig,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Ejecuta el job completo y devuelve el path del output."""
    _guard_phase(cfg)

    meta = probe_video(input_file)
    rows, cols = grid_shape(meta.width, meta.height, cfg.cols)
    work_w, work_h = work_resolution(rows, cols)

    ramp = ramp_chars(cfg.preset.ramp)
    levels = len(ramp)
    atlas = build_atlas(ramp)
    offsets = bayer_offsets(rows, cols, levels)
    state = HysteresisState(rows, cols)

    env = probe()
    use_hw = env.hw_pipeline and not env.gpu_disabled_by_env

    output = cfg.output or input_file.with_name(f"{input_file.stem}_ascii.mp4")
    out_meta = VideoMeta(
        width=work_w,
        height=work_h,
        fps=meta.fps,
        n_frames=meta.n_frames,
        duration_s=meta.duration_s,
        rotation=0,
        has_audio=meta.has_audio,
        codec="h264",
        fps_rational=meta.fps_rational,
    )

    # Fast path E1+E2 fusionadas: si ninguna etapa necesita píxeles a resolución
    # de trabajo (refine/saliencia son Fase 1+), ffmpeg entrega la grilla
    # directamente con scale=area — el promedio por celda corre dentro de
    # ffmpeg (SIMD, proceso paralelo) y el pipe baja ~128× (docs/02 E1).
    # En Fase 0 el guard de arriba garantiza que siempre aplica.
    with tempfile.TemporaryDirectory(prefix="kurai-job-") as tmp:
        audio = extract_audio(input_file, Path(tmp) / "audio")
        with Encoder(output, out_meta, audio, use_nvenc=use_hw) as enc:
            for i, cell_frame in enumerate(
                iter_frames(input_file, meta, cols, rows, hwaccel=use_hw)
            ):
                cm = cells_to_charmatrix(cell_frame, state, offsets, levels, cfg.preset.gamma)
                enc.write(compose(cm, atlas, cfg.preset.color))
                if on_progress is not None:
                    on_progress(i + 1, meta.n_frames)
    return output


def frames_to_charmatrices(
    frames: list[npt.NDArray[np.uint8]],
    rows: int,
    cols: int,
    cfg: JobConfig,
) -> list[CharMatrix]:
    """Etapas 2-7 puras sobre frames en memoria: la vía de los golden files
    (docs/06 §1) — sin códecs de por medio, reproducible entre versiones de ffmpeg."""
    _guard_phase(cfg)
    ramp = ramp_chars(cfg.preset.ramp)
    levels = len(ramp)
    offsets = bayer_offsets(rows, cols, levels)
    state = HysteresisState(rows, cols)
    result: list[CharMatrix] = []
    for frame in frames:
        luma_grid, color_grid = to_grids(frame.astype(np.uint8), rows, cols)
        lg = apply_gamma(luma_grid, cfg.preset.gamma)
        state.detect_scene_cut(lg)
        idx = state.apply(lg, quantize(lg, levels, offsets), levels)
        result.append(
            CharMatrix(char_idx=idx, fg=np.clip(color_grid * 255.0, 0, 255).astype(np.uint8))
        )
    return result
