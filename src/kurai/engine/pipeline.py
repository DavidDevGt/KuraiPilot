"""Orquestador del pipeline (docs/01 §5, docs/02 §11).

Fase 0/1: pipeline streaming — decode → grids → gamma → (saliencia E3) → Bayer+
cuantización → histéresis → (edges E5) → atlas → encode+mux. Los presets con
componentes de fases posteriores (FS, CNN, flow, fg+bg, --auto) levantan
NotImplementedError con la fase que los trae. La saliencia (E3) degrada sola:
sin modelo, density ≡ 1.0 y el resultado es el determinista. RAM constante.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kurai.ai.saliency import SaliencyScheduler, load_saliency
from kurai.config import ColorMode, DitherMode, FlickerMode, JobConfig, RefineMode
from kurai.engine.decode import extract_audio, iter_frames, probe_video
from kurai.engine.dither import bayer_offsets
from kurai.engine.edges import refine_edges
from kurai.engine.encode import Encoder
from kurai.engine.grid import LUMA_WEIGHTS, grid_shape, to_grids, work_resolution
from kurai.engine.mapping import apply_gamma, quantize
from kurai.engine.stability import HysteresisState
from kurai.probe import probe
from kurai.render.atlas import build_atlas, compose
from kurai.render.glyphs import directional_glyph_string, ramp_chars
from kurai.types import CharMatrix, VideoMeta

MAX_QUEUE_FRAMES = 64  # docs/02 §11 — hoy el streaming es sincrónico; aplica al paralelizar
SALIENCY_MODEL_FILE = "u2net_lite.onnx"  # models/manifest.toml [models.saliency]

ProgressFn = Callable[[int, int], None]  # (frames_done, total)


def guard_phase(cfg: JobConfig) -> None:
    """Componentes de fases futuras fallan ANTES de decodificar nada.

    Fase 1 (saliencia E3 + edges E5) ya está soportada; lo que sigue bloqueado
    es Fase 2 (CNN de glifos, Floyd-Steinberg, flow, fg+bg) y Fase 3 (--auto).
    La saliencia NO se bloquea acá: degrada sola si falta el modelo.
    """
    if cfg.preset.refine is RefineMode.EDGES_CNN:
        raise NotImplementedError("Fase 2")
    if cfg.preset.dither is DitherMode.FLOYD_STEINBERG:
        raise NotImplementedError("Fase 2")
    if cfg.preset.flicker is FlickerMode.HYSTERESIS_FLOW:
        raise NotImplementedError("Fase 2")
    if cfg.preset.color is ColorMode.FG_BG:
        raise NotImplementedError("Fase 2")
    if cfg.auto_scene:
        raise NotImplementedError("Fase 3")


def saliency_model_path() -> Path:
    """Ruta al ONNX de saliencia (KURAI_MODELS_DIR o models/ del repo, docs/03 §6)."""
    base = os.environ.get("KURAI_MODELS_DIR")
    root = Path(base) if base else Path(__file__).resolve().parents[3] / "models"
    return root / SALIENCY_MODEL_FILE


def cells_to_charmatrix(
    cell_frame: npt.NDArray[np.uint8],
    state: HysteresisState,
    offsets: npt.NDArray[np.float32],
    levels: int,
    gamma: float,
    *,
    scheduler: SaliencyScheduler | None = None,
    frame_idx: int = 0,
    refine: bool = False,
) -> CharMatrix:
    """Etapas 3-7 sobre un frame ya reducido a celdas (rows, cols, 3).

    El núcleo compartido de run_job, el preview y el live. Sin ``scheduler`` ni
    ``refine`` (default) es el camino determinista puro de Fase 0 — mismo
    resultado bit a bit, por eso el preview lo comparte para el gate de igualdad.
    Con ``scheduler`` (E3): la densidad de saliencia modula la rampa por celda.
    Con ``refine`` (E5): los bordes estructurales pasan a glifos direccionales.
    """
    rows, cols = int(cell_frame.shape[0]), int(cell_frame.shape[1])
    color_grid = cell_frame.astype(np.float32) * np.float32(1.0 / 255.0)
    luma_grid = (color_grid @ LUMA_WEIGHTS).astype(np.float32)
    lg = apply_gamma(luma_grid, gamma)
    scene_cut = state.detect_scene_cut(lg)
    density = (
        scheduler.density_for(frame_idx, cell_frame, rows, cols, scene_cut)
        if scheduler is not None
        else None
    )
    idx = state.apply(lg, quantize(lg, levels, offsets, density), levels)
    if refine:
        idx = refine_edges(lg, idx, levels)
    return CharMatrix(char_idx=idx, fg=np.ascontiguousarray(cell_frame))


def run_job(
    input_file: Path,
    cfg: JobConfig,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Ejecuta el job completo y devuelve el path del output."""
    guard_phase(cfg)

    meta = probe_video(input_file)
    rows, cols = grid_shape(meta.width, meta.height, cfg.cols)
    work_w, work_h = work_resolution(rows, cols)

    ramp = ramp_chars(cfg.preset.ramp)
    levels = len(ramp)
    # El atlas gana los glifos direccionales cuando E5 `edges` está activo: el
    # char_idx refinado indexa `levels + k` (docs/02 E5). El char_idx tonal
    # sigue en [0, levels); la rampa tonal y su hash-guard no cambian.
    refine = cfg.preset.refine is RefineMode.EDGES
    glyphs = ramp + directional_glyph_string() if refine else ramp
    atlas = build_atlas(glyphs)
    offsets = bayer_offsets(rows, cols, levels)
    state = HysteresisState(rows, cols)

    # E3 saliencia: sin el ONNX (ausente o sin onnxruntime) degrada a density≡1.0
    # (idéntico a sin saliencia) con un warning — regla 5, docs/02 §11.
    scheduler: SaliencyScheduler | None = None
    if cfg.preset.saliency:
        model = load_saliency(saliency_model_path())
        if model is None:
            warnings.warn(
                "saliencia: modelo no disponible (models/u2net_lite.onnx u onnxruntime) "
                "— degradando a densidad uniforme; el resto del pipeline es idéntico.",
                UserWarning,
                stacklevel=2,
            )
        scheduler = SaliencyScheduler(model)

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

    # ffmpeg entrega la grilla directamente con scale=area (E1+E2 fusionadas):
    # el promedio por celda corre dentro de ffmpeg (SIMD) y el pipe baja ~128×
    # (docs/02 E1). Saliencia (E3) y edges (E5) operan sobre esas celdas ya
    # reducidas — la saliencia re-muestrea la grilla RGB a 320×320; el mapa de
    # densidad es de resolución de grilla de todos modos (saliencia a resolución
    # de trabajo es un refinamiento futuro, docs/02 E1).
    with tempfile.TemporaryDirectory(prefix="kurai-job-") as tmp:
        audio = extract_audio(input_file, Path(tmp) / "audio")
        with Encoder(output, out_meta, audio, use_nvenc=use_hw) as enc:
            for i, cell_frame in enumerate(
                iter_frames(input_file, meta, cols, rows, hwaccel=use_hw)
            ):
                cm = cells_to_charmatrix(
                    cell_frame,
                    state,
                    offsets,
                    levels,
                    cfg.preset.gamma,
                    scheduler=scheduler,
                    frame_idx=i,
                    refine=refine,
                )
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
    guard_phase(cfg)
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
