"""Orquestador del pipeline (docs/01 §5, docs/02 §11).

DAG de etapas con colas bounded (backpressure, K=64 frames). Fallo de etapa
opcional-IA degrada con warning; fallo de etapa obligatoria aborta el job
reportando el timestamp, sin outputs parciales silenciosos.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kurai.config import JobConfig

MAX_QUEUE_FRAMES = 64  # docs/02 §11: RAM constante en videos de horas

ProgressFn = Callable[[int, int, float], None]  # (frames_done, total, speed_factor)


def run_job(
    input_file: Path,
    cfg: JobConfig,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Ejecuta el job completo y devuelve el path del output.

    Orden: probe → decode/demux → grids → [saliencia] → mapeo(+dither) →
    [refine] → estabilidad → render → encode+mux. Ver docs/02 para cada contrato.
    """
    raise NotImplementedError("Fase 0")
