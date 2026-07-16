"""kurai bench (docs/05 §6) — se construye ANTES que las etapas del pipeline.

Modo passthrough: decode → (nada) → encode, con los frames cruzando Python.
Mide el techo de la infraestructura (pipes, subprocess, copias) — todo lo que
las etapas 2-8 van a tener que compartir. El resultado aceptado se versiona en
bench/results/accepted.json; --check falla con regresión >10% de speed factor.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kurai.engine.decode import extract_audio, iter_frames, probe_video
from kurai.engine.encode import Encoder

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "bench"
ACCEPTED = BENCH_DIR / "results" / "accepted.json"
LAST_RUN = BENCH_DIR / "last_run.json"
CACHE_DIR = BENCH_DIR / ".cache"

REGRESSION_TOLERANCE = 0.10  # docs/05 §6

# Clip de referencia: 1080p30, 10 s, testsrc2 + tono. Sintético y determinista:
# el bench no depende de material con derechos ni de descargas.
CLIP_W, CLIP_H, CLIP_FPS, CLIP_SECONDS = 1920, 1080, 30, 10


@dataclass
class BenchResult:
    mode: str
    clip: str
    encoder: str
    video_seconds: float
    wall_seconds: float
    speed_factor: float
    frames: int
    commit: str
    timestamp: str


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def ensure_bench_clip() -> Path:
    """Genera (una vez) y cachea el clip de referencia."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clip = CACHE_DIR / f"ref_{CLIP_W}x{CLIP_H}_{CLIP_FPS}fps_{CLIP_SECONDS}s.mp4"
    if clip.exists():
        return clip
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={CLIP_W}x{CLIP_H}:rate={CLIP_FPS}:duration={CLIP_SECONDS}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={CLIP_SECONDS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    return clip


def _make_result(
    mode: str, clip: Path, encoder: str, video_s: float, wall: float, frames: int
) -> BenchResult:
    return BenchResult(
        mode=mode,
        clip=clip.name,
        encoder=encoder,
        video_seconds=video_s,
        wall_seconds=round(wall, 3),
        speed_factor=round(video_s / wall, 2),
        frames=frames,
        commit=_git_commit(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def run_passthrough(clip: Path, workdir: Path, use_nvenc: bool) -> BenchResult:
    """Decode completo → encode completo, sin procesamiento entre medio.
    Mide el techo de la infraestructura de I/O."""
    meta = probe_video(clip)
    audio = extract_audio(clip, workdir / "audio")
    out = workdir / "passthrough_out.mp4"

    start = time.perf_counter()
    with Encoder(out, meta, audio, use_nvenc=use_nvenc) as enc:
        for frame in iter_frames(clip, meta, meta.width, meta.height, hwaccel=use_nvenc):
            enc.write(frame)
    wall = time.perf_counter() - start
    encoder = "h264_nvenc" if use_nvenc else "libx264"
    return _make_result("passthrough", clip, encoder, meta.duration_s, wall, enc.frames_written)


def run_retro(clip: Path, workdir: Path, use_nvenc: bool) -> BenchResult:
    """Pipeline completo con preset retro — el número del gate de Fase 0
    (target ≥ 4×, docs/05 §2 / docs/07)."""
    from kurai.config import JobConfig, load_preset
    from kurai.engine.pipeline import run_job

    meta = probe_video(clip)
    cfg = JobConfig(preset=load_preset("retro"), cols=160, output=workdir / "retro_out.mp4")
    start = time.perf_counter()
    run_job(clip, cfg)
    wall = time.perf_counter() - start
    encoder = "h264_nvenc" if use_nvenc else "libx264"
    return _make_result("retro", clip, encoder, meta.duration_s, wall, meta.n_frames)


def load_accepted() -> dict[str, BenchResult]:
    """Baselines aceptados, por modo."""
    if not ACCEPTED.exists():
        return {}
    raw = json.loads(ACCEPTED.read_text())
    return {mode: BenchResult(**data) for mode, data in raw.items()}


def save(results: list[BenchResult], accept: bool) -> None:
    as_dict = {r.mode: asdict(r) for r in results}
    LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(json.dumps(as_dict, indent=2) + "\n")
    if accept:
        merged = {mode: asdict(r) for mode, r in load_accepted().items()}
        merged.update(as_dict)
        ACCEPTED.parent.mkdir(parents=True, exist_ok=True)
        ACCEPTED.write_text(json.dumps(merged, indent=2) + "\n")


def check_regression(result: BenchResult, baseline: BenchResult) -> str | None:
    """None si está dentro de tolerancia; mensaje de fallo si hay regresión."""
    if baseline.encoder != result.encoder:
        return (
            f"El baseline se midió con {baseline.encoder} y este run con "
            f"{result.encoder}: no son comparables. Re-aceptar con --accept."
        )
    floor = baseline.speed_factor * (1 - REGRESSION_TOLERANCE)
    if result.speed_factor < floor:
        return (
            f"Regresión: {result.speed_factor}× vs. baseline {baseline.speed_factor}× "
            f"(piso {floor:.2f}×, commit baseline {baseline.commit})"
        )
    return None
