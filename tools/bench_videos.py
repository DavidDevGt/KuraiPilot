#!/usr/bin/env python3
"""Benchmark de tiempos reales por video: retro vs detallado (ver BENCHMARK.md).

Complementa `kurai bench` (clip sintético fijo para el gate de CI) con tiempos
sobre la biblioteca de clips variados de output/samples/. Mide run_job() con
time.perf_counter (excluye arranque del intérprete); warmup global para
amortizar la init de CUDA/ONNX; la mejor de 2 corridas por (video, preset).
Solo vale en la máquina de referencia (ADR-001).

Uso: uv run python tools/bench_videos.py
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from kurai.config import JobConfig, load_preset
from kurai.engine.decode import probe_video
from kurai.engine.pipeline import run_job

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "output" / "samples"
OUT_JSON = ROOT / "output" / "bench_results.json"
COLS = 160
RUNS = 2

VIDEOS = [
    "sintel_trailer_720p",
    "bbb_1080p_10s",
    "jellyfish_1080p_10s",
    "tears_of_steel_60s",
    "notld_1968_60s",
    "portrait_test",
    "artemis_launch",
    "tiktok_dancer_white",
    "tiktok_dance_neon",
    "tiktok_dance_red",
    "tiktok_city_night",
    "tiktok_coast",
]


def time_run(inp: Path, preset: str, out: Path) -> float:
    cfg = JobConfig(preset=load_preset(preset), cols=COLS, output=out)
    t0 = time.perf_counter()
    run_job(inp, cfg)
    return time.perf_counter() - t0


def main() -> int:
    if not SAMPLES.is_dir():
        print(f"No existe {SAMPLES} — descargá clips de prueba primero.")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="kurai-bench-"))
    warm = SAMPLES / "jellyfish_1080p_10s.mp4"
    if warm.is_file():
        time_run(warm, "detallado", tmp / "warmup.mp4")  # init CUDA/ONNX

    rows: list[dict[str, object]] = []
    for name in VIDEOS:
        inp = SAMPLES / f"{name}.mp4"
        if not inp.is_file():
            continue
        meta = probe_video(inp)
        row: dict[str, object] = {
            "video": name,
            "w": meta.width,
            "h": meta.height,
            "fps": round(meta.fps, 2),
            "dur_s": round(meta.duration_s, 1),
            "frames": meta.n_frames,
        }
        for preset in ("retro", "detallado"):
            times = [time_run(inp, preset, tmp / f"{name}_{preset}.mp4") for _ in range(RUNS)]
            best = min(times)
            row[f"{preset}_s"] = round(best, 2)
            row[f"{preset}_x"] = round(meta.duration_s / best, 1) if best > 0 else 0.0
        rows.append(row)
        print(
            f"{name:24s} {meta.width}x{meta.height} {meta.n_frames:4d}f | "
            f"retro {row['retro_s']}s ({row['retro_x']}x) | "
            f"detallado {row['detallado_s']}s ({row['detallado_x']}x)"
        )

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"\nJSON -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
