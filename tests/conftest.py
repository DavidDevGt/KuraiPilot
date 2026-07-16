"""Fixtures compartidos: frames sintéticos deterministas y clips generados con
ffmpeg lavfi (no van al repo — se generan por sesión de test, docs/06 §4).

Marcadores: `gpu` se salta sin GPU o con KURAI_DISABLE_GPU=1; `ffmpeg` se salta
sin ffmpeg en PATH (en CI siempre está — ver .github/workflows/ci.yml).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest


def _gpu_available() -> bool:
    return os.environ.get("KURAI_DISABLE_GPU") != "1" and shutil.which("nvidia-smi") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="GPU no disponible o KURAI_DISABLE_GPU=1")
    skip_ffmpeg = pytest.mark.skip(reason="ffmpeg no está en PATH")
    has_ffmpeg = shutil.which("ffmpeg") is not None
    for item in items:
        if "gpu" in item.keywords and not _gpu_available():
            item.add_marker(skip_gpu)
        if "ffmpeg" in item.keywords and not has_ffmpeg:
            item.add_marker(skip_ffmpeg)


# ---------------------------------------------------------------- frames sintéticos


@pytest.fixture()
def gradient_frame() -> npt.NDArray[np.uint8]:
    """Gradiente horizontal 0→255, 180×320 RGB. Estrés de cuantización/dithering."""
    row = np.linspace(0, 255, 320, dtype=np.uint8)
    return np.broadcast_to(row[None, :, None], (180, 320, 3)).copy()


@pytest.fixture()
def circle_frame() -> npt.NDArray[np.uint8]:
    """Círculo blanco sobre negro, 180×320. Valida corrección de aspecto (docs/06 §2 E2)."""
    yy, xx = np.mgrid[0:180, 0:320]
    mask = (yy - 90) ** 2 + (xx - 160) ** 2 <= 70**2
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[mask] = 255
    return frame


@pytest.fixture()
def noisy_static_frames() -> list[npt.NDArray[np.float32]]:
    """20 frames de luma estática + ruido gaussiano pequeño (semilla fija).

    Fixture de la propiedad de histéresis (docs/06 §2 E7): con σ < h/2 debe
    haber CERO cambios de carácter tras el frame 1.
    """
    rng = np.random.default_rng(42)
    base = np.full((45, 160), 0.5, dtype=np.float32)
    return [
        np.clip(base + rng.normal(0.0, 0.01, base.shape), 0, 1).astype(np.float32)
        for _ in range(20)
    ]


# ------------------------------------------------------------- clips vía ffmpeg lavfi


def _make_clip(dest: Path, vf_source: str, duration: float, with_audio: bool) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"{vf_source}=size=320x180:rate=30:duration={duration}",
    ]
    if with_audio:
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:a",
            "aac",
            "-shortest",
        ]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


@pytest.fixture(scope="session")
def clip_testsrc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Clip sintético de 2 s con audio (testsrc2): el input estándar de E1/E9."""
    return _make_clip(
        tmp_path_factory.mktemp("clips") / "testsrc.mp4", "testsrc2", 2.0, with_audio=True
    )


@pytest.fixture(scope="session")
def clip_silent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Clip sin stream de audio: el caso borde de extract_audio → None."""
    return _make_clip(
        tmp_path_factory.mktemp("clips") / "silent.mp4", "smptebars", 1.0, with_audio=False
    )
