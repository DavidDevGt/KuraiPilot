"""Verificación del entorno de ejecución. Usado por `kurai doctor` y al inicio
de cada job (fail-fast con mensajes accionables, ADR-003 / docs/02 §11).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ProbeReport:
    ffmpeg_path: str | None = None
    ffmpeg_version: str | None = None
    hwaccel_cuda: bool = False
    nvenc_h264: bool = False
    nvdec_h264: bool = False
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    ollama_up: bool = False
    ollama_vision_model: str | None = None
    gpu_disabled_by_env: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def can_convert(self) -> bool:
        """Mínimo para un job: ffmpeg presente. Todo lo demás degrada."""
        return self.ffmpeg_path is not None

    @property
    def hw_pipeline(self) -> bool:
        """Camino NVDEC→NVENC completo disponible (target 4x de docs/05)."""
        return self.hwaccel_cuda and self.nvenc_h264 and self.nvdec_h264


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout


def probe(ollama_url: str | None = None) -> ProbeReport:
    r = ProbeReport()
    r.gpu_disabled_by_env = os.environ.get("KURAI_DISABLE_GPU") == "1"

    r.ffmpeg_path = shutil.which("ffmpeg")
    if r.ffmpeg_path is None:
        r.errors.append("ffmpeg no está en PATH — instalar: sudo apt install ffmpeg")
        return r
    if shutil.which("ffprobe") is None:
        r.errors.append("ffprobe no está en PATH (viene con ffmpeg)")

    first_line = _run(["ffmpeg", "-version"]).splitlines()
    r.ffmpeg_version = first_line[0].split()[2] if first_line else None
    r.hwaccel_cuda = "cuda" in _run(["ffmpeg", "-hide_banner", "-hwaccels"])
    r.nvenc_h264 = "h264_nvenc" in _run(["ffmpeg", "-hide_banner", "-encoders"])
    r.nvdec_h264 = "h264_cuvid" in _run(["ffmpeg", "-hide_banner", "-decoders"])

    if not r.gpu_disabled_by_env:
        if shutil.which("nvidia-smi"):
            out = _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ]
            ).strip()
            if out:
                name, total, free = (x.strip() for x in out.splitlines()[0].split(","))
                r.gpu_name, r.vram_total_mb, r.vram_free_mb = name, int(total), int(free)
        else:
            r.warnings.append("nvidia-smi no disponible: pipeline correrá en CPU")
        if not r.hw_pipeline:
            r.warnings.append("ffmpeg sin NVDEC/NVENC completo: decode/encode por software")

    # Ollama es opcional (ADR-005): su ausencia es informativa, nunca error
    url = ollama_url or os.environ.get("KURAI_OLLAMA_URL", "http://127.0.0.1:11434")
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2.0) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
        r.ollama_up = True
        vision = [m for m in models if "minicpm-v" in m]
        r.ollama_vision_model = vision[0] if vision else None
        if r.ollama_vision_model is None:
            r.warnings.append(
                "Ollama corre pero sin modelo de visión (minicpm-v*): Scene Analyst inactivo"
            )
    except (urllib.error.URLError, TimeoutError, OSError):
        r.warnings.append("Ollama no responde: Scene Analyst desactivado (opcional)")

    return r
