"""Probe del entorno: degradaciones limpias, nunca excepciones (docs/02 §11)."""

from __future__ import annotations

import pytest

from kurai.probe import ProbeReport, probe


def test_probe_never_raises_and_finds_ffmpeg() -> None:
    r = probe()
    assert r.can_convert, f"ffmpeg debería existir en dev/CI: {r.errors}"
    assert r.ffmpeg_version


def test_ollama_down_is_warning_not_error() -> None:
    """ADR-005: Ollama es opcional — su ausencia jamás es un error."""
    r = probe(ollama_url="http://127.0.0.1:1")  # puerto imposible → connection refused
    assert not r.ollama_up
    assert not any("Ollama" in e for e in r.errors)
    assert any("Ollama" in w for w in r.warnings)


def test_disable_gpu_env_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KURAI_DISABLE_GPU", "1")
    r = probe(ollama_url="http://127.0.0.1:1")
    assert r.gpu_disabled_by_env
    assert r.gpu_name is None  # no consulta nvidia-smi con GPU deshabilitada


def test_missing_ffmpeg_is_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    r = probe(ollama_url="http://127.0.0.1:1")
    assert not r.can_convert
    assert any("apt install ffmpeg" in e for e in r.errors)


def test_hw_pipeline_requires_all_three() -> None:
    r = ProbeReport(hwaccel_cuda=True, nvenc_h264=True, nvdec_h264=False)
    assert not r.hw_pipeline
    r.nvdec_h264 = True
    assert r.hw_pipeline
