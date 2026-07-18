"""Etapa 3 — saliencia (docs/04 §2, ADR-004). Se testea sin onnxruntime ni modelo
real: una sesión falsa devuelve una máscara sintética determinista y los helpers
numpy puros se verifican con casos chicos comprobables a mano."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kurai.ai.saliency import (
    SaliencyModel,
    SaliencyScheduler,
    _gaussian_blur,
    _resize_bilinear,
    load_saliency,
)


class FakeSession:
    """Sesión ONNX falsa: cuenta llamadas y devuelve siempre la misma máscara."""

    def __init__(self, mask: npt.NDArray[np.float32]) -> None:
        self._mask = mask
        self.run_count = 0

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        self.run_count += 1
        return [self._mask]


def _gradient_mask() -> npt.NDArray[np.float32]:
    """Gradiente horizontal 0→1 en layout U2Net (1, 1, 320, 320)."""
    row = np.linspace(0.0, 1.0, 320, dtype=np.float32)
    mask = np.broadcast_to(row[np.newaxis, :], (320, 320)).astype(np.float32)
    return mask.reshape(1, 1, 320, 320)


def _rgb_frame() -> npt.NDArray[np.uint8]:
    row = np.linspace(0, 255, 320, dtype=np.uint8)
    return np.broadcast_to(row[np.newaxis, :, np.newaxis], (180, 320, 3)).astype(np.uint8)


# --------------------------------------------------------------------- infer


def test_infer_shape_dtype_range() -> None:
    model = SaliencyModel(FakeSession(_gradient_mask()))
    density = model.infer(_rgb_frame(), rows=45, cols=160)
    assert density.shape == (45, 160)
    assert density.dtype == np.float32
    assert float(density.min()) >= 0.0
    assert float(density.max()) <= 1.0
    # El gradiente re-normalizado toca ambos extremos.
    assert float(density.min()) < 0.05
    assert float(density.max()) > 0.95


def test_infer_deterministic() -> None:
    frame = _rgb_frame()
    m1 = SaliencyModel(FakeSession(_gradient_mask())).infer(frame, 45, 160)
    m2 = SaliencyModel(FakeSession(_gradient_mask())).infer(frame, 45, 160)
    assert np.array_equal(m1, m2)


# --------------------------------------------------------- _resize_bilinear


def test_resize_constant_stays_constant() -> None:
    img = np.full((10, 10), 0.42, dtype=np.float32)
    out = _resize_bilinear(img, 25, 7)
    assert out.shape == (25, 7)
    assert np.allclose(out, 0.42, atol=1e-6)


def test_resize_identity_is_exact() -> None:
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = _resize_bilinear(img, 3, 4)
    assert np.array_equal(out, img)


def test_resize_preserves_horizontal_symmetry() -> None:
    row = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)  # simétrica
    img = np.broadcast_to(row[np.newaxis, :], (5, 5)).astype(np.float32)
    out = _resize_bilinear(img, 5, 9)
    assert out.shape == (5, 9)
    assert np.allclose(out, out[:, ::-1], atol=1e-6)


def test_resize_supports_rgb_channels() -> None:
    img = np.zeros((8, 8, 3), dtype=np.float32)
    out = _resize_bilinear(img, 320, 320)
    assert out.shape == (320, 320, 3)


# ---------------------------------------------------------- _gaussian_blur


def test_blur_constant_stays_constant() -> None:
    mask = np.full((16, 16), 0.7, dtype=np.float32)
    out = _gaussian_blur(mask, 2.0)
    assert np.allclose(out, 0.7, atol=1e-6)


def test_blur_softens_a_step() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[:, 16:] = 1.0
    out = _gaussian_blur(mask, 2.0)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    # La frontera dura (salto 1.0) pasa a un gradiente: gradiente máximo menor.
    orig_grad = float(np.abs(np.diff(mask, axis=1)).max())
    blur_grad = float(np.abs(np.diff(out, axis=1)).max())
    assert blur_grad < orig_grad
    # Aparecen valores intermedios que en el escalón no existían.
    assert bool(np.any((out > 0.05) & (out < 0.95)))


def test_blur_preserves_symmetry() -> None:
    mask = np.zeros((15, 15), dtype=np.float32)
    mask[7, 7] = 1.0  # impulso centrado (tamaño impar)
    out = _gaussian_blur(mask, 1.5)
    assert np.allclose(out, out[::-1, :], atol=1e-6)
    assert np.allclose(out, out[:, ::-1], atol=1e-6)
    assert np.allclose(out, out.T, atol=1e-6)


# ------------------------------------------------------------ scheduler


def test_scheduler_infers_on_multiples_and_reuses_between() -> None:
    session = FakeSession(_gradient_mask())
    sched = SaliencyScheduler(SaliencyModel(session))
    frame = _rgb_frame()
    for idx in range(6):  # 0..5
        sched.density_for(idx, frame, 45, 160, scene_cut=False)
    # Infiere en 0 y 5, reusa en 1-4.
    assert session.run_count == 2


def test_scheduler_reuse_returns_cached_map() -> None:
    session = FakeSession(_gradient_mask())
    sched = SaliencyScheduler(SaliencyModel(session))
    frame = _rgb_frame()
    d0 = sched.density_for(0, frame, 45, 160, scene_cut=False)
    d1 = sched.density_for(1, frame, 45, 160, scene_cut=False)
    assert session.run_count == 1
    assert np.array_equal(d0, d1)


def test_scheduler_scene_cut_forces_inference() -> None:
    session = FakeSession(_gradient_mask())
    sched = SaliencyScheduler(SaliencyModel(session))
    frame = _rgb_frame()
    sched.density_for(0, frame, 45, 160, scene_cut=False)  # infiere (1)
    sched.density_for(1, frame, 45, 160, scene_cut=False)  # reusa
    sched.density_for(3, frame, 45, 160, scene_cut=True)  # forzada (2)
    assert session.run_count == 2


def test_scheduler_degrades_to_ones_without_model() -> None:
    sched = SaliencyScheduler(None)
    density = sched.density_for(0, _rgb_frame(), 45, 160, scene_cut=False)
    assert density.shape == (45, 160)
    assert density.dtype == np.float32
    assert np.all(density == 1.0)


# ------------------------------------------------------------ load_saliency


def test_load_saliency_missing_file_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "no_existe.onnx"
    assert load_saliency(missing) is None
