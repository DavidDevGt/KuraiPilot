"""E7 — histéresis: la propiedad FCR del gate de Fase 0 (docs/06 §3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.engine.mapping import apply_gamma, quantize
from kurai.engine.stability import HysteresisState


def _run_sequence(
    frames: list[npt.NDArray[np.float32]], levels: int = 10, gamma: float = 0.8
) -> list[npt.NDArray[np.uint8]]:
    rows, cols = frames[0].shape
    state = HysteresisState(rows, cols)
    out = []
    for luma in frames:
        lg = apply_gamma(luma, gamma)
        out.append(state.apply(lg, quantize(lg, levels), levels))
    return out


def test_fcr_zero_on_noisy_static(noisy_static_frames: list[npt.NDArray[np.float32]]) -> None:
    """El criterio del gate: ruido σ < h/2 sobre fondo estático produce CERO
    cambios de carácter tras el frame 1 (FCR = 0 ≤ 0.05, docs/07 Fase 0)."""
    results = _run_sequence(noisy_static_frames)
    changes = sum(
        int(np.count_nonzero(results[i] != results[i - 1])) for i in range(1, len(results))
    )
    assert changes == 0


def test_real_change_is_adopted() -> None:
    """La histéresis frena ruido, no contenido: un cambio real pasa."""
    dark = np.full((4, 4), 0.1, dtype=np.float32)
    bright = np.full((4, 4), 0.9, dtype=np.float32)
    results = _run_sequence([dark, dark, bright])
    assert np.array_equal(results[0], results[1])
    assert not np.array_equal(results[1], results[2])
    assert int(results[2].min()) > int(results[1].max())


def test_no_drift_accumulation() -> None:
    """committed es la luma del último CAMBIO, no del frame anterior: una
    rampa lenta bajo h por paso no arrastra el carácter (anti-drift)."""
    state = HysteresisState(1, 1)
    levels = 10
    h = 0.6 / levels
    start = 0.55  # centro del nivel 5: el cruce de nivel coincide con superar h
    luma = np.array([[start]], dtype=np.float32)
    first = state.apply(luma, quantize(luma, levels), levels).copy()
    # Pasos de h/3: contra el frame ANTERIOR nunca se supera h (impl. rota =
    # nunca cambia); contra el committed se supera al 4to paso → cambia ahí.
    changed_at = None
    for step in range(1, 7):
        luma = np.array([[start + step * h / 3]], dtype=np.float32)
        out = state.apply(luma, quantize(luma, levels), levels)
        if changed_at is None and not np.array_equal(out, first):
            changed_at = step
    assert changed_at is not None, "una rampa lenta debe terminar cambiando (anti-drift)"
    # Con float32 el paso 3 cae exactamente en h (borde); los pasos 1-2 están
    # claramente por debajo y NO deben cambiar. Una impl. rota que compara
    # contra el frame anterior nunca cambiaría (changed_at=None, ya cubierto).
    assert changed_at >= 3, "cambió antes de acumular h contra el committed"


def test_scene_cut_resets() -> None:
    state = HysteresisState(2, 2)
    dark = np.full((2, 2), 0.1, dtype=np.float32)
    bright = np.full((2, 2), 0.9, dtype=np.float32)
    # El pipeline llama detect en CADA frame (prima el estado del detector)
    assert not state.detect_scene_cut(dark)  # frame 0: sin referencia previa
    state.apply(dark, quantize(dark, 10), 10)
    assert state.detect_scene_cut(bright)  # salto global > umbral
    assert bool((state.luma_committed < 0).all())  # como frame 0 de nuevo


def test_no_cut_on_stable_content() -> None:
    state = HysteresisState(2, 2)
    luma = np.full((2, 2), 0.5, dtype=np.float32)
    assert not state.detect_scene_cut(luma)
    assert not state.detect_scene_cut(luma + 0.01)
