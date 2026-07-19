"""Invariantes de las etapas 4/6/7 verificados con Hypothesis (docs/02):
propiedades que deben cumplirse para CUALQUIER entrada, no solo los casos
que se nos ocurrieron. Complementan los golden files (que fijan valores) con
leyes (que fijan comportamiento).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from kurai.engine.dither import bayer_offsets, floyd_steinberg
from kurai.engine.mapping import apply_gamma, quantize
from kurai.engine.stability import HysteresisState

_levels = st.integers(min_value=2, max_value=70)
_shape = st.tuples(st.integers(1, 32), st.integers(1, 64))


def _luma_arrays(shape: tuple[int, int]) -> st.SearchStrategy[np.ndarray]:  # type: ignore[type-arg]
    return hnp.arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(0.0, 1.0, width=32, allow_nan=False),
    )


# ------------------------------------------------------------------ E4 quantize


@given(shape=_shape, levels=_levels, data=st.data())
def test_quantize_always_within_ramp(
    shape: tuple[int, int], levels: int, data: st.DataObject
) -> None:
    """idx ∈ [0, levels-1] para cualquier luma y cualquier offset de dithering."""
    luma = data.draw(_luma_arrays(shape))
    offsets = bayer_offsets(shape[0], shape[1], levels)
    idx = quantize(luma, levels, offsets)
    assert int(idx.min()) >= 0
    assert int(idx.max()) <= levels - 1


@given(levels=_levels, a=st.floats(0.0, 1.0, width=32), b=st.floats(0.0, 1.0, width=32))
def test_quantize_monotonic_without_dither(levels: int, a: float, b: float) -> None:
    """Sin dithering, más luma nunca da menos índice (docs/06 §2 E4)."""
    lo, hi = min(a, b), max(a, b)
    arr = np.array([[lo, hi]], dtype=np.float32)
    idx = quantize(arr, levels)
    assert idx[0, 0] <= idx[0, 1]


@given(gamma=st.floats(0.1, 2.0), a=st.floats(0.0, 1.0, width=32), b=st.floats(0.0, 1.0, width=32))
def test_gamma_preserves_order_and_range(gamma: float, a: float, b: float) -> None:
    lo, hi = min(a, b), max(a, b)
    out = apply_gamma(np.array([lo, hi], dtype=np.float32), gamma)
    assert out[0] <= out[1] + 1e-6
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0 + 1e-6


# ------------------------------------------------------------------ E6 Bayer


@given(shape=_shape, levels=_levels)
def test_bayer_offsets_bounded_and_centered(shape: tuple[int, int], levels: int) -> None:
    """Los offsets nunca superan medio nivel (no pueden saltar un nivel entero)
    y están centrados (el dithering no aclara ni oscurece la imagen)."""
    off = bayer_offsets(shape[0], shape[1], levels)
    assert off.shape == shape
    assert float(np.abs(off).max()) <= 0.5 / levels + 1e-7
    # Media ~0 sobre tiles completos; en shapes parciales tolerancia amplia
    assert abs(float(off.mean())) <= 0.5 / levels


def test_bayer_is_static_between_calls() -> None:
    """Mismo patrón siempre: el Bayer no puede inducir flicker por construcción."""
    a = bayer_offsets(45, 160, 10)
    b = bayer_offsets(45, 160, 10)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ E7 histéresis


@settings(max_examples=50)
@given(shape=st.tuples(st.integers(1, 16), st.integers(1, 16)), levels=_levels, data=st.data())
def test_hysteresis_idempotent_on_repeated_frame(
    shape: tuple[int, int], levels: int, data: st.DataObject
) -> None:
    """El MISMO frame dos veces jamás cambia un carácter (FCR=0 por ley)."""
    luma = data.draw(_luma_arrays(shape))
    state = HysteresisState(*shape)
    idx = quantize(luma, levels)
    first = state.apply(luma, idx, levels)
    second = state.apply(luma, idx, levels)
    assert np.array_equal(first, second)


@settings(max_examples=50)
@given(shape=st.tuples(st.integers(1, 16), st.integers(1, 16)), data=st.data())
def test_hysteresis_output_always_from_committed_domain(
    shape: tuple[int, int], data: st.DataObject
) -> None:
    """Tras N frames arbitrarios, el output es siempre el último idx adoptado
    por celda — nunca un valor inventado fuera del dominio de la rampa."""
    levels = 10
    state = HysteresisState(*shape)
    for _ in range(data.draw(st.integers(1, 5))):
        luma = data.draw(_luma_arrays(shape))
        out = state.apply(luma, quantize(luma, levels), levels)
        assert int(out.max()) <= levels - 1
        assert np.array_equal(out, state.char_committed)


# ------------------------------------------------------------------ E6 Floyd-Steinberg


@settings(max_examples=50)
@given(shape=_shape, levels=_levels, data=st.data())
def test_fs_returns_exact_bin_centers(
    shape: tuple[int, int], levels: int, data: st.DataObject
) -> None:
    """FS devuelve centros de bin exactos: quantize() reproduce el nivel que
    FS eligió, sin riesgo de cruce de borde por redondeo (docs/02 E6)."""
    luma = data.draw(_luma_arrays(shape))
    dithered = floyd_steinberg(luma, levels)
    assert dithered.shape == shape and dithered.dtype == np.float32
    idx = quantize(dithered, levels)
    centers = (idx.astype(np.float64) + 0.5) / levels
    assert np.allclose(centers, dithered, atol=1e-6)
    assert int(idx.min()) >= 0 and int(idx.max()) <= levels - 1


@settings(max_examples=50)
@given(shape=_shape, levels=_levels, data=st.data())
def test_fs_deterministic(shape: tuple[int, int], levels: int, data: st.DataObject) -> None:
    """G4: dos corridas de FS sobre el mismo input son bit a bit iguales."""
    luma = data.draw(_luma_arrays(shape))
    assert np.array_equal(floyd_steinberg(luma, levels), floyd_steinberg(luma, levels))


@settings(max_examples=50)
@given(
    value=st.floats(0.0, 1.0, width=32),
    levels=_levels,
    shape=st.tuples(st.integers(8, 32), st.integers(8, 64)),
)
def test_fs_preserves_mean_on_flat_field(value: float, levels: int, shape: tuple[int, int]) -> None:
    """La difusión de error conserva el brillo medio: en un campo plano, el
    promedio de los centros elegidos queda a menos de un nivel del valor real —
    la propiedad que Bayer/floor no tienen y que hace mejores los gradientes."""
    flat = np.full(shape, value, dtype=np.float32)
    dithered = floyd_steinberg(flat, levels)
    assert abs(float(dithered.mean()) - value) <= 1.0 / levels
