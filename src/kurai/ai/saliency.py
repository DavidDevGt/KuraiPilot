"""Etapa 3 — Saliencia con U2Net-lite ONNX (docs/04 §2, ADR-004). Fase 1.

Entrada RGB 320×320 con normalización ImageNet; inferencia cada N=5 frames con
propagación del mapa entre corridas (la saliencia cambia lento); inferencia
forzada en corte de escena. Post-proceso: resize a la grilla + blur gaussiano
σ=2 celdas + normalización a [0,1] — una frontera dura de densidad se ve peor
que no tener saliencia (docs/04 §2).

Degradación limpia (docs/02 E3 "Apagado", regla de gobernanza ai/): si el modelo
no carga (archivo ausente u onnxruntime no importable) el scheduler devuelve
`density_map ≡ 1.0` y el pipeline queda idéntico a no tener la etapa; nunca aborta.

`onnxruntime` se importa de forma perezosa dentro de `load_saliency` para que el
resto del sistema (y los tests) no dependan de la librería ni del modelo real.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

INFER_EVERY_N = 5
INPUT_SIZE = 320
BLUR_SIGMA_CELLS = 2.0

# Normalización ImageNet en orden RGB (docs/04 §2).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_INPUT_NAME = "input"


class _Session(Protocol):
    """Interfaz mínima de una sesión ONNX Runtime (evita depender del tipo de ort).

    La firma coincide estructuralmente con `onnxruntime.InferenceSession.run`.
    """

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]: ...


# ------------------------------------------------------------- helpers numpy puros


def _resize_bilinear(
    img: npt.NDArray[np.float32], out_h: int, out_w: int
) -> npt.NDArray[np.float32]:
    """Resize bilineal con convención de centro de píxel (como INTER_LINEAR).

    Puro NumPy, determinista. Soporta `(H, W)` y `(H, W, C)`. Una imagen
    constante se re-muestrea a la misma constante; el caso identidad copia.
    """
    in_h, in_w = int(img.shape[0]), int(img.shape[1])
    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32, copy=True)

    src = img.astype(np.float32, copy=False)

    # Coordenadas fuente con medio-píxel de centro; se clampan al borde.
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    ys = np.clip(ys, 0.0, in_h - 1)
    xs = np.clip(xs, 0.0, in_w - 1)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)

    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)

    top_left = src[y0][:, x0]
    top_right = src[y0][:, x1]
    bot_left = src[y1][:, x0]
    bot_right = src[y1][:, x1]

    extra = (1,) * (src.ndim - 2)
    wy_b = wy.reshape((out_h, 1, *extra))
    wx_b = wx.reshape((1, out_w, *extra))

    top = top_left * (1.0 - wx_b) + top_right * wx_b
    bottom = bot_left * (1.0 - wx_b) + bot_right * wx_b
    out = top * (1.0 - wy_b) + bottom * wy_b
    result: npt.NDArray[np.float32] = np.asarray(out, dtype=np.float32)
    return result


def _blur_axis0(a: npt.NDArray[np.float32], sigma: float) -> npt.NDArray[np.float32]:
    """Convolución gaussiana 1D a lo largo del eje 0, padding reflect.

    El radio se clampa a `n-1` para que el reflect sea válido en grillas chicas.
    """
    n = int(a.shape[0])
    if n < 2 or sigma <= 0:
        return a.astype(np.float32, copy=True)

    radius = max(1, min(int(math.ceil(3.0 * sigma)), n - 1))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.sum()

    padded = np.pad(a.astype(np.float32, copy=False), ((radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(a, dtype=np.float32)
    for k in range(kernel.shape[0]):
        out += kernel[k] * padded[k : k + n]
    return out


def _gaussian_blur(mask: npt.NDArray[np.float32], sigma: float) -> npt.NDArray[np.float32]:
    """Blur gaussiano separable (σ en celdas) con padding reflect (docs/04 §2).

    Kernel separable: se convoluciona por filas y luego por columnas. Una máscara
    constante se preserva; una frontera dura se suaviza a un gradiente.
    """
    if sigma <= 0:
        return mask.astype(np.float32, copy=True)
    blurred = _blur_axis0(mask, sigma)
    blurred = _blur_axis0(blurred.T, sigma).T
    return np.ascontiguousarray(blurred, dtype=np.float32)


def _normalize01(a: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normaliza a [0,1] por min-max. Máscara plana → se clampa a [0,1]."""
    lo = float(a.min())
    hi = float(a.max())
    span = hi - lo
    if span < 1e-6:
        return np.clip(a, 0.0, 1.0).astype(np.float32)
    return ((a - lo) / span).astype(np.float32)


def _preprocess(frame_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """RGB `(H,W,3)` → blob NCHW `(1,3,320,320)` float32, norm ImageNet."""
    scaled = frame_rgb.astype(np.float32) / 255.0
    resized = _resize_bilinear(scaled, INPUT_SIZE, INPUT_SIZE)
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    blob = chw[np.newaxis, ...]
    return np.ascontiguousarray(blob, dtype=np.float32)


def _postprocess(
    raw_mask: npt.NDArray[np.float32], rows: int, cols: int
) -> npt.NDArray[np.float32]:
    """Máscara del modelo → `density_map (rows,cols)` en [0,1] (resize+blur+norm)."""
    mask = np.asarray(raw_mask, dtype=np.float32)
    if mask.ndim > 2:
        # Dimensiones batch/canal son 1 en U2Net: se colapsan a (H, W).
        mask = mask.reshape(int(mask.shape[-2]), int(mask.shape[-1]))
    resized = _resize_bilinear(mask, rows, cols)
    blurred = _gaussian_blur(resized, BLUR_SIGMA_CELLS)
    return _normalize01(blurred)


# ----------------------------------------------------------------- modelo y scheduler


class SaliencyModel:
    """U2Net-lite envuelto sobre una sesión ONNX inyectable (`_Session`)."""

    def __init__(self, session: _Session, input_name: str = _DEFAULT_INPUT_NAME) -> None:
        self._session = session
        self._input_name = input_name

    def infer(
        self, frame_rgb: npt.NDArray[np.uint8], rows: int, cols: int
    ) -> npt.NDArray[np.float32]:
        """→ `density_map (rows, cols)` en [0,1], ya re-muestreado y blureado."""
        blob = _preprocess(frame_rgb)
        outputs = self._session.run(None, {self._input_name: blob})
        return _postprocess(outputs[0], rows, cols)


def load_saliency(model_path: Path) -> SaliencyModel | None:
    """Carga el modelo de saliencia; degrada a `None` sin lanzar (docs/02 §11).

    Devuelve `None` si el archivo no existe, si onnxruntime no se puede importar,
    o si la sesión falla al crearse. El caller degrada con un warning y usa
    `density_map ≡ 1.0`. Import perezoso de onnxruntime: solo se paga acá.
    """
    if not model_path.is_file():
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    try:
        # Solo pedimos los providers realmente instalados (CUDA si está, si no
        # CPU): pedir CUDAExecutionProvider sin la wheel GPU emite un warning
        # ruidoso por cada job. CPU alcanza para la corrección y el A/B.
        available = set(ort.get_available_providers())
        wanted = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        providers = wanted or ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = str(session.get_inputs()[0].name)
    except Exception:  # noqa: BLE001 — degradación: cualquier fallo de carga → sin saliencia
        return None

    return SaliencyModel(session, input_name)


class SaliencyScheduler:
    """Amortiza la inferencia cada N=5 frames y propaga el mapa (docs/04 §2).

    Con `model is None` degrada a `density_map ≡ 1.0` (pipeline sin saliencia).
    Con modelo: infiere en múltiplos de N o en corte de escena; entre corridas
    reusa el último mapa cacheado.
    """

    def __init__(self, model: SaliencyModel | None) -> None:
        self._model = model
        self._cache: npt.NDArray[np.float32] | None = None
        self._cache_shape: tuple[int, int] | None = None

    def density_for(
        self,
        frame_idx: int,
        frame_rgb: npt.NDArray[np.uint8],
        rows: int,
        cols: int,
        scene_cut: bool,
    ) -> npt.NDArray[np.float32]:
        """`density_map (rows, cols)` para el frame, infiriendo o propagando."""
        if self._model is None:
            return np.ones((rows, cols), dtype=np.float32)

        cache = self._cache
        needs_infer = (
            scene_cut
            or cache is None
            or self._cache_shape != (rows, cols)
            or frame_idx % INFER_EVERY_N == 0
        )
        if needs_infer:
            density = self._model.infer(frame_rgb, rows, cols)
            self._cache = density
            self._cache_shape = (rows, cols)
            return density.copy()

        # cache no es None: `needs_infer` contempla `cache is None`.
        assert cache is not None
        return cache.copy()
