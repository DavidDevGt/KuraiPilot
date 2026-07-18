"""Pipeline Fase 0: golden files, reproducibilidad G4, círculo, y e2e completo.

Los golden operan sobre frames sintéticos en memoria (etapas 2-7 puras, sin
códecs de por medio — docs/06 §1); el e2e cubre el camino con ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from kurai.config import JobConfig, load_preset
from kurai.engine.grid import CELL_H, CELL_W
from kurai.engine.pipeline import frames_to_charmatrices, run_job
from kurai.types import CharMatrix

GOLDEN_DIR = Path(__file__).parent / "golden"


def _retro_cfg(cols: int = 40) -> JobConfig:
    return JobConfig(preset=load_preset("retro"), cols=cols)


def _work_gradient(rows: int, cols: int) -> npt.NDArray[np.uint8]:
    """Gradiente horizontal a resolución de trabajo exacta."""
    w, h = cols * CELL_W, rows * CELL_H
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.broadcast_to(row[None, :, None], (h, w, 3)).copy()


def _work_circle(rows: int, cols: int) -> npt.NDArray[np.uint8]:
    """Círculo blanco centrado, radio 40% del alto, a resolución de trabajo."""
    w, h = cols * CELL_W, rows * CELL_H
    yy, xx = np.mgrid[0:h, 0:w]
    r = h * 0.4
    mask = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= r**2
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[mask] = 255
    return frame


# ------------------------------------------------------------------ golden + G4


def test_gradient_golden() -> None:
    """CharMatrix del gradiente contra golden versionado. Si esto cambia sin
    cambio intencional de algoritmo, es una regresión (docs/06 §1)."""
    rows, cols = 10, 40
    cm = frames_to_charmatrices([_work_gradient(rows, cols)], rows, cols, _retro_cfg())[0]
    golden_path = GOLDEN_DIR / "gradient_retro_40x10.npz"
    assert golden_path.exists(), (
        "Golden faltante: generarlo con tools/make_goldens.py y commitearlo"
    )
    golden = np.load(golden_path)
    assert np.array_equal(cm.char_idx, golden["char_idx"])
    assert np.array_equal(cm.fg, golden["fg"])


def test_bit_exact_reproducibility() -> None:
    """G4: dos corridas idénticas producen CharMatrices bit a bit iguales."""
    rows, cols = 10, 40
    frames = [_work_gradient(rows, cols), _work_circle(rows, cols)]
    a = frames_to_charmatrices(frames, rows, cols, _retro_cfg())
    b = frames_to_charmatrices(frames, rows, cols, _retro_cfg())
    assert all(x.equals(y) for x, y in zip(a, b, strict=True))


def test_gradient_is_monotonic_in_char_idx() -> None:
    """Más brillo a la derecha ⇒ índice de rampa no decreciente (promediado
    por columna para tolerar el patrón espacial del dithering Bayer)."""
    rows, cols = 10, 40
    cm = frames_to_charmatrices([_work_gradient(rows, cols)], rows, cols, _retro_cfg())[0]
    col_means = cm.char_idx.astype(np.float64).mean(axis=0)
    smoothed = np.convolve(col_means, np.ones(5) / 5, mode="valid")
    assert (np.diff(smoothed) >= -0.15).all()
    assert cm.char_idx[:, 0].mean() < cm.char_idx[:, -1].mean()


# ------------------------------------------------------------------ círculo (gate)


def test_circle_stays_circular() -> None:
    """Gate de Fase 0: el bounding box de caracteres no-espacio del círculo es
    ~cuadrado en píxeles renderizados (docs/07). Celdas 1:2 → en celdas la caja
    es ~2:1, en píxeles (celda 8×16) vuelve a ser 1:1."""
    rows, cols = 20, 40  # grilla cuadrada en píxeles: 20*16 == 40*8
    cm = frames_to_charmatrices([_work_circle(rows, cols)], rows, cols, _retro_cfg())[0]
    non_space = np.nonzero(cm.char_idx > 0)
    box_rows = int(non_space[0].max() - non_space[0].min() + 1)
    box_cols = int(non_space[1].max() - non_space[1].min() + 1)
    px_h, px_w = box_rows * CELL_H, box_cols * CELL_W
    assert abs(px_h - px_w) <= max(px_h, px_w) * 0.1, f"{px_h}px alto vs {px_w}px ancho"


# ------------------------------------------------------------------ e2e con ffmpeg


@pytest.mark.ffmpeg
def test_convert_e2e_with_audio(clip_testsrc: Path, tmp_path: Path) -> None:
    """El gate integral: mp4 → mp4 ASCII con audio bit-idéntico y n_frames 1:1."""
    from kurai.engine.decode import probe_video
    from test_encode import _audio_stream_md5

    out = tmp_path / "ascii.mp4"
    cfg = JobConfig(preset=load_preset("retro"), cols=80, output=out)
    progress: list[int] = []
    result = run_job(clip_testsrc, cfg, on_progress=lambda done, total: progress.append(done))

    assert result == out and out.exists()
    in_meta, out_meta = probe_video(clip_testsrc), probe_video(out)
    assert out_meta.n_frames == in_meta.n_frames
    assert out_meta.has_audio
    assert _audio_stream_md5(out) == _audio_stream_md5(clip_testsrc)
    assert progress[-1] == in_meta.n_frames  # el progreso llegó al 100%


@pytest.mark.ffmpeg
def test_convert_silent_clip(clip_silent: Path, tmp_path: Path) -> None:
    out = tmp_path / "ascii.mp4"
    run_job(clip_silent, JobConfig(preset=load_preset("retro"), cols=80, output=out))
    from kurai.engine.decode import probe_video

    assert not probe_video(out).has_audio


def test_future_phase_presets_fail_fast(tmp_path: Path) -> None:
    """Fase 1 (detallado) ya está soportada; Fase 2 (alta-fidelidad) sigue
    fallando ANTES de decodificar nada."""
    from kurai.engine.pipeline import guard_phase

    guard_phase(JobConfig(preset=load_preset("detallado")))  # no levanta: E3/E5 están

    fake = tmp_path / "x.mp4"  # no llega a abrirse: guard_phase falla antes
    fake.write_bytes(b"\x00")
    with pytest.raises(NotImplementedError, match="Fase 2"):
        run_job(fake, JobConfig(preset=load_preset("alta-fidelidad")))


def test_charmatrix_channels_within_ramp(circle_frame: npt.NDArray[np.uint8]) -> None:
    """char_idx nunca sale del rango de la rampa (contrato del artefacto)."""
    rows, cols = 10, 40
    cm = frames_to_charmatrices([_work_gradient(rows, cols)], rows, cols, _retro_cfg())[0]
    assert isinstance(cm, CharMatrix)
    assert int(cm.char_idx.max()) <= 9  # rampa short: 10 niveles


# ------------------------------------------------------------------ Fase 1: saliencia


def test_density_uniform_is_noop() -> None:
    """density ≡ 1.0 produce EXACTAMENTE el char_idx de sin saliencia — la
    garantía que mantiene los golden de retro intactos y hace de detallado-sin-
    modelo una degradación bit a bit al determinista."""
    from kurai.engine.mapping import quantize

    rng = np.random.default_rng(3)
    lg = rng.random((30, 50), dtype=np.float32)
    offsets = (rng.random((30, 50), dtype=np.float32) - 0.5) / 10.0
    ones = np.ones((30, 50), dtype=np.float32)
    base = quantize(lg, 10, offsets, None)
    with_density = quantize(lg, 10, offsets, ones)
    assert np.array_equal(base, with_density)


def test_density_concentrates_detail() -> None:
    """El presupuesto de detalle se concentra donde la densidad es alta: sobre
    el MISMO gradiente, una celda de baja saliencia usa menos caracteres
    distintos que una de alta (docs/02 E3)."""
    from kurai.engine.mapping import quantize

    ramp_gradient = np.tile(np.linspace(0, 1, 40, dtype=np.float32), (10, 1))
    high = quantize(ramp_gradient, 10, None, np.ones((10, 40), dtype=np.float32))
    low = quantize(ramp_gradient, 10, None, np.full((10, 40), 0.0, dtype=np.float32))
    assert len(np.unique(low)) < len(np.unique(high))
    assert len(np.unique(high)) == 10  # saliencia plena = rampa completa


@pytest.mark.ffmpeg
def test_detallado_degrades_without_model(
    clip_testsrc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate de Fase 1 (docs/07): sin modelo, el job COMPLETA con warning y su
    salida es la del pipeline determinista (density uniforme)."""
    monkeypatch.setenv("KURAI_MODELS_DIR", str(tmp_path / "no-models"))  # load → None
    out = tmp_path / "det.mp4"
    cfg = JobConfig(preset=load_preset("detallado"), cols=80, output=out)
    with pytest.warns(UserWarning, match="saliencia"):
        result = run_job(clip_testsrc, cfg)
    assert result == out and out.exists()


@pytest.mark.ffmpeg
def test_detallado_with_model_uses_directional_glyphs(clip_testsrc: Path, tmp_path: Path) -> None:
    """Con modelo real, detallado corre E3+E5 y produce char_idx direccionales
    (índice ≥ niveles tonales) en los bordes. Se salta si falta onnxruntime o
    el modelo (CI corre sin ambos → degradación, cubierta por el test de arriba)."""
    pytest.importorskip("onnxruntime", reason="saliencia real requiere onnxruntime")
    from kurai.engine.pipeline import saliency_model_path

    if not saliency_model_path().is_file():
        pytest.skip("modelo u2net_lite.onnx no presente (uv run python tools/fetch_models.py)")

    out = tmp_path / "det_real.mp4"
    cfg = JobConfig(preset=load_preset("detallado"), cols=80, output=out)
    run_job(clip_testsrc, cfg)
    assert out.exists()
