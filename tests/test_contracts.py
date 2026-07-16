"""Contratos del esqueleto: los stubs pendientes declaran una fase válida del
roadmap (docs/07) y los fixtures sintéticos del conftest cumplen su spec.

Cuando una fase se implementa, sus stubs desaparecen y este test se encoge solo.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

SRC = Path(__file__).resolve().parents[1] / "src" / "kurai"
VALID_PHASES = {"0", "0.5", "1", "2", "3"}
STUB_RE = re.compile(r'NotImplementedError\("Fase ([^"]+)"\)')


def test_all_stubs_reference_valid_roadmap_phase() -> None:
    found = 0
    for py in SRC.rglob("*.py"):
        for match in STUB_RE.finditer(py.read_text()):
            found += 1
            assert match.group(1) in VALID_PHASES, (
                f"{py.relative_to(SRC)}: fase '{match.group(1)}' no existe en docs/07"
            )
    assert found > 0, "No quedan stubs: eliminar este test o actualizar el roadmap"


def test_gradient_fixture_spans_full_range(gradient_frame: npt.NDArray[np.uint8]) -> None:
    assert gradient_frame.shape == (180, 320, 3)
    assert gradient_frame.min() == 0 and gradient_frame.max() == 255


def test_circle_fixture_is_pixel_circular(circle_frame: npt.NDArray[np.uint8]) -> None:
    """El bounding box del círculo es cuadrado EN PÍXELES; tras la grilla 1:2
    deberá seguir siéndolo en celdas — esa es la prueba de E2 (docs/06 §2)."""
    ys, xs = np.nonzero(circle_frame[:, :, 0])
    box_h, box_w = ys.max() - ys.min(), xs.max() - xs.min()
    assert abs(int(box_h) - int(box_w)) <= 1


def test_noisy_static_fixture_noise_is_small(
    noisy_static_frames: list[npt.NDArray[np.float32]],
) -> None:
    stack = np.stack(noisy_static_frames)
    assert stack.shape == (20, 45, 160)
    assert float(stack.std(axis=0).max()) < 0.05  # σ mucho menor que h/2 típico
