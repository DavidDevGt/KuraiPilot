#!/usr/bin/env python3
"""(Re)genera los golden files de CharMatrix (docs/06 §1).

Correr SOLO tras un cambio intencional de algoritmo, y justificar el cambio
en el commit que actualiza tests/golden/ — un golden que cambia sin motivo
documentado es una regresión.

Uso: uv run python tools/make_goldens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from kurai.config import JobConfig, load_preset  # noqa: E402
from kurai.engine.pipeline import frames_to_charmatrices  # noqa: E402
from test_pipeline import _work_gradient  # noqa: E402

GOLDEN_DIR = REPO / "tests" / "golden"


def main() -> int:
    GOLDEN_DIR.mkdir(exist_ok=True)
    rows, cols = 10, 40
    cfg = JobConfig(preset=load_preset("retro"), cols=cols)
    cm = frames_to_charmatrices([_work_gradient(rows, cols)], rows, cols, cfg)[0]
    dest = GOLDEN_DIR / "gradient_retro_40x10.npz"
    np.savez_compressed(dest, char_idx=cm.char_idx, fg=cm.fg)
    print(f"✓ {dest.relative_to(REPO)} ({cm.shape[0]}×{cm.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
