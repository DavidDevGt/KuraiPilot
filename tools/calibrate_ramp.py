#!/usr/bin/env python3
"""Verifica la calibración de las rampas: la cobertura de tinta de cada glifo
debe ser estrictamente creciente a lo largo de la rampa (docs/02 E4).

El resultado ya está versionado en src/kurai/render/glyphs.py; este tool lo
audita e imprime la tabla. tests/test_render.py hace cumplir lo mismo en CI.

Uso: uv run python tools/calibrate_ramp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kurai.render.glyphs import GLYPH_H, GLYPH_W, RAMPS, ink_coverage  # noqa: E402


def main() -> int:
    total = GLYPH_H * GLYPH_W
    ok = True
    for ramp, chars in RAMPS.items():
        print(f"\nRampa '{ramp.value}' ({len(chars)} niveles):")
        prev = -1
        for ch in chars:
            cov = ink_coverage(ch)
            marker = ""
            if cov <= prev:
                marker = "  ✗ NO MONÓTONA"
                ok = False
            label = "espacio" if ch == " " else f"  '{ch}'  "
            print(f"  {label:>9}  {cov:3d}/{total} px  ({cov / total:5.1%}){marker}")
            prev = cov
    print("\n✓ Todas las rampas son monótonas" if ok else "\n✗ Calibración rota")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
