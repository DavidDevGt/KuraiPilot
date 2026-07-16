"""Glifos bitmap 8×16 embebidos — la fuente de referencia del proyecto (docs/02 E8).

Los bitmaps viven en el repo (no en una fuente del sistema) porque son parte
del contrato de reproducibilidad G4: el mismo char_idx debe producir el mismo
píxel en cualquier máquina, hoy y en dos años. Cambiar un bitmap cambia los
golden files — requiere justificación (docs/06 §1).

La rampa `short` está ordenada por cobertura de tinta medida (píxeles
encendidos), no por intuición: tools/calibrate_ramp.py la verifica y
tests/test_render.py la hace cumplir (monotonía estricta).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.config import Ramp

GLYPH_W = 8
GLYPH_H = 16

# Cada glifo: 16 filas de 8 bits (MSB = columna izquierda). Diseño propio
# inspirado en fuentes bitmap clásicas 8×N, ajustado para cobertura monótona.
_BITMAPS: dict[str, list[int]] = {
    " ": [0x00] * 16,
    ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x18, 0x18, 0, 0, 0, 0],
    ":": [0, 0, 0, 0, 0, 0x18, 0x18, 0, 0, 0, 0x18, 0x18, 0, 0, 0, 0],
    "-": [0, 0, 0, 0, 0, 0, 0, 0x7E, 0x7E, 0, 0, 0, 0, 0, 0, 0],
    "=": [0, 0, 0, 0, 0, 0x7E, 0x7E, 0, 0, 0x7E, 0x7E, 0, 0, 0, 0, 0],
    "+": [0, 0, 0, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x7E, 0x18, 0x18, 0x18, 0x18, 0, 0, 0],
    "*": [0, 0, 0, 0, 0x18, 0xDB, 0x7E, 0x3C, 0x7E, 0xDB, 0x18, 0, 0, 0, 0, 0],
    "#": [0, 0, 0x24, 0x24, 0x24, 0xFF, 0x24, 0x24, 0x24, 0x24, 0xFF, 0x24, 0x24, 0x24, 0, 0],
    "%": [0, 0, 0, 0xC6, 0xC6, 0xCC, 0xCC, 0x18, 0x18, 0x30, 0x30, 0x66, 0x66, 0xC6, 0xC6, 0],
    "@": [0, 0, 0, 0x7C, 0x7C, 0xC6, 0xC6, 0xDE, 0xDE, 0xDE, 0xDE, 0xC0, 0x7C, 0x7C, 0, 0],
}

# Bloques Unicode: procedurales por densidad, patrón espacialmente estable.
_BLOCK_FILL = {" ": 0.0, "░": 0.25, "▒": 0.5, "▓": 0.75, "█": 1.0}

RAMPS: dict[Ramp, str] = {
    Ramp.SHORT: " .:-=+*#%@",
    Ramp.BLOCKS: " ░▒▓█",
    # Ramp.LONG llega con el preset alta-fidelidad (docs/07 Fase 2)
}


def _block_bitmap(fill: float) -> npt.NDArray[np.uint8]:
    if fill == 0.0:
        return np.zeros((GLYPH_H, GLYPH_W), dtype=np.uint8)
    if fill == 1.0:
        return np.full((GLYPH_H, GLYPH_W), 255, dtype=np.uint8)
    rr, cc = np.mgrid[0:GLYPH_H, 0:GLYPH_W]
    if fill == 0.5:
        mask = (rr + cc) % 2 == 0
    elif fill == 0.25:
        mask = (rr % 2 == 0) & ((cc + rr // 2) % 2 == 0)
    else:  # 0.75 = inverso del 0.25
        mask = ~((rr % 2 == 0) & ((cc + rr // 2) % 2 == 0))
    out = np.zeros((GLYPH_H, GLYPH_W), dtype=np.uint8)
    out[mask] = 255
    return out


def bitmap(char: str) -> npt.NDArray[np.uint8]:
    """Bitmap (GLYPH_H, GLYPH_W) uint8 {0, 255} del glifo."""
    rows = _BITMAPS.get(char)
    if rows is not None:
        bits = np.array(rows, dtype=np.uint8)
        return (np.unpackbits(bits[:, None], axis=1) * 255).astype(np.uint8)
    if char in _BLOCK_FILL:
        return _block_bitmap(_BLOCK_FILL[char])
    raise KeyError(f"Glifo '{char}' no está en la fuente embebida")


def ink_coverage(char: str) -> int:
    """Píxeles encendidos del glifo — la métrica que ordena las rampas."""
    return int(np.count_nonzero(bitmap(char)))


def ramp_chars(ramp: Ramp) -> str:
    chars = RAMPS.get(ramp)
    if chars is None:
        raise NotImplementedError("Fase 2")  # Ramp.LONG, docs/07
    return chars
