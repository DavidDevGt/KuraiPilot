"""Proyección de la CharMatrix a ANSI para terminal (docs/01 §4, modo live).

La CharMatrix es el artefacto canónico; esto es una proyección más, igual que
el atlas (docs/03 §3). Mono emite un solo código de color por frame; fg emite
24-bit por celda con run-length por fila (un SGR solo cuando el color cambia:
en video los runs son largos y el stream se achica varias veces).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kurai.config import ColorMode
from kurai.types import CharMatrix

ESC = "\x1b"
HOME = f"{ESC}[H"
ENTER_ALT_SCREEN = f"{ESC}[?1049h{ESC}[?25l"  # alt buffer + ocultar cursor
EXIT_ALT_SCREEN = f"{ESC}[?1049l{ESC}[?25h{ESC}[0m"
MONO_SGR = f"{ESC}[38;2;102;255;102m"  # verde fósforo, mismo de atlas.MONO_COLOR


def charmatrix_to_ansi(cm: CharMatrix, ramp: str, color: ColorMode) -> str:
    """Frame ANSI completo: cursor home + contenido (se sobreescribe, sin clear
    por frame — evita el flash del borrado)."""
    lookup = np.array(list(ramp), dtype="<U1")
    chars = lookup[cm.char_idx]  # (rows, cols) de str

    if color is ColorMode.MONO:
        body = "\n".join("".join(row) for row in chars)
        return f"{HOME}{MONO_SGR}{body}"
    if color is ColorMode.FG:
        return _fg_run_length(cm, chars)
    raise NotImplementedError("Fase 2")  # fg+bg


def _fg_run_length(cm: CharMatrix, chars: npt.NDArray[np.str_]) -> str:
    """fg 24-bit con run-length por fila (los runs no cruzan saltos de línea)."""
    rows, cols = cm.shape
    lines: list[str] = []
    for r in range(rows):
        row_fg = cm.fg[r]
        row_ch = chars[r]
        change = np.ones(cols, dtype=bool)
        change[1:] = np.any(row_fg[1:] != row_fg[:-1], axis=1)
        bounds = np.append(np.flatnonzero(change), cols)
        parts: list[str] = []
        for s, e in zip(bounds[:-1], bounds[1:], strict=True):
            red, green, blue = (int(x) for x in row_fg[s])
            parts.append(f"{ESC}[38;2;{red};{green};{blue}m" + "".join(row_ch[s:e]))
        lines.append("".join(parts))
    return HOME + "\n".join(lines)
