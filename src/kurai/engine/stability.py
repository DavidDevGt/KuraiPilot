"""Etapa 7 — Estabilidad temporal / anti-flicker (docs/02 E7).

Histéresis por celda contra luma_committed (la luma post-gamma del último
cambio, no la del frame anterior — evita drift acumulado). Una celda solo
cambia de carácter si |luma - committed| > h. Reset en corte de escena.
Vectorizada con máscaras (ADR-006). Métrica que la gobierna: FCR ≤ 0.05 en
zonas estáticas (docs/06 §3).

Nivel avanzado (Fase 2): el mapa committed se warpea con optical flow.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

HYSTERESIS_FACTOR = 0.6  # h = 0.6 × ancho del nivel de cuantización (docs/02 E7)

# Umbral de corte de escena sobre diferencia media de luma post-gamma entre
# frames consecutivos; PySceneDetect lo reemplaza en Fase 1 (docs/03 §1)
SCENE_CUT_THRESHOLD = 0.25


class HysteresisState:
    """Estado por job: luma y carácter comprometidos por celda."""

    def __init__(self, rows: int, cols: int) -> None:
        self.luma_committed: npt.NDArray[np.float32] = np.full((rows, cols), -1.0, dtype=np.float32)
        self.char_committed: npt.NDArray[np.uint8] = np.zeros((rows, cols), dtype=np.uint8)
        self._prev_mean_luma: float | None = None

    def reset(self) -> None:
        """Corte de escena: el siguiente frame se comporta como frame 0."""
        self.luma_committed.fill(-1.0)

    def detect_scene_cut(self, luma_gamma: npt.NDArray[np.float32]) -> bool:
        """Detector barato por diferencia global; resetea el estado si hay corte."""
        mean = float(luma_gamma.mean())
        prev = self._prev_mean_luma
        self._prev_mean_luma = mean
        if prev is not None and abs(mean - prev) > SCENE_CUT_THRESHOLD:
            self.reset()
            return True
        return False

    def apply(
        self,
        luma_gamma: npt.NDArray[np.float32],
        char_idx: npt.NDArray[np.uint8],
        levels: int,
    ) -> npt.NDArray[np.uint8]:
        """Devuelve char_idx estabilizado y actualiza el estado.

        Celdas nuevas (committed < 0) adoptan siempre; el resto solo si el
        movimiento de luma supera la histéresis h.
        """
        h = np.float32(HYSTERESIS_FACTOR / levels)
        fresh = self.luma_committed < 0.0
        moved = np.abs(luma_gamma - self.luma_committed) > h
        adopt = fresh | moved

        np.copyto(self.luma_committed, luma_gamma, where=adopt)
        np.copyto(self.char_committed, char_idx, where=adopt)
        return self.char_committed.copy()
