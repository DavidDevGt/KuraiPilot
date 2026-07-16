"""Etapa 5 nivel cnn — Clasificador de glifos (docs/04 §3). Fase 2.

CNN ~200k params, entrada parche 8×16 gris, softmax sobre glifos del atlas.
Solo celdas marcadas por Sobel (5-20%), en UN batch ONNX por frame.
Gate: mejora perceptible en subset textura Y ≤3 ms/frame p95, o SE ELIMINA
(la eliminación es resultado aceptable — docs/07 Fase 2).
Entrenamiento: training/ (dataset sintético auto-generado, corre en la 5070 Ti).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


class GlyphClassifier:
    def __init__(self, model_path: Path) -> None:
        raise NotImplementedError("Fase 2")

    def refine_batch(
        self,
        patches: npt.NDArray[np.float32],  # (n_cells, 16, 8)
    ) -> npt.NDArray[np.uint8]:
        """→ char_idx refinado por parche."""
        raise NotImplementedError("Fase 2")
