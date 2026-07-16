"""Configuración y presets. Los presets viven en presets/*.toml (docs/02 §10);
este módulo los valida — un preset inválido falla al cargar, no a mitad de un export.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"


class Ramp(StrEnum):
    SHORT = "short"
    LONG = "long"
    BLOCKS = "blocks"


class RefineMode(StrEnum):
    OFF = "off"
    EDGES = "edges"
    EDGES_CNN = "edges+cnn"


class DitherMode(StrEnum):
    BAYER = "bayer"
    FLOYD_STEINBERG = "fs"


class FlickerMode(StrEnum):
    HYSTERESIS = "hysteresis"
    HYSTERESIS_FLOW = "hysteresis+flow"


class ColorMode(StrEnum):
    MONO = "mono"
    FG = "fg"
    FG_BG = "fg+bg"


class Preset(BaseModel):
    """Un preset mapea 1:1 a la tabla de docs/02 §10."""

    name: str
    saliency: bool = False
    refine: RefineMode = RefineMode.OFF
    dither: DitherMode = DitherMode.BAYER
    flicker: FlickerMode = FlickerMode.HYSTERESIS
    color: ColorMode = ColorMode.MONO
    ramp: Ramp = Ramp.SHORT
    gamma: float = Field(default=0.8, gt=0.0, le=2.0)


class JobConfig(BaseModel):
    """Config completa de un job de conversión (preset + overrides de CLI)."""

    preset: Preset
    cols: int = Field(default=160, ge=20, le=600)
    output: Path | None = None
    auto_scene: bool = False  # --auto: Scene Analyst ajusta preset por escena (ADR-005)


def load_preset(name: str, presets_dir: Path = PRESETS_DIR) -> Preset:
    path = presets_dir / f"{name}.toml"
    if not path.exists():
        available = sorted(p.stem for p in presets_dir.glob("*.toml"))
        raise FileNotFoundError(f"Preset '{name}' no existe. Disponibles: {available}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Preset(name=name, **data)
