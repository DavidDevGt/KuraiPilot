"""Presets y configuración: validación al cargar, coincidencia con docs/02 §10."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kurai.config import (
    ColorMode,
    DitherMode,
    FlickerMode,
    Preset,
    Ramp,
    RefineMode,
    load_preset,
)

# La tabla de docs/02 §10, como dato. Si un preset TOML se desvía de la spec,
# este test falla — mantener tabla y TOMLs en sync es deliberadamente manual.
SPEC_TABLE = {
    "retro": dict(
        saliency=False,
        refine=RefineMode.OFF,
        dither=DitherMode.BAYER,
        flicker=FlickerMode.HYSTERESIS,
        color=ColorMode.MONO,
        ramp=Ramp.SHORT,
    ),
    "detallado": dict(
        saliency=True,
        refine=RefineMode.EDGES,
        dither=DitherMode.BAYER,
        flicker=FlickerMode.HYSTERESIS,
        color=ColorMode.FG,
        ramp=Ramp.SHORT,
    ),
    "alta-fidelidad": dict(
        saliency=True,
        refine=RefineMode.EDGES_CNN,
        dither=DitherMode.FLOYD_STEINBERG,
        flicker=FlickerMode.HYSTERESIS_FLOW,
        color=ColorMode.FG_BG,
        ramp=Ramp.LONG,
    ),
}


@pytest.mark.parametrize("name", sorted(SPEC_TABLE))
def test_preset_matches_spec_table(name: str) -> None:
    preset = load_preset(name)
    for field, expected in SPEC_TABLE[name].items():
        assert getattr(preset, field) == expected, f"{name}.{field}"


def test_all_shipped_presets_are_in_spec_table() -> None:
    """Un preset TOML nuevo exige actualizar docs/02 §10 y esta tabla primero."""
    from kurai.config import PRESETS_DIR

    shipped = {p.stem for p in PRESETS_DIR.glob("*.toml")}
    assert shipped == set(SPEC_TABLE)


def test_unknown_preset_lists_available() -> None:
    with pytest.raises(FileNotFoundError, match="retro"):
        load_preset("inexistente")


def test_invalid_preset_field_fails_at_load(tmp_path: Path) -> None:
    """Un preset inválido falla al cargar, no a mitad de un export (docs/02)."""
    bad = tmp_path / "roto.toml"
    bad.write_text('dither = "magico"\n')
    with pytest.raises(ValidationError):
        load_preset("roto", presets_dir=tmp_path)


def test_gamma_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        Preset(name="x", gamma=0.0)
    with pytest.raises(ValidationError):
        Preset(name="x", gamma=2.5)
