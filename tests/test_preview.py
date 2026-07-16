"""Fase 0.5 — los dos criterios del gate (docs/07) más el contrato del protocolo:

1. IGUALDAD: la CharMatrix del preview es bit a bit idéntica a la del export
   con la misma config — test, no promesa.
2. LATENCIA: un cambio de rampa/gamma/color recomputa en <100 ms.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="preview extra no instalado")

from fastapi.testclient import TestClient

from kurai.engine.dither import bayer_offsets
from kurai.engine.pipeline import cells_to_charmatrix
from kurai.engine.stability import HysteresisState
from kurai.preview.protocol import pack_frame, parse_client_message, unpack_frame
from kurai.preview.server import create_app
from kurai.preview.session import PreviewConfig, PreviewSession
from kurai.types import CharMatrix

pytestmark = pytest.mark.ffmpeg


# ------------------------------------------------------------ GATE 1: igualdad


def test_preview_charmatrix_identical_to_export_path(clip_testsrc: Path) -> None:
    """Todos los frames del preview == los del camino de export, bit a bit.

    El "camino de export" es exactamente lo que run_job hace por frame antes
    de componer/encodear: decode a grilla + cells_to_charmatrix con estado.
    """
    session = PreviewSession(clip_testsrc, PreviewConfig(cols=80))
    rows, cols = session.grid

    export_state = HysteresisState(rows, cols)
    export_offsets = bayer_offsets(rows, cols, session.levels)

    n = 0
    for cell_frame in session.frames(0):
        preview_cm = session.compute(cell_frame)
        export_cm = cells_to_charmatrix(
            cell_frame, export_state, export_offsets, session.levels, session.config.gamma
        )
        assert preview_cm.equals(export_cm), f"frame {n} difiere"
        n += 1
    assert n == session.meta.n_frames


# ------------------------------------------------------------ GATE 2: latencia


def test_config_change_under_100ms(clip_testsrc: Path) -> None:
    """Cambio de gamma+rampa con recompute del frame actual: <100 ms medidos
    (el presupuesto incluye el roundtrip WS real, que en localhost es ~1 ms;
    acá se mide el camino de cómputo, que es el que podría ser lento)."""
    session = PreviewSession(clip_testsrc, PreviewConfig(cols=200))  # peor caso: grilla máxima
    cell = next(iter(session.frames(0)))
    session.compute(cell)

    start = time.perf_counter()
    needs_redecode, recomputed = session.update_config(gamma=1.2, ramp="blocks")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert not needs_redecode
    assert recomputed is not None
    assert elapsed_ms < 100, f"{elapsed_ms:.1f} ms"


def test_cols_change_requires_redecode(clip_testsrc: Path) -> None:
    session = PreviewSession(clip_testsrc, PreviewConfig(cols=80))
    needs_redecode, recomputed = session.update_config(cols=120)
    assert needs_redecode and recomputed is None
    assert session.grid[1] == 120


def test_config_clamping(clip_testsrc: Path) -> None:
    """cols y gamma fuera de rango se clampean, no explotan (input de red)."""
    session = PreviewSession(clip_testsrc, PreviewConfig(cols=80))
    session.update_config(cols=10_000, gamma=99.0)
    assert session.config.cols == 200  # MAX_PREVIEW_COLS
    assert session.config.gamma == 2.0


def test_seek_resets_hysteresis(clip_testsrc: Path) -> None:
    """Un seek es un corte de escena: sin reset, el frame destino heredaría
    caracteres comprometidos de otra parte del video."""
    session = PreviewSession(clip_testsrc, PreviewConfig(cols=80))
    for i, cell in enumerate(session.frames(0)):
        session.compute(cell)
        if i >= 5:
            break
    session.seek(30)
    assert session.frame_idx == 30
    assert bool((session._state.luma_committed < 0).all())


def test_seek_clamps_to_video_bounds(clip_testsrc: Path) -> None:
    session = PreviewSession(clip_testsrc, PreviewConfig(cols=80))
    assert session.seek(-5) == 0
    assert session.seek(10_000) == session.meta.n_frames - 1


# ------------------------------------------------------------ protocolo


def test_frame_roundtrip() -> None:
    rng = np.random.default_rng(7)
    cm = CharMatrix(
        char_idx=rng.integers(0, 10, (45, 160), dtype=np.uint8),
        fg=rng.integers(0, 256, (45, 160, 3), dtype=np.uint8),
    )
    idx, decoded = unpack_frame(pack_frame(1234, cm))
    assert idx == 1234
    assert decoded.equals(cm)


def test_client_message_validation() -> None:
    assert parse_client_message('{"type": "play"}')["type"] == "play"
    with pytest.raises(ValueError):
        parse_client_message('{"type": "rm -rf"}')
    with pytest.raises(ValueError):
        parse_client_message('"no-un-dict"')


# ------------------------------------------------------------ WebSocket e2e


def test_websocket_full_flow(clip_testsrc: Path) -> None:
    """Conexión completa: meta → primer frame → cambio de config → frame nuevo
    + meta nueva. Por el TestClient de Starlette (server real, sin red)."""
    client = TestClient(create_app(clip_testsrc))

    resp = client.get("/")
    assert resp.status_code == 200 and "kurai preview" in resp.text

    with client.websocket_connect("/ws") as ws:
        meta = ws.receive_json()
        assert meta["type"] == "meta"
        assert meta["ramp"] == " .:-=+*#%@"
        assert len(meta["atlas_b64"]) > 0

        first = ws.receive_bytes()
        idx, cm = unpack_frame(first)
        assert idx == 0
        assert cm.shape == (meta["rows"], meta["cols"])

        state = ws.receive_json()
        assert state["type"] == "state" and state["playing"] is False

        # Cambio de rampa: meta nueva (atlas nuevo) + frame recomputado
        ws.send_json({"type": "config", "ramp": "blocks"})
        meta2 = ws.receive_json()
        assert meta2["ramp"] == " ░▒▓█"
        _, cm2 = unpack_frame(ws.receive_bytes())
        assert int(cm2.char_idx.max()) <= 4  # rampa de 5 niveles
        ws.receive_json()  # state

        # Seek: frame en la nueva posición
        ws.send_json({"type": "seek", "frame": 30})
        idx3, _ = unpack_frame(ws.receive_bytes())
        assert idx3 == 30


def test_websocket_play_streams_frames(clip_testsrc: Path) -> None:
    client = TestClient(create_app(clip_testsrc))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # meta
        ws.receive_bytes()  # primer frame
        ws.receive_json()  # state pausa

        ws.send_json({"type": "play"})
        ws.receive_json()  # state playing
        indices = []
        for _ in range(3):
            idx, _cm = unpack_frame(ws.receive_bytes())
            indices.append(idx)
        assert indices == sorted(indices) and len(set(indices)) == 3
