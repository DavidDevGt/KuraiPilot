"""Preview server (docs/03 §3): FastAPI + WebSocket, bind SOLO a 127.0.0.1.

Una task por WebSocket maneja la sesión: bombea frames a ritmo de fps con
asyncio y atiende comandos del cliente (config/seek/play/pause). El decode
bloqueante corre vía asyncio.to_thread para no frenar el event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kurai.preview.protocol import (
    meta_message,
    pack_frame,
    parse_client_message,
    state_message,
)
from kurai.preview.session import PreviewSession

STATIC_DIR = Path(__file__).parent / "static"


def create_app(input_file: Path) -> FastAPI:
    app = FastAPI(title="kurai preview", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        session = PreviewSession(input_file)
        await _send_meta(ws, session)
        with contextlib.suppress(WebSocketDisconnect):
            await _session_loop(ws, session)

    return app


async def _send_meta(ws: WebSocket, session: PreviewSession) -> None:
    rows, cols = session.grid
    await ws.send_text(
        meta_message(
            rows,
            cols,
            session.meta.fps,
            session.meta.n_frames,
            session.ramp_str,
            session.atlas,
            session.config.color.value,
        )
    )


class _FrameSource:
    """Iterador de cell-frames con reinicio (seek / cambio de grilla)."""

    def __init__(self, session: PreviewSession) -> None:
        self.session = session
        self._it: Iterator[npt.NDArray[np.uint8]] | None = None

    def restart(self) -> None:
        self.close()

    async def next_frame(self) -> npt.NDArray[np.uint8] | None:
        if self._it is None:
            self._it = self.session.frames(self.session.frame_idx)
        return await asyncio.to_thread(next, self._it, None)

    def close(self) -> None:
        it = self._it
        self._it = None
        if it is not None:
            with contextlib.suppress(Exception):
                it.close()  # type: ignore[attr-defined]  # generador de iter_frames


async def _session_loop(ws: WebSocket, session: PreviewSession) -> None:
    source = _FrameSource(session)
    frame_period = 1.0 / session.meta.fps
    try:
        # Primer frame inmediato, en pausa: el usuario ve algo al conectar
        cell = await source.next_frame()
        if cell is not None:
            await ws.send_bytes(pack_frame(session.frame_idx, session.compute(cell)))
        await ws.send_text(state_message(session.frame_idx, session.playing))

        while True:
            timeout = frame_period if session.playing else None
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=timeout)
            except TimeoutError:
                await _advance(ws, session, source)
                continue
            await _handle_message(ws, session, source, raw)
    finally:
        source.close()


async def _advance(ws: WebSocket, session: PreviewSession, source: _FrameSource) -> None:
    cell = await source.next_frame()
    if cell is None:  # fin del video: pausa en el último frame
        session.playing = False
        await ws.send_text(state_message(session.frame_idx, session.playing))
        return
    session.frame_idx += 1
    await ws.send_bytes(pack_frame(session.frame_idx, session.compute(cell)))


async def _handle_message(
    ws: WebSocket, session: PreviewSession, source: _FrameSource, raw: str
) -> None:
    msg = parse_client_message(raw)
    kind = msg["type"]

    if kind == "play":
        session.playing = True
    elif kind == "pause":
        session.playing = False
    elif kind == "seek":
        session.seek(int(msg.get("frame", 0)))
        source.restart()
        cell = await source.next_frame()
        if cell is not None:
            await ws.send_bytes(pack_frame(session.frame_idx, session.compute(cell)))
    elif kind == "config":
        needs_redecode, recomputed = session.update_config(
            **{k: v for k, v in msg.items() if k != "type"}
        )
        await _send_meta(ws, session)  # rampa/atlas/grilla pueden haber cambiado
        if needs_redecode:
            source.restart()
            cell = await source.next_frame()
            if cell is not None:
                await ws.send_bytes(pack_frame(session.frame_idx, session.compute(cell)))
        elif recomputed is not None:
            # El camino del gate: recompute desde el cell-frame cacheado, <100 ms
            await ws.send_bytes(pack_frame(session.frame_idx, recomputed))
    await ws.send_text(state_message(session.frame_idx, session.playing))


def serve(input_file: Path, port: int = 8420) -> None:
    """Arranca uvicorn en 127.0.0.1 (bloqueante). El CLI abre el navegador."""
    import uvicorn

    uvicorn.run(create_app(input_file), host="127.0.0.1", port=port, log_level="warning")
