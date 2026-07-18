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

from kurai.engine.decode import DecodeError
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
        try:
            session = PreviewSession(input_file)
        except DecodeError:
            # El CLI valida antes de servir; esto cubre el archivo que se
            # corrompió/borró con el server ya corriendo.
            await ws.close(code=1011, reason="el input no es un video legible")
            return
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
    """Bombea frames con deadline ABSOLUTO (due = anterior + período, como
    live.py) y recibe comandos con una task persistente que nunca se cancela
    por timeout: sin carrera de wait_for (mensajes que se pierden al expirar
    el timer) y sin starvation (un drag de slider no congela el playback)."""
    source = _FrameSource(session)
    frame_period = 1.0 / session.meta.fps
    loop = asyncio.get_running_loop()
    recv: asyncio.Task[str] | None = None
    next_due: float | None = None
    try:
        # Primer frame inmediato, en pausa: el usuario ve algo al conectar
        cell = await source.next_frame()
        if cell is not None:
            await ws.send_bytes(pack_frame(session.frame_idx, session.compute(cell)))
        await ws.send_text(state_message(session.frame_idx, session.playing))

        while True:
            if recv is None:
                recv = asyncio.create_task(ws.receive_text())
            if session.playing:
                now = loop.time()
                if next_due is None:
                    next_due = now
                timeout: float | None = max(0.0, next_due - now)
            else:
                next_due = None
                timeout = None
            done, _ = await asyncio.wait({recv}, timeout=timeout)

            if recv in done:
                raw = recv.result()  # WebSocketDisconnect propaga al endpoint
                recv = None
                try:
                    await _handle_message(ws, session, source, raw)
                except ValueError:
                    # Payload malformado (input de red): se ignora y el cliente
                    # sigue vivo; el state lo deja resincronizado.
                    await ws.send_text(state_message(session.frame_idx, session.playing))
                continue

            # Venció el deadline: siguiente frame, y el próximo due se ancla al
            # anterior (no a "ahora") para no acumular drift por compute/send.
            await _advance(ws, session, source)
            if next_due is not None:
                next_due += frame_period
                if next_due < loop.time():  # más lentos que el video: sin deuda
                    next_due = loop.time()
    finally:
        if recv is not None:
            recv.cancel()
            with contextlib.suppress(Exception):
                await recv
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
