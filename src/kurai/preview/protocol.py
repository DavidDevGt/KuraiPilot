"""Protocolo del preview (docs/03 §3): la CharMatrix viaja al cliente, no
píxeles — el WebGL del navegador y el Renderer de export son proyecciones
del mismo artefacto.

Mensajes texto (JSON): meta inicial, estado, config del cliente.
Mensajes binarios (server→cliente): frames CharMatrix empaquetados.

Layout binario de un frame (little-endian):
  u32 frame_idx | u16 rows | u16 cols | u8 flags (bit0: has_bg)
  | char_idx (rows*cols u8) | fg (rows*cols*3 u8)
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from kurai.types import CharMatrix

_HEADER = struct.Struct("<IHHB")


def pack_frame(frame_idx: int, cm: CharMatrix) -> bytes:
    rows, cols = cm.shape
    header = _HEADER.pack(frame_idx, rows, cols, 0)
    return header + cm.char_idx.tobytes() + cm.fg.tobytes()


def unpack_frame(data: bytes) -> tuple[int, CharMatrix]:
    """Inversa de pack_frame — la usa el test de roundtrip, no el server."""
    frame_idx, rows, cols, _flags = _HEADER.unpack_from(data)
    off = _HEADER.size
    n = rows * cols
    char_idx = np.frombuffer(data, dtype=np.uint8, count=n, offset=off).reshape(rows, cols)
    fg = np.frombuffer(data, dtype=np.uint8, count=n * 3, offset=off + n).reshape(rows, cols, 3)
    return frame_idx, CharMatrix(char_idx=char_idx.copy(), fg=fg.copy())


def meta_message(
    rows: int,
    cols: int,
    fps: float,
    n_frames: int,
    ramp: str,
    atlas: npt.NDArray[np.uint8],  # (n_glyphs, GH, GW)
    color_mode: str,
) -> str:
    """Mensaje inicial: geometría + atlas de glifos embebido en base64
    (~1.7 KB para la rampa short; se envía una vez por cambio de config)."""
    n_glyphs, gh, gw = atlas.shape
    return json.dumps(
        {
            "type": "meta",
            "rows": rows,
            "cols": cols,
            "fps": fps,
            "n_frames": n_frames,
            "ramp": ramp,
            "color_mode": color_mode,
            "glyph_h": gh,
            "glyph_w": gw,
            "n_glyphs": n_glyphs,
            "atlas_b64": base64.b64encode(atlas.tobytes()).decode(),
        }
    )


def state_message(frame_idx: int, playing: bool) -> str:
    return json.dumps({"type": "state", "frame": frame_idx, "playing": playing})


def parse_client_message(raw: str) -> dict[str, Any]:
    """Valida el JSON del cliente; tipos soportados: config, seek, play, pause."""
    msg = json.loads(raw)
    if not isinstance(msg, dict) or msg.get("type") not in {"config", "seek", "play", "pause"}:
        raise ValueError(f"Mensaje de cliente inválido: {raw[:100]}")
    return msg
