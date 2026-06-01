"""
Wire protocol shared by the Jetson server and the laptop command center.

A single TCP connection carries two directions of traffic:

  Jetson -> laptop : MSG_FRAME  (annotated JPEG) and MSG_TELEMETRY (JSON stats)
  laptop -> Jetson : MSG_COMMAND (JSON control messages)

Every message on the wire is framed as:

    | 1 byte  msg_type | 4 bytes big-endian payload length | payload bytes |

JSON payloads are UTF-8 encoded; frame payloads are raw JPEG bytes.  This
length-prefixed framing means we never have to guess message boundaries on a
stream socket, and it works identically over a raw LAN socket or an SSH
tunnel (the tunnel is transparent to us).
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Any

# ── Message type tags ──────────────────────────────────────────────────────
MSG_TELEMETRY = 0x01   # Jetson -> laptop : JSON dict of run stats
MSG_FRAME     = 0x02   # Jetson -> laptop : JPEG bytes (annotated frame)
MSG_COMMAND   = 0x03   # laptop -> Jetson : JSON dict {"cmd": ..., ...}
MSG_HELLO     = 0x04   # either way       : JSON handshake / capability info
MSG_BYE       = 0x05   # either way       : graceful shutdown notice

_HEADER = struct.Struct(">BI")   # 1-byte type + 4-byte length
MAX_PAYLOAD = 32 * 1024 * 1024   # 32 MB hard ceiling, guards against junk


# ── Low-level send / recv ───────────────────────────────────────────────────

def send_message(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    """Frame and send one message. Raises on socket error."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} bytes")
    sock.sendall(_HEADER.pack(msg_type, len(payload)) + payload)


def send_json(sock: socket.socket, msg_type: int, obj: Any) -> None:
    send_message(sock, msg_type, json.dumps(obj).encode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes, or return None if the peer closed the socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket) -> tuple[int, bytes] | None:
    """
    Receive one framed message. Returns (msg_type, payload) or None if the
    connection was closed cleanly.
    """
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    msg_type, length = _HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ValueError(f"declared payload too large: {length} bytes")
    payload = _recv_exactly(sock, length) if length else b""
    if payload is None:
        return None
    return msg_type, payload


def recv_json(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


# ── Command vocabulary (laptop -> Jetson) ───────────────────────────────────
# Kept here so both ends agree on the strings without magic literals.

CMD_SELECT_TARGET = "select_target"   # {"cmd":..., "track_id": int}
CMD_RESET_TARGET  = "reset_target"    # {"cmd":...}
CMD_SET_SOURCE    = "set_source"      # {"cmd":..., "source": str}  (restarts capture)
CMD_SET_PARAM     = "set_param"       # {"cmd":..., "name": str, "value": float|bool}
CMD_START_REC     = "start_record"    # {"cmd":...}  (server-side raw record, optional)
CMD_STOP_REC      = "stop_record"     # {"cmd":...}
CMD_PAUSE         = "pause"           # {"cmd":..., "value": bool}
CMD_SET_MASKS     = "set_masks"       # {"cmd":..., "masks": [[fx1,fy1,fx2,fy2],...]}
CMD_SHUTDOWN      = "shutdown"        # {"cmd":...}  (stop the run)

# Tunable parameter names accepted by CMD_SET_PARAM.
TUNABLE_PARAMS = (
    "conf_thres",            # detection confidence threshold
    "latency_skip_ms",       # adaptive-skip latency threshold (ms); 0 disables
    "predict_on_skip",       # bool
    "auto_switch_on_loss",   # bool
    "jpeg_quality",          # 1..100, stream quality
)
