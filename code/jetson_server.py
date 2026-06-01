#!/usr/bin/env python3
"""
jetson_server.py — headless tracking server that runs on the Jetson.

It runs the TrackerEngine in a worker thread and serves a single command-center
client over TCP.  Per connected client it:

  * streams annotated JPEG frames        (MSG_FRAME)
  * streams telemetry JSON each frame     (MSG_TELEMETRY)
  * receives + applies operator commands  (MSG_COMMAND)

Designed to run over an SSH tunnel.  On your laptop:

    ssh -N -L 8000:localhost:8000 user@jetson      # in one terminal
    python3 command_center.py --host localhost --port 8000

On the Jetson:

    python3 jetson_server.py --source jetson --engine nano_best.engine

Because the tunnel terminates on localhost, bind to 127.0.0.1 by default — the
stream never touches the network unencrypted.  Use --bind 0.0.0.0 only if you
deliberately want a raw LAN socket with no SSH.

Frame pacing: the newest annotated frame is always what gets sent; if the
client (or WiFi) is slow, intermediate frames are dropped rather than queued,
so end-to-end latency stays bounded instead of growing without limit.
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time

import cv2
import numpy as np

import protocol as proto
from tracker_engine import TrackerEngine


# ── shared latest-frame slot (newest wins, old frames dropped) ───────────────

class LatestSlot:
    """Single-slot mailbox: producer overwrites, consumer takes newest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item = None          # (jpeg_bytes, telemetry_dict)
        self._seq = 0
        self._cv = threading.Condition(self._lock)

    def put(self, item) -> None:
        with self._cv:
            self._item = item
            self._seq += 1
            self._cv.notify()

    def get(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than last_seq is available."""
        with self._cv:
            if self._seq <= last_seq:
                self._cv.wait(timeout)
            if self._seq <= last_seq:
                return None, last_seq
            return self._item, self._seq


# ── engine worker: produce annotated frames + telemetry continuously ─────────

class EngineWorker(threading.Thread):
    def __init__(self, engine: TrackerEngine, slot: LatestSlot,
                 jpeg_quality: int = 80) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.slot = slot
        self.jpeg_quality = jpeg_quality
        self._stop_evt = threading.Event()
        self._rec_writer = None
        self._rec_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_evt.set()

    def set_jpeg_quality(self, q: int) -> None:
        self.jpeg_quality = max(1, min(100, int(q)))

    # server-side raw recording (annotated frames) ---------------------------

    def start_recording(self, path: str, w: int, h: int, fps: float) -> str:
        with self._rec_lock:
            if self._rec_writer is not None:
                return "already recording"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._rec_writer = cv2.VideoWriter(
                path, fourcc, max(1.0, fps), (w, h))
            return f"recording -> {path}"

    def stop_recording(self) -> str:
        with self._rec_lock:
            if self._rec_writer is None:
                return "not recording"
            self._rec_writer.release()
            self._rec_writer = None
            return "recording stopped"

    def run(self) -> None:
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while not self._stop_evt.is_set() and self.engine.running:
            frame, telem = self.engine.process_next()
            if frame is None:
                break
            with self._rec_lock:
                if self._rec_writer is not None:
                    self._rec_writer.write(frame)
            enc[1] = self.jpeg_quality
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if not ok:
                continue
            self.slot.put((buf.tobytes(), telem))
        self.engine.close()
        with self._rec_lock:
            if self._rec_writer is not None:
                self._rec_writer.release()
                self._rec_writer = None


# ── per-client handling ──────────────────────────────────────────────────────

def handle_client(conn: socket.socket, addr, engine: TrackerEngine,
                  worker: EngineWorker, slot: LatestSlot, caps: dict) -> None:
    print(f"[server] client connected: {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Handshake: tell the client the stream's capabilities.
    try:
        proto.send_json(conn, proto.MSG_HELLO, caps)
    except OSError:
        return

    stop_rx = threading.Event()

    def rx_loop() -> None:
        """Receive + apply operator commands until the client disconnects."""
        try:
            while not stop_rx.is_set():
                msg = proto.recv_message(conn)
                if msg is None:
                    break
                mtype, payload = msg
                if mtype != proto.MSG_COMMAND:
                    continue
                cmd = proto.recv_json(payload)
                # A single malformed/unknown command must not tear down the
                # connection (e.g. a version mismatch between ends). Log and
                # keep serving.
                try:
                    _apply_command(cmd, engine, worker, caps)
                except Exception as exc:
                    print(f"[server] command '{cmd.get('cmd')}' failed: {exc}")
        except (OSError, ValueError) as exc:
            print(f"[server] rx loop ended: {exc}")
        finally:
            stop_rx.set()

    rx = threading.Thread(target=rx_loop, daemon=True)
    rx.start()

    last_seq = 0
    try:
        while not stop_rx.is_set() and engine.running:
            item, last_seq = slot.get(last_seq, timeout=1.0)
            if item is None:
                continue
            jpeg, telem = item
            # Telemetry first (cheap), then the frame it describes.
            proto.send_json(conn, proto.MSG_TELEMETRY, telem)
            proto.send_message(conn, proto.MSG_FRAME, jpeg)
    except OSError as exc:
        print(f"[server] tx loop ended: {exc}")
    finally:
        stop_rx.set()
        try:
            conn.close()
        except OSError:
            pass
        print(f"[server] client disconnected: {addr}")


def _apply_command(cmd: dict, engine: TrackerEngine,
                   worker: EngineWorker, caps: dict) -> None:
    c = cmd.get("cmd")
    if c == proto.CMD_SELECT_TARGET:
        engine.select_target(int(cmd["track_id"]))
    elif c == proto.CMD_RESET_TARGET:
        engine.reset_target()
    elif c == proto.CMD_SET_SOURCE:
        engine.set_source(cmd["source"])
    elif c == proto.CMD_PAUSE:
        engine.set_paused(bool(cmd.get("value", True)))
    elif c == getattr(proto, "CMD_SET_MASKS", "set_masks"):
        engine.set_masks(cmd.get("masks", []))
    elif c == proto.CMD_SET_PARAM:
        name, value = cmd["name"], cmd["value"]
        if name == "jpeg_quality":
            worker.set_jpeg_quality(int(value))
        else:
            engine.set_param(name, value)
    elif c == proto.CMD_START_REC:
        os.makedirs("recordings", exist_ok=True)
        path = cmd.get("path") or time.strftime(
            "recordings/jetson_%Y%m%d_%H%M%S.mp4")
        msg = worker.start_recording(path, caps["width"], caps["height"],
                                     caps.get("fps", 30.0))
        print(f"[server] {msg}")
    elif c == proto.CMD_STOP_REC:
        print(f"[server] {worker.stop_recording()}")
    elif c == proto.CMD_SHUTDOWN:
        engine.running = False
    else:
        print(f"[server] unknown command: {c}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jetson headless tracking server")
    p.add_argument("--engine",   default="nano_best.engine")
    p.add_argument("--backbone", default="nanotrack_backbone_sim.onnx")
    p.add_argument("--neckhead", default="nanotrack_head_sim.onnx")
    p.add_argument("--source",   default="jetson",
                   help="webcam index, 'jetson', 'jetson60', or a video file")
    p.add_argument("--bind", default="127.0.0.1",
                   help="bind address (default 127.0.0.1 for SSH tunnel; "
                        "use 0.0.0.0 for raw LAN)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--jpeg-quality", type=int, default=80)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine = TrackerEngine(args.engine, args.backbone, args.neckhead,
                           source=args.source, draw_hud_on_frame=False)
    caps = engine.open()
    caps["fps"] = 30.0
    print(f"[server] engine ready: {caps}")

    slot = LatestSlot()
    worker = EngineWorker(engine, slot, jpeg_quality=args.jpeg_quality)
    worker.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"[server] listening on {args.bind}:{args.port} "
          f"(serving one client at a time)")

    try:
        while engine.running:
            srv.settimeout(1.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            # One client at a time keeps the Jetson's uplink predictable.
            handle_client(conn, addr, engine, worker, slot, caps)
    except KeyboardInterrupt:
        print("\n[server] interrupted")
    finally:
        worker.stop()
        engine.running = False
        worker.join(timeout=2.0)
        srv.close()
        print("[server] shut down")


if __name__ == "__main__":
    main()
