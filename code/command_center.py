#!/usr/bin/env python3
"""
command_center.py — laptop-side ground-control GUI for the drone tracker.

Connects to jetson_server.py over TCP (typically through an SSH tunnel),
renders the live annotated video, shows live telemetry (FPS, detection
latency, resolution, state, track count), lets the operator click a detected
track to lock the laser target, exposes runtime tunables, and logs the run
(telemetry CSV/JSONL + optional MP4 of the received video).

Run:
    python3 command_center.py --host localhost --port 8000

Dependencies (laptop only):
    pip install PyQt6 opencv-python numpy

A network thread owns the socket and emits Qt signals; all widget updates
happen on the GUI thread, so the UI stays responsive even if the link stalls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from collections import deque

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint
from PyQt6.QtGui import (QImage, QPixmap, QFont, QColor, QPalette, QIcon,
                        QPainter, QPen)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSlider, QCheckBox, QComboBox, QLineEdit, QPlainTextEdit,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QFrame,
)

# ─────────────────────── inlined wire protocol ──────────────────────────────
# (Previously protocol.py — folded in so this GUI is a single standalone file.
#  The server side uses the same definitions.)
import struct as _struct
from types import SimpleNamespace as _NS


def _proto_send_message(sock, msg_type, payload):
    if len(payload) > _PROTO_MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} bytes")
    sock.sendall(_PROTO_HEADER.pack(msg_type, len(payload)) + payload)


def _proto_send_json(sock, msg_type, obj):
    _proto_send_message(sock, msg_type, json.dumps(obj).encode("utf-8"))


def _proto_recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _proto_recv_message(sock):
    header = _proto_recv_exactly(sock, _PROTO_HEADER.size)
    if header is None:
        return None
    msg_type, length = _PROTO_HEADER.unpack(header)
    if length > _PROTO_MAX_PAYLOAD:
        raise ValueError(f"declared payload too large: {length} bytes")
    payload = _proto_recv_exactly(sock, length) if length else b""
    if payload is None:
        return None
    return msg_type, payload


def _proto_recv_json(payload):
    return json.loads(payload.decode("utf-8"))


_PROTO_HEADER = _struct.Struct(">BI")
_PROTO_MAX_PAYLOAD = 32 * 1024 * 1024

# Namespace mirroring the old `import protocol as proto`, so every existing
# `proto.X` call site keeps working unchanged.
proto = _NS(
    MSG_TELEMETRY=0x01,
    MSG_FRAME=0x02,
    MSG_COMMAND=0x03,
    MSG_HELLO=0x04,
    MSG_BYE=0x05,
    CMD_SELECT_TARGET="select_target",
    CMD_RESET_TARGET="reset_target",
    CMD_SET_SOURCE="set_source",
    CMD_SET_PARAM="set_param",
    CMD_START_REC="start_record",
    CMD_STOP_REC="stop_record",
    CMD_PAUSE="pause",
    CMD_SHUTDOWN="shutdown",
    CMD_SET_MASKS="set_masks",
    send_message=_proto_send_message,
    send_json=_proto_send_json,
    recv_message=_proto_recv_message,
    recv_json=_proto_recv_json,
)


# ───────────────────────────── aesthetic ────────────────────────────────────
# Industrial mission-control: near-black panels, mono telemetry, amber accent
# for "armed/active", cyan for data, red for lost/alarm.

# Monochrome + single champagne-gold accent. State is conveyed by fill vs
# outline vs dim, not by hue, so the palette stays black / white / gold only.
GOLD     = "#E6C97A"   # champagne gold — the one accent
GOLD_DIM = "#8A7B4B"   # muted gold for secondary accents
WHITE    = "#F2F2F0"   # primary text
BG       = "#000000"   # window — true black
PANEL    = "#0B0B0B"   # panel fill
PANEL_HI = "#161616"   # raised
EDGE     = "#2E2E2A"   # borders (warm grey)
TEXT     = WHITE
MUTED    = "#8C8C86"   # dim warm grey

# Semantic aliases kept so existing call sites need no renaming.
ACCENT = GOLD          # active / armed / selected
DATA   = WHITE         # data readouts (white digits on black)
GOOD   = GOLD          # tracking / nominal  -> gold
WARN   = GOLD          # coasting / caution  -> gold (distinguished by style)
BAD    = WHITE         # lost / alarm        -> dim white (no red available)

STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT};
    font-family: 'DejaVu Sans Mono','Consolas','Menlo',monospace; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 6px;
    margin-top: 14px; padding: 10px 10px 8px 10px; font-size: 11px;
    letter-spacing: 2px; color: {MUTED}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px;
    text-transform: uppercase; }}
QLabel {{ color: {TEXT}; }}
QPushButton {{ background: {PANEL_HI}; border: 1px solid {EDGE};
    border-radius: 4px; padding: 7px 12px; color: {TEXT}; font-size: 12px;
    letter-spacing: 1px; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: {BG}; }}
QPushButton#danger:hover {{ border-color: {BAD}; color: {BAD}; }}
QPushButton#primary {{ border-color: {ACCENT}; color: {ACCENT}; }}
QTableWidget {{ background: {PANEL}; gridline-color: {EDGE};
    border: 1px solid {EDGE}; border-radius: 4px; font-size: 12px;
    selection-background-color: {ACCENT}; selection-color: {BG}; }}
QHeaderView::section {{ background: {PANEL_HI}; color: {MUTED};
    border: none; border-bottom: 1px solid {EDGE}; padding: 5px;
    font-size: 10px; letter-spacing: 1px; }}
QPlainTextEdit {{ background: {PANEL}; border: 1px solid {EDGE};
    border-radius: 4px; font-size: 11px; color: {MUTED}; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {PANEL_HI};
    border: 1px solid {EDGE}; border-radius: 4px; padding: 5px; color: {TEXT}; }}
QComboBox:hover {{ border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {EDGE}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px;
    margin: -6px 0; border-radius: 7px; }}
QCheckBox {{ spacing: 8px; font-size: 12px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {EDGE};
    border-radius: 3px; background: {PANEL_HI}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
"""


# ───────────────────────────── network thread ───────────────────────────────

class NetworkThread(QThread):
    """Owns the socket. Emits frames, telemetry, status, and the hello caps."""
    frame_received     = pyqtSignal(np.ndarray)
    telemetry_received = pyqtSignal(dict)
    hello_received     = pyqtSignal(dict)
    status_changed     = pyqtSignal(str, str)   # (message, color)

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._running = True
        self._send_lock = None  # set in run() (threading import kept local)

    def run(self) -> None:
        import threading
        self._send_lock = threading.Lock()
        while self._running:
            try:
                self.status_changed.emit(
                    f"connecting to {self.host}:{self.port} ...", WARN)
                s = socket.create_connection((self.host, self.port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                self.status_changed.emit(
                    f"LINK UP — {self.host}:{self.port}", GOOD)
                self._rx_loop(s)
            except OSError as exc:
                self.status_changed.emit(f"link down: {exc} — retrying", BAD)
                self._sock = None
                if self._running:
                    self.msleep(1500)

    def _rx_loop(self, s: socket.socket) -> None:
        pending_telem: dict | None = None
        while self._running:
            msg = proto.recv_message(s)
            if msg is None:
                self.status_changed.emit("peer closed connection", BAD)
                return
            mtype, payload = msg
            if mtype == proto.MSG_HELLO:
                self.hello_received.emit(proto.recv_json(payload))
            elif mtype == proto.MSG_TELEMETRY:
                pending_telem = proto.recv_json(payload)
                self.telemetry_received.emit(pending_telem)
            elif mtype == proto.MSG_FRAME:
                arr = np.frombuffer(payload, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    self.frame_received.emit(img)

    def send_command(self, cmd: dict) -> None:
        s = self._sock
        if s is None:
            return
        try:
            with self._send_lock:
                proto.send_json(s, proto.MSG_COMMAND, cmd)
        except OSError as exc:
            self.status_changed.emit(f"command failed: {exc}", BAD)

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass


# ───────────────────────────── small widgets ────────────────────────────────

class Readout(QFrame):
    """A labelled telemetry value with a unit, big mono digits."""

    def __init__(self, label: str, unit: str = "") -> None:
        super().__init__()
        self.setStyleSheet(
            f"QFrame{{background:{PANEL_HI};border:1px solid {EDGE};"
            f"border-radius:5px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(1)
        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            f"color:{MUTED};font-size:10px;letter-spacing:2px;")
        self._val = QLabel("--")
        f = QFont("DejaVu Sans Mono")
        f.setPointSize(20)
        f.setBold(True)
        self._val.setFont(f)
        self._val.setStyleSheet(f"color:{DATA};")
        self._unit = unit
        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def set(self, value, color: str = DATA) -> None:
        txt = f"{value}{(' ' + self._unit) if self._unit else ''}"
        self._val.setText(txt)
        self._val.setStyleSheet(f"color:{color};")


class StatePill(QLabel):
    """
    Large state indicator. With a black/white/gold palette only, state is
    shown by treatment rather than colour:
        TRACKING  -> solid gold fill, black text   (armed, locked on)
        COASTING  -> gold outline, gold text        (bridging, caution)
        LOST      -> dim white outline, muted text   (no target)
    """

    def __init__(self) -> None:
        super().__init__("— — —")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont("DejaVu Sans Mono")
        f.setPointSize(15)
        f.setBold(True)
        self.setFont(f)
        self.set_state("LOST")

    def set_state(self, state: str) -> None:
        self.setText(state)
        if state == "TRACKING":
            css = (f"color:{BG};background:{GOLD};border:2px solid {GOLD};")
        elif state == "COASTING":
            css = (f"color:{GOLD};background:{PANEL_HI};"
                   f"border:2px solid {GOLD};")
        else:  # LOST or unknown
            css = (f"color:{MUTED};background:{PANEL_HI};"
                   f"border:2px solid {MUTED};")
        self.setStyleSheet(css + "border-radius:6px;padding:10px;"
                                 "letter-spacing:3px;")


class VideoCanvas(QWidget):
    """
    Displays the live frame (letterboxed, aspect-preserved) and lets the
    operator draw reject masks over fixed-background false positives:

      * left-drag           -> draw a new mask rectangle
      * right-click a mask  -> delete it
      * masks are stored as NORMALISED fractions of the frame (0..1) so they
        are resolution-independent and map cleanly to the engine's frame.

    Emits masks_changed(list) whenever the mask set changes, carrying
    [(fx1, fy1, fx2, fy2), ...] for the main window to push to the engine.
    Masks are per-scene / in-memory only (cleared on demand).
    """
    masks_changed = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(880, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:#000;border:1px solid {EDGE};"
                           f"border-radius:6px;")
        self.setMouseTracking(True)
        self._pix: QPixmap | None = None
        self._masks: list[tuple] = []        # normalised (fx1,fy1,fx2,fy2)
        self._drag_start: QPoint | None = None
        self._drag_now: QPoint | None = None
        # The rect (in widget pixels) the frame is actually drawn into.
        self._img_rect = QRect()

    # ── frame input ──────────────────────────────────────────────────────────

    def set_frame(self, pix: QPixmap) -> None:
        self._pix = pix
        self.update()

    def masks(self) -> list[tuple]:
        return list(self._masks)

    def clear_masks(self) -> None:
        if self._masks:
            self._masks.clear()
            self.masks_changed.emit([])
            self.update()

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _compute_img_rect(self) -> QRect:
        """Where the aspect-preserved frame sits inside this widget."""
        if self._pix is None or self._pix.isNull():
            return QRect()
        ww, wh = self.width(), self.height()
        pw, ph = self._pix.width(), self._pix.height()
        if pw == 0 or ph == 0:
            return QRect()
        scale = min(ww / pw, wh / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        return QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def _widget_to_frac(self, p: QPoint):
        """Map a widget-pixel point to a (fx, fy) fraction of the frame."""
        r = self._img_rect
        if r.width() == 0 or r.height() == 0:
            return None
        fx = (p.x() - r.x()) / r.width()
        fy = (p.y() - r.y()) / r.height()
        return (min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0))

    def _frac_to_widget_rect(self, m: tuple) -> QRect:
        r = self._img_rect
        x1 = r.x() + m[0] * r.width()
        y1 = r.y() + m[1] * r.height()
        x2 = r.x() + m[2] * r.width()
        y2 = r.y() + m[3] * r.height()
        return QRect(int(min(x1, x2)), int(min(y1, y2)),
                     int(abs(x2 - x1)), int(abs(y2 - y1)))

    # ── mouse interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, e) -> None:
        if self._pix is None:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.pos()
            self._drag_now = e.pos()
        elif e.button() == Qt.MouseButton.RightButton:
            # Delete a mask under the cursor.
            frac = self._widget_to_frac(e.pos())
            if frac is not None:
                for i, m in enumerate(self._masks):
                    if (min(m[0], m[2]) <= frac[0] <= max(m[0], m[2]) and
                            min(m[1], m[3]) <= frac[1] <= max(m[1], m[3])):
                        del self._masks[i]
                        self.masks_changed.emit(list(self._masks))
                        self.update()
                        break

    def mouseMoveEvent(self, e) -> None:
        if self._drag_start is not None:
            self._drag_now = e.pos()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if (e.button() == Qt.MouseButton.LeftButton
                and self._drag_start is not None):
            a = self._widget_to_frac(self._drag_start)
            b = self._widget_to_frac(e.pos())
            self._drag_start = None
            self._drag_now = None
            if a and b:
                fx1, fy1 = min(a[0], b[0]), min(a[1], b[1])
                fx2, fy2 = max(a[0], b[0]), max(a[1], b[1])
                # Ignore accidental tiny drags (a click).
                if (fx2 - fx1) > 0.01 and (fy2 - fy1) > 0.01:
                    self._masks.append((fx1, fy1, fx2, fy2))
                    self.masks_changed.emit(list(self._masks))
            self.update()

    # ── painting ───────────────────────────────────────────────────────────

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))
        if self._pix is None or self._pix.isNull():
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "AWAITING VIDEO LINK")
            p.end()
            return
        self._img_rect = self._compute_img_rect()
        p.drawPixmap(self._img_rect, self._pix)

        # Existing masks: dim gold outline + faint fill.
        pen = QPen(QColor(GOLD)); pen.setWidth(2)
        p.setPen(pen)
        for m in self._masks:
            wr = self._frac_to_widget_rect(m)
            p.fillRect(wr, QColor(230, 201, 122, 40))
            p.drawRect(wr)
            p.drawText(wr.x() + 3, wr.y() + 14, "MASK")

        # In-progress drag rectangle.
        if self._drag_start is not None and self._drag_now is not None:
            pen2 = QPen(QColor(WHITE)); pen2.setWidth(1)
            pen2.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen2)
            p.drawRect(QRect(self._drag_start, self._drag_now).normalized())
        p.end()


# ───────────────────────────── main window ──────────────────────────────────

class CommandCenter(QMainWindow):
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.setWindowTitle("ARGUS Control")
        # In-app window/taskbar icon (PPM). Loaded if present beside the app.
        _icon = QIcon("hyperion_mark.ppm")
        if not _icon.isNull():
            self.setWindowIcon(_icon)
        self.resize(1480, 900)
        self.setStyleSheet(STYLE)

        self._caps: dict = {}
        self._last_telem: dict = {}
        self._frame_size = (0, 0)
        # Stream-health tracking
        self._link_times = deque(maxlen=30)   # arrival timestamps of frames
        self._last_frame_no = None
        self._dropped_total = 0

        # Logging state
        self._logging = False
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None
        self._video_writer = None
        self._log_dir = None
        # Standalone laptop-side video recording (REC VIDEO button)
        self._recording = False
        self._rec_writer = None
        self._rec_path = None

        self._build_ui()

        # Network
        self.net = NetworkThread(host, port)
        self.net.frame_received.connect(self._on_frame)
        self.net.telemetry_received.connect(self._on_telemetry)
        self.net.hello_received.connect(self._on_hello)
        self.net.status_changed.connect(self._on_status)
        self.net.start()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── LEFT: video + status bar ──
        left = QVBoxLayout()
        left.setSpacing(10)

        self.status = QLabel("INITIALISING")
        self.status.setStyleSheet(
            f"color:{WARN};font-size:12px;letter-spacing:2px;padding:4px;")
        left.addWidget(self.status)

        self.video = VideoCanvas()
        self.video.masks_changed.connect(self._on_masks_changed)
        left.addWidget(self.video, stretch=1)

        # readout row
        ro = QHBoxLayout()
        ro.setSpacing(8)
        self.ro_fps   = Readout("fps")
        self.ro_lat   = Readout("det latency", "ms")
        self.ro_ontgt = Readout("on target", "%")
        self.ro_res   = Readout("resolution")
        self.ro_trk   = Readout("tracks")
        self.ro_skip  = Readout("skip")
        self.ro_link  = Readout("link", "fps")
        for w in (self.ro_fps, self.ro_lat, self.ro_ontgt, self.ro_res,
                  self.ro_trk, self.ro_skip, self.ro_link):
            ro.addWidget(w)
        left.addLayout(ro)
        root.addLayout(left, stretch=3)

        # ── RIGHT: control column ──
        right = QVBoxLayout()
        right.setSpacing(10)
        right.setContentsMargins(0, 0, 0, 0)

        # state pill
        self.pill = StatePill()
        right.addWidget(self.pill)

        # primary target line
        self.primary_lbl = QLabel("PRIMARY: none")
        self.primary_lbl.setStyleSheet(
            f"color:{ACCENT};font-size:13px;letter-spacing:1px;padding:2px;")
        right.addWidget(self.primary_lbl)

        # ── track table (click to lock) ──
        gb_tracks = QGroupBox("DETECTED TRACKS — CLICK TO LOCK")
        v = QVBoxLayout(gb_tracks)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "CONF", "BOX (x1,y1,x2,y2)"])
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._lock_selected_row)
        self.table.setMinimumHeight(150)
        v.addWidget(self.table)
        row = QHBoxLayout()
        btn_lock = QPushButton("LOCK SELECTED")
        btn_lock.setObjectName("primary")
        btn_lock.clicked.connect(self._lock_selected_row)
        btn_reset = QPushButton("RESET TARGET")
        btn_reset.clicked.connect(
            lambda: self.net.send_command({"cmd": proto.CMD_RESET_TARGET}))
        row.addWidget(btn_lock)
        row.addWidget(btn_reset)
        v.addLayout(row)
        right.addWidget(gb_tracks)

        # ── source / run control ──
        gb_run = QGroupBox("RUN CONTROL")
        g = QGridLayout(gb_run)
        g.addWidget(QLabel("SOURCE"), 0, 0)
        self.source_box = QComboBox()
        self.source_box.setEditable(True)
        self.source_box.addItems(["jetson", "jetson60", "0", "1"])
        g.addWidget(self.source_box, 0, 1)
        btn_src = QPushButton("APPLY")
        btn_src.clicked.connect(self._apply_source)
        g.addWidget(btn_src, 0, 2)

        self.btn_pause = QPushButton("PAUSE")
        self._paused = False
        self.btn_pause.clicked.connect(self._toggle_pause)
        g.addWidget(self.btn_pause, 1, 0, 1, 3)

        self.btn_clear_masks = QPushButton("CLEAR MASKS")
        self.btn_clear_masks.clicked.connect(self.video.clear_masks)
        g.addWidget(self.btn_clear_masks, 2, 0, 1, 3)
        hint = QLabel("drag on video = reject zone · right-click = delete")
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;letter-spacing:1px;")
        g.addWidget(hint, 3, 0, 1, 3)
        right.addWidget(gb_run)

        # ── tunables ──
        gb_tune = QGroupBox("LIVE TUNABLES")
        t = QGridLayout(gb_tune)

        t.addWidget(QLabel("conf"), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.35)
        self.conf_spin.editingFinished.connect(
            lambda: self._set_param("conf_thres", self.conf_spin.value()))
        t.addWidget(self.conf_spin, 0, 1)

        t.addWidget(QLabel("skip ms"), 1, 0)
        self.lat_spin = QSpinBox()
        self.lat_spin.setRange(0, 500)
        self.lat_spin.setValue(110)
        self.lat_spin.editingFinished.connect(
            lambda: self._set_param("latency_skip_ms", self.lat_spin.value()))
        t.addWidget(self.lat_spin, 1, 1)

        t.addWidget(QLabel("jpeg q"), 2, 0)
        self.jq_slider = QSlider(Qt.Orientation.Horizontal)
        self.jq_slider.setRange(20, 95)
        self.jq_slider.setValue(80)
        self.jq_slider.sliderReleased.connect(
            lambda: self._set_param("jpeg_quality", self.jq_slider.value()))
        t.addWidget(self.jq_slider, 2, 1)

        self.cb_predict = QCheckBox("predict on skip")
        self.cb_predict.setChecked(True)
        self.cb_predict.toggled.connect(
            lambda v: self._set_param("predict_on_skip", v))
        t.addWidget(self.cb_predict, 3, 0, 1, 2)

        self.cb_switch = QCheckBox("auto-switch on loss")
        self.cb_switch.setChecked(True)
        self.cb_switch.toggled.connect(
            lambda v: self._set_param("auto_switch_on_loss", v))
        t.addWidget(self.cb_switch, 4, 0, 1, 2)
        right.addWidget(gb_tune)

        # ── logging ──
        gb_log = QGroupBox("LOGGING / RECORDING")
        lg = QHBoxLayout(gb_log)
        self.btn_log = QPushButton("START LOG")
        self.btn_log.clicked.connect(self._toggle_logging)
        self.btn_rec = QPushButton("REC VIDEO")
        self._recording = False
        self._rec_writer = None
        self._rec_path = None
        self.btn_rec.clicked.connect(self._toggle_recording)
        lg.addWidget(self.btn_log)
        lg.addWidget(self.btn_rec)
        right.addWidget(gb_log)

        # ── event console ──
        gb_ev = QGroupBox("EVENT LOG")
        ev = QVBoxLayout(gb_ev)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(400)
        self.console.setMinimumHeight(110)
        ev.addWidget(self.console)
        right.addWidget(gb_ev)

        right.addStretch(1)
        root.addLayout(right, stretch=1)

    # ── network event handlers (run on GUI thread) ──────────────────────────

    def _on_status(self, msg: str, color: str) -> None:
        self.status.setText(msg.upper())
        self.status.setStyleSheet(
            f"color:{color};font-size:12px;letter-spacing:2px;padding:4px;")
        self._log_console(msg)

    def _on_hello(self, caps: dict) -> None:
        self._caps = caps
        self._log_console(
            f"engine: {caps.get('width')}x{caps.get('height')} "
            f"src={caps.get('source')} slices={caps.get('sahi_slices')}")

    def _on_frame(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        self._frame_size = (w, h)
        # Link FPS = rate at which frames actually arrive at the laptop. If this
        # is well below the engine's reported FPS, the link (WiFi/SSH) is the
        # bottleneck, not the tracker.
        now = time.perf_counter()
        self._link_times.append(now)
        if self._video_writer is not None:      # part of a full log run
            self._video_writer.write(img)
        if self._rec_writer is not None:        # standalone REC VIDEO
            self._rec_writer.write(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.video.set_frame(QPixmap.fromImage(qimg))

    def _on_telemetry(self, t: dict) -> None:
        self._last_telem = t
        self.ro_fps.set(f"{t.get('fps', 0):.1f}", DATA)
        lat = t.get("det_latency_ms", 0)
        self.ro_lat.set(f"{lat:.0f}", GOLD if lat > 110 else DATA)

        # On-target % — headline spec metric (laser on drone >= 80%).
        # Gold when the spec threshold is met, dim otherwise.
        ot = t.get("on_target_pct")
        if ot is None:
            self.ro_ontgt.set("--", MUTED)
        else:
            self.ro_ontgt.set(f"{ot:.0f}", GOLD if ot >= 80 else WHITE)

        self.ro_res.set(f"{t.get('width')}x{t.get('height')}", DATA)
        self.ro_trk.set(t.get("n_tracks", 0), DATA)
        self.ro_skip.set("YES" if t.get("skipped") else "no",
                         WARN if t.get("skipped") else MUTED)

        # Link health: dropped frames are gaps in the engine's frame_no that
        # never reached us (newest-wins streaming drops stale frames). Link FPS
        # is the actual arrival rate at the laptop.
        fn = t.get("frame_no")
        if fn is not None and self._last_frame_no is not None:
            gap = fn - self._last_frame_no - 1
            if gap > 0:
                self._dropped_total += gap
        if fn is not None:
            self._last_frame_no = fn
        if len(self._link_times) >= 2:
            span = self._link_times[-1] - self._link_times[0]
            link_fps = (len(self._link_times) - 1) / span if span > 0 else 0.0
            eng_fps = t.get("fps", 0) or 0.0
            # Gold-flag when the link is dropping >25% vs the engine rate.
            lagging = eng_fps > 1 and link_fps < 0.75 * eng_fps
            self.ro_link.set(f"{link_fps:.0f}", GOLD if lagging else DATA)

        self.pill.set_state(t.get("state", "LOST"))
        pid = t.get("primary_id")          # stable, operator-facing
        raw_pid = t.get("raw_primary_id")  # current OC-SORT id (for starring)
        self.primary_lbl.setText(
            f"PRIMARY: ID{pid}" if pid is not None else "PRIMARY: none")

        self._refresh_table(t.get("tracks", []), raw_pid)
        for ev in t.get("events", []):
            self._log_console(f"· {ev}")

        if self._logging:
            self._write_log(t)

    def _refresh_table(self, tracks: list, primary_id) -> None:
        # Preserve selection by id across refreshes.
        sel_id = None
        items = self.table.selectedItems()
        if items:
            sel_id = self.table.item(items[0].row(), 0).text()

        self.table.setRowCount(len(tracks))
        for r, trk in enumerate(tracks):
            tid = str(trk["id"])
            conf = trk["score"]
            conf_s = f"{conf:.2f}" if conf is not None else "—"
            box_s = ",".join(str(int(v)) for v in trk["box"])
            id_item = QTableWidgetItem(tid)
            if trk["id"] == primary_id:
                id_item.setForeground(QColor(ACCENT))
                id_item.setText(f"★ {tid}")
            self.table.setItem(r, 0, id_item)
            self.table.setItem(r, 1, QTableWidgetItem(conf_s))
            self.table.setItem(r, 2, QTableWidgetItem(box_s))
            if sel_id is not None and tid == sel_id.lstrip("★ ").strip():
                self.table.selectRow(r)

    # ── control actions ──────────────────────────────────────────────────────

    def _lock_selected_row(self, *args) -> None:
        items = self.table.selectedItems()
        if not items:
            self._log_console("no track selected")
            return
        raw = self.table.item(items[0].row(), 0).text()
        tid = int(raw.lstrip("★ ").strip())
        self.net.send_command(
            {"cmd": proto.CMD_SELECT_TARGET, "track_id": tid})
        self._log_console(f">> lock target ID{tid}")

    def _apply_source(self) -> None:
        src = self.source_box.currentText().strip()
        self.net.send_command({"cmd": proto.CMD_SET_SOURCE, "source": src})
        self._log_console(f">> set source: {src}")

    def _on_masks_changed(self, masks: list) -> None:
        # masks: list of (fx1,fy1,fx2,fy2) normalised fractions.
        self.net.send_command({
            "cmd": proto.CMD_SET_MASKS,
            "masks": [list(m) for m in masks],
        })
        self._log_console(f">> reject masks: {len(masks)}")

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.net.send_command({"cmd": proto.CMD_PAUSE, "value": self._paused})
        self.btn_pause.setText("RESUME" if self._paused else "PAUSE")

    def _set_param(self, name: str, value) -> None:
        self.net.send_command(
            {"cmd": proto.CMD_SET_PARAM, "name": name, "value": value})
        self._log_console(f">> {name} = {value}")

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        w, h = self._frame_size
        if (w, h) == (0, 0):
            self._log_console("cannot record: no video yet")
            return
        os.makedirs("recordings", exist_ok=True)
        self._rec_path = time.strftime("recordings/argus_%Y%m%d_%H%M%S.mp4")
        self._rec_writer = cv2.VideoWriter(
            self._rec_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
        if not self._rec_writer.isOpened():
            self._rec_writer = None
            self._log_console("recorder failed to open")
            return
        self._recording = True
        self.btn_rec.setText("STOP REC")
        self.btn_rec.setObjectName("primary")
        self.btn_rec.setStyle(self.btn_rec.style())
        self._log_console(f">> recording video -> {self._rec_path}")

    def _stop_recording(self) -> None:
        self._recording = False
        if self._rec_writer is not None:
            self._rec_writer.release()
            self._rec_writer = None
        self.btn_rec.setText("REC VIDEO")
        self.btn_rec.setObjectName("")
        self.btn_rec.setStyle(self.btn_rec.style())
        self._log_console(f">> video saved: {self._rec_path}")

    # ── local logging ────────────────────────────────────────────────────────

    def _toggle_logging(self) -> None:
        if self._logging:
            self._stop_logging()
        else:
            self._start_logging()

    def _start_logging(self) -> None:
        self._log_dir = time.strftime("logs/run_%Y%m%d_%H%M%S")
        os.makedirs(self._log_dir, exist_ok=True)
        self._csv_file = open(os.path.join(self._log_dir, "telemetry.csv"),
                              "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["t_wall", "frame_no", "fps", "det_latency_ms", "width", "height",
             "state", "n_tracks", "skipped", "primary_id",
             "on_target_pct", "on_target_session_pct"])
        self._jsonl_file = open(
            os.path.join(self._log_dir, "telemetry.jsonl"), "w")
        w, h = self._frame_size if self._frame_size != (0, 0) else (
            self._caps.get("width", 1920), self._caps.get("height", 1080))
        self._video_writer = cv2.VideoWriter(
            os.path.join(self._log_dir, "received.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
        self._logging = True
        self.btn_log.setText("STOP LOG")
        self.btn_log.setObjectName("danger")
        self.btn_log.setStyle(self.btn_log.style())
        self._log_console(f">> logging to {self._log_dir}/")

    def _write_log(self, t: dict) -> None:
        try:
            self._csv_writer.writerow([
                f"{time.time():.3f}", t.get("frame_no"), t.get("fps"),
                t.get("det_latency_ms"), t.get("width"), t.get("height"),
                t.get("state"), t.get("n_tracks"), int(bool(t.get("skipped"))),
                t.get("primary_id"),
                t.get("on_target_pct"), t.get("on_target_session_pct"),
            ])
            self._jsonl_file.write(json.dumps(
                {"t_wall": time.time(), **t}) + "\n")
        except (ValueError, OSError):
            pass

    def _stop_logging(self) -> None:
        self._logging = False
        for f in (self._csv_file, self._jsonl_file):
            try:
                if f:
                    f.close()
            except OSError:
                pass
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self.btn_log.setText("START LOG")
        self.btn_log.setObjectName("")
        self.btn_log.setStyle(self.btn_log.style())
        self._log_console(f">> log saved: {self._log_dir}/")

    # ── console ──────────────────────────────────────────────────────────────

    def _log_console(self, msg: str) -> None:
        self.console.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── shutdown ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._logging:
            self._stop_logging()
        if self._recording:
            self._stop_recording()
        self.net.stop()
        self.net.wait(1500)
        super().closeEvent(event)


def main() -> None:
    p = argparse.ArgumentParser(description="Drone tracker ground-control GUI")
    p.add_argument("--host", default="localhost",
                   help="server host (localhost when using an SSH tunnel)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    app = QApplication([])
    app.setApplicationName("ARGUS Control")
    app.setStyle("Fusion")
    # Application-level icon (used by some window managers / the taskbar).
    _app_icon = QIcon("hyperion_mark.ppm")
    if _app_icon.isNull():
        _app_icon = QIcon("hyperion_icon.png")
    if not _app_icon.isNull():
        app.setWindowIcon(_app_icon)
    win = CommandCenter(args.host, args.port)
    win.showMaximized()
    app.exec()


if __name__ == "__main__":
    main()
