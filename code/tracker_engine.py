from __future__ import annotations
import argparse
import contextlib
import inspect
import math
import os
import sys
import time
from collections import deque
from enum import Enum, auto
from pathlib import Path

# Prevent the PyPI ocsort package from opening a Matplotlib toolbar.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass

import cv2
import numpy as np
import torch

try:
    import trtsahi
except ImportError as exc:
    raise ImportError(
        "trtsahi not found.\n"
        "Build from: https://github.com/leon0514/trt-sahi-yolo\n"
        "  cd trt-sahi-yolo && make && pip install ./python"
    ) from exc

try:
    from ocsort import OCSort
except ImportError as exc:
    raise ImportError(
        "ocsort not installed.  Try:\n"
        "  pip install ocsort\n"
        "Or clone and copy the folder:\n"
        "  git clone https://github.com/noahcao/OC_SORT && cp -r OC_SORT/ocsort ."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE_PATH = "nano_best.engine"
CLASS_NAME  = "drone"
MODEL_TYPE  = trtsahi.ModelType.YOLO11SAHI

CONF_THRES = 0.40
NMS_THRES  = 0.45

# IMPORTANT: auto_slice lets the trtsahi library pick its own slice size; keep
# it OFF so the explicit values below are used.
#
# Tile size is the single biggest lever for both FPS and the false-positive
# rate. Smaller tiles (e.g. 640) keep a distant drone at native pixel size but
# (a) multiply the number of tiles, so more independent inferences over busy
# background texture each get a fresh chance to hallucinate a "drone", driving
# false positives up, and (b) cost more compute. Larger tiles cover more
# context per inference and produce fewer false positives, at the cost of
# downscaling the target on resize to the model's 640 input.
#
# Empirically (operator report) ~920-px tiles gave far fewer false positives at
# a lower conf threshold than 640. 832x832 / 0.2 overlap on 1920x1080 yields
# 3x2 = 6 tiles (vs 8 at 640): ~25% less inference per frame AND fewer
# false-positive opportunities. 832 is a multiple of the network stride (32).
SAHI_AUTO_SLICE       = False
SAHI_SLICE_WIDTH      = 832
SAHI_SLICE_HEIGHT     = 832
SAHI_HORIZONTAL_RATIO = 0.2
SAHI_VERTICAL_RATIO   = 0.2
SAHI_MAX_BATCH        = 8   # 6 tiles fit in one batched pass; 8 leaves headroom
GPU_ID                = 0

# Raised from 10 → 25 px²: rejects single-pixel noise while still catching
# a ~5×5 blob (a 30 cm drone at 50+ m on 1080p).
MIN_BOX_AREA = 25

# A drone should never cover more than 10% of the frame area.
# At 5 m distance a 30 cm drone on 1080p is still only ~2-3% of frame area.
# This kills false positives caused by large uniform regions (ceilings,
# bright lights, sky patches) that the model mistakes for a drone.
MAX_BOX_AREA_FRACTION = 0.10

# OC-SORT
OCS_DET_THRESH = CONF_THRES
OCS_MAX_AGE    = 45
OCS_MIN_HITS   = 1
OCS_IOU_THRESH = 0.15
OCS_DELTA_T    = 3
OCS_ASSO_FUNC  = "giou"
OCS_INERTIA    = 0.2
OCS_USE_BYTE   = True

# NanoTrack coasting
MAX_COAST_FRAMES = 30
NANO_BACKBONE = "nanotrack_backbone_sim.onnx"
NANO_NECKHEAD = "nanotrack_head_sim.onnx"
REACQUIRE_DIST = 80    # px — max centre-to-centre distance
REACQUIRE_IOU  = 0.10  # min IoU(nano_box, candidate_box)
BOX_AREA_RATIO_MIN = 0.25
BOX_AREA_RATIO_MAX = 4.0
SAHI_LATENCY_SKIP_THRESH = 0.110   # 110 ms
PREDICT_ON_SKIP        = True
VEL_EMA                = 0.5     # EMA weight for the per-frame centre velocity
MAX_PREDICT_FRAMES     = 3       # cap on consecutive predicted frames
MAX_PREDICT_CENTRE_VEL = 200.0   # px/frame clamp, guards against glitch flings
SCORE_MATCH_IOU = 0.30
LOST_GATE_DIST       = 120
AUTO_SELECT_MIN_CONF = CONF_THRES
AUTO_SWITCH_ON_LOSS  = True
FPS_EMA_ALPHA = 0.1

# On-target metric: rolling window length (frames) for the live percentage.
# ~150 frames ≈ 5 s at 30 fps — long enough to be stable, short enough to react.
ONTARGET_WINDOW = 150

# Webcam fallback settings
CAM_INDEX  = 0
CAM_WIDTH  = 1920   
CAM_HEIGHT = 1080
CAM_FPS    = 30

# Jetson / IMX-219 CSI settings
CSI_WIDTH  = 1920
CSI_HEIGHT = 1080
CSI_FPS_30 = 30
CSI_FPS_60 = 60   # 60 fps crops the sensor; FOV narrows slightly


# ---------------------------------------------------------------------------
# Native-output suppression
# ---------------------------------------------------------------------------
#
# The compiled trtsahi.so library writes a lot of chatter straight to the OS
# stdout/stderr file descriptors (the SIMKAI.TTF font error, the "CUDA SAHI
# CROP IMAGE" tables, "num image:" lines, etc.).  Because it bypasses Python's
# `print`/logging, the only reliable way to silence it is to redirect the
# underlying file descriptors (1 and 2) at the OS level for the duration of the
# noisy call.  Set the env var ARGUS_VERBOSE=1 to leave everything visible for
# debugging.

_VERBOSE = os.environ.get("ARGUS_VERBOSE", "0") == "1"


@contextlib.contextmanager
def _suppress_native_output():
    """Redirect OS-level stdout+stderr to /dev/null inside the block."""
    if _VERBOSE:
        yield
        return
    # Flush Python-level buffers first so our own prints aren't lost/reordered.
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


# ---------------------------------------------------------------------------
# Tracking state
# ---------------------------------------------------------------------------

class TrackState(Enum):
    TRACKING = auto()   # OC-SORT confirmed track, or short prediction bridge
    COASTING = auto()   # NanoTrack coasting (OC-SORT lost target)
    LOST     = auto()   # No target at all


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid SAHI + OC-SORT + NanoTrack drone tracker"
    )
    p.add_argument("--engine",   default=ENGINE_PATH,
                   help="Path to TensorRT .engine file")
    p.add_argument("--backbone", default=NANO_BACKBONE,
                   help="NanoTrack backbone ONNX path")
    p.add_argument("--neckhead", default=NANO_NECKHEAD,
                   help="NanoTrack neck/head ONNX path")
    p.add_argument(
        "--source", default=None,
        help=(
            "Video source.  Options:\n"
            "  (omit)     : default webcam (index 0)\n"
            "  <int>      : webcam at that index\n"
            "  jetson     : IMX-219 CSI @ 1920×1080 30 fps\n"
            "  jetson60   : IMX-219 CSI @ 1920×1080 60 fps\n"
            "  <file.mp4> : video file"
        ),
    )
    # ── Runtime overrides for the tuning knobs touched by this revision ──
    p.add_argument(
        "--latency-skip", type=float, default=None, metavar="MS",
        help=(
            "Override the adaptive-skip latency threshold, in milliseconds "
            f"(default {int(SAHI_LATENCY_SKIP_THRESH * 1000)}). "
            "Set 0 to disable detection skipping entirely."
        ),
    )
    p.add_argument(
        "--no-skip-predict", action="store_true",
        help=(
            "Disable constant-velocity prediction on skipped detection "
            "frames; revert to reusing the previous frame's detections."
        ),
    )
    p.add_argument(
        "--no-auto-switch", action="store_true",
        help=(
            "After losing the primary target, do NOT auto-lock a different "
            "target — only re-lock the same one, or wait for a manual reset."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# GStreamer / Jetson helpers
# ---------------------------------------------------------------------------

def _gst_pipeline(width: int, height: int, fps: int, sensor_id: int = 0) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )


def _open_jetson(fps: int, sensor_id: int = 0) -> cv2.VideoCapture:
    pipeline = _gst_pipeline(CSI_WIDTH, CSI_HEIGHT, fps, sensor_id)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open CSI camera sensor-id={sensor_id} via GStreamer.\n"
            "Ensure nvarguscamerasrc is available (JetPack ≥ 5) and the\n"
            "camera ribbon is seated correctly.\n"
            f"Pipeline attempted:\n  {pipeline}"
        )
    print(f"Opened IMX-219 CSI camera sensor-id={sensor_id}  "
          f"({CSI_WIDTH}×{CSI_HEIGHT} @ {fps} fps)")
    return cap


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _parse_jetson_source(source) -> tuple[int, int] | None:
    """
    Decode a CSI-camera source string into (sensor_id, fps), or None if the
    string isn't a jetson source.

    Accepted forms (case-insensitive):
        jetson            -> sensor 0, 30 fps   (back-compat)
        jetson0 / jetson1 -> that sensor, 30 fps
        jetson60          -> sensor 0, 60 fps   (back-compat)
        jetson0_60 / jetson1_60 -> that sensor, 60 fps
        jetson60_1        -> also accepted (sensor 1, 60 fps)
    """
    if not isinstance(source, str):
        return None
    s = source.lower()
    if not s.startswith("jetson"):
        return None
    rest = s[len("jetson"):]            # e.g. "", "1", "60", "1_60", "60_1"
    digits = [int(tok) for tok in rest.replace("_", " ").split() if tok.isdigit()]
    sensor_id = 0
    fps = CSI_FPS_30
    for d in digits:
        if d in (30, 60):
            fps = d
        else:
            sensor_id = d              # any other bare number is a sensor id
    return sensor_id, fps


def open_capture(source) -> tuple[cv2.VideoCapture, bool]:
    """Returns (VideoCapture, is_file)."""
    if source is None or (isinstance(source, str) and source.isdigit()):
        idx = 0 if source is None else int(source)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        print(f"Opened webcam {idx}  ({CAM_WIDTH}×{CAM_HEIGHT} @ {CAM_FPS} fps)")
        return cap, False

    jetson = _parse_jetson_source(source)
    if jetson is not None:
        sensor_id, fps = jetson
        return _open_jetson(fps, sensor_id), False

    cap   = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened file: {source}  ({w}×{h} @ {fps:.1f} fps, {total} frames)")
    return cap, True


def nominal_fps(source) -> float:
    """Best-effort nominal fps used to seed the FPS EMA (issue 6)."""
    if source is None or (isinstance(source, str) and source.isdigit()):
        return float(CAM_FPS)
    jetson = _parse_jetson_source(source)
    if jetson is not None:
        return float(jetson[1])
    return float(CAM_FPS)


# ---------------------------------------------------------------------------
# NanoTrack wrapper
# ---------------------------------------------------------------------------

# (Issue 7) Log the chosen NanoTrack backend only once for the whole process,
# rather than on every coast re-init.
_NANO_DEVICE_LOGGED = False


def _select_nano_device(params: "cv2.TrackerNano_Params") -> None:
    """
    Configure `params` to use the CUDA DNN backend when available, otherwise
    fall back to CPU.  Prints the decision exactly once per process.
    """
    global _NANO_DEVICE_LOGGED
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if cuda_available:
        params.backend = cv2.dnn.DNN_BACKEND_CUDA
        params.target  = cv2.dnn.DNN_TARGET_CUDA
        msg = "NanoTrack: using CUDA backend"
    else:
        params.backend = cv2.dnn.DNN_BACKEND_OPENCV
        params.target  = cv2.dnn.DNN_TARGET_CPU
        msg = "NanoTrack: CUDA not available — falling back to CPU backend"
    if not _NANO_DEVICE_LOGGED:
        print(msg)
        _NANO_DEVICE_LOGGED = True


def make_nano_tracker(backbone_path: str, neckhead_path: str) -> cv2.TrackerNano:
    """
    Build a cv2.TrackerNano instance configured to run on the CUDA
    backend (GPU).  Falls back to CPU if CUDA is not available.

    NanoTrack operates on the full frame — no manual cropping needed.
    The 255×255 search region and 127×127 template are handled internally
    by the tracker's DNN module.
    """
    # Validate model files up-front for a clear error message.
    for path, label in ((backbone_path, "backbone"), (neckhead_path, "neckhead")):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"NanoTrack {label} model not found: {path!r}\n"
                "Download from:\n"
                "  wget https://github.com/opencv/opencv_extra/raw/master/"
                "testdata/cv/tracking/nanotrack_backbone_sim.onnx\n"
                "  wget https://github.com/opencv/opencv_extra/raw/master/"
                "testdata/cv/tracking/nanotrack_head_sim.onnx"
            )

    params          = cv2.TrackerNano_Params()
    params.backbone = backbone_path
    params.neckhead = neckhead_path

    # Prefer CUDA; fall back gracefully to CPU.  The decision is logged once.
    _select_nano_device(params)

    return cv2.TrackerNano_create(params)


# ---------------------------------------------------------------------------
# SAHI + TRT helpers
# ---------------------------------------------------------------------------

def estimate_sahi_slices(fw: int, fh: int) -> int:
    def ax(dim, tile, ratio):
        if dim <= tile:
            return 1
        return math.ceil((dim - tile * ratio) / (tile * (1.0 - ratio)))
    return (ax(fw, SAHI_SLICE_WIDTH,  SAHI_HORIZONTAL_RATIO) *
            ax(fh, SAHI_SLICE_HEIGHT, SAHI_VERTICAL_RATIO))


def load_sahi_model(engine_path: str) -> trtsahi.TrtSahi:
    with _suppress_native_output():
        return trtsahi.TrtSahi(
            model_path=engine_path,
            model_type=MODEL_TYPE,
            names=[CLASS_NAME],
            gpu_id=GPU_ID,
            confidence_threshold=CONF_THRES,
            nms_threshold=NMS_THRES,
            max_batch_size=SAHI_MAX_BATCH,
            auto_slice=SAHI_AUTO_SLICE,
            slice_width=SAHI_SLICE_WIDTH,
            slice_height=SAHI_SLICE_HEIGHT,
            slice_horizontal_ratio=SAHI_HORIZONTAL_RATIO,
            slice_vertical_ratio=SAHI_VERTICAL_RATIO,
        )


def _unpack_box(box):
    for attrs in (
        ("left", "top", "right", "bottom"),
        ("x1",   "y1",  "x2",   "y2"),
        ("tl_x", "tl_y", "br_x", "br_y"),
    ):
        if all(hasattr(box, a) for a in attrs):
            return tuple(float(getattr(box, a)) for a in attrs)
    try:
        return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    except (TypeError, KeyError):
        pass
    raise AttributeError(f"Cannot unpack trtsahi.Box: {dir(box)}")


_box_fields_printed = False


def run_sahi(model: trtsahi.TrtSahi, frame: np.ndarray) -> np.ndarray:
    """Returns (N, 6): [x1, y1, x2, y2, score, 0.0]"""
    with _suppress_native_output():
        results = model.forwards([frame])
    detections = []
    for det in results[0]:
        x1, y1, x2, y2 = _unpack_box(det.box)
        score = float(det.score)
        x1 = max(0, min(int(x1), frame.shape[1] - 1))
        y1 = max(0, min(int(y1), frame.shape[0] - 1))
        x2 = max(0, min(int(x2), frame.shape[1] - 1))
        y2 = max(0, min(int(y2), frame.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        box_area   = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        if box_area < MIN_BOX_AREA:
            continue
        if box_area > frame_area * MAX_BOX_AREA_FRACTION:
            continue
        detections.append([x1, y1, x2, y2, score, 0.0])
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(detections, dtype=np.float32)


# ---------------------------------------------------------------------------
# OC-SORT helpers
# ---------------------------------------------------------------------------

def make_ocsort_tracker() -> OCSort:
    sig    = inspect.signature(OCSort.__init__)
    params = sig.parameters
    candidates: dict = {
        "det_thresh":    OCS_DET_THRESH,
        "max_age":       OCS_MAX_AGE,
        "min_hits":      OCS_MIN_HITS,
        "iou_threshold": OCS_IOU_THRESH,
        "delta_t":       OCS_DELTA_T,
        "asso_func":     OCS_ASSO_FUNC,
        "inertia":       OCS_INERTIA,
        "use_byte":      OCS_USE_BYTE,
    }
    kwargs = {k: v for k, v in candidates.items() if k in params}
    try:
        tracker = OCSort(**kwargs)
    except TypeError:
        try:
            tracker = OCSort(OCS_DET_THRESH, OCS_MAX_AGE, OCS_MIN_HITS,
                             OCS_IOU_THRESH, OCS_DELTA_T)
        except TypeError:
            tracker = OCSort(OCS_DET_THRESH)
    print(f"OC-SORT init: {kwargs}")
    return tracker


_ocsort_fmt_printed = False


def parse_ocsort_outputs(raw, orig_h: int, orig_w: int) -> list[dict]:
    global _ocsort_fmt_printed

    if raw is None:
        return []
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().numpy()

    def _clip(x1, y1, x2, y2):
        return (
            float(max(0, min(x1, orig_w - 1))),
            float(max(0, min(y1, orig_h - 1))),
            float(max(0, min(x2, orig_w - 1))),
            float(max(0, min(y2, orig_h - 1))),
        )

    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return []
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if not _ocsort_fmt_printed:
            print(f"[OC-SORT output] shape={arr.shape}  first row={arr[0]}")
            _ocsort_fmt_printed = True
        tracks = []
        for row in arr:
            if row.shape[0] < 5:
                continue
            x1, y1, x2, y2, track_id = row[:5]
            # OC-SORT row layout here is [x1,y1,x2,y2,track_id,cls,score];
            # the score is the LAST column, not row[5] (which is the class).
            score = float(row[-1]) if row.shape[0] >= 7 else (
                float(row[5]) if row.shape[0] == 6 else None)
            tracks.append({
                "box":      _clip(x1, y1, x2, y2),
                "track_id": int(track_id),
                "score":    score,
            })
        return tracks

    tracks = []
    for t in (raw if hasattr(raw, "__iter__") else [raw]):
        if hasattr(t, "tlbr"):
            x1, y1, x2, y2 = t.tlbr
        elif hasattr(t, "tlwh"):
            x, y, w, h = t.tlwh
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            continue
        tracks.append({
            "box":      _clip(x1, y1, x2, y2),
            "track_id": int(getattr(t, "track_id", -1)),
            "score":    float(getattr(t, "score",
                              getattr(t, "det_score", 1.0))),
        })
    return tracks


def update_ocsort_compat(ocsort: OCSort, detections: np.ndarray,
                         orig_h: int, orig_w: int):
    try:
        return ocsort.update(detections,
                             img_info=(orig_h, orig_w),
                             img_size=(orig_h, orig_w))
    except TypeError:
        pass
    try:
        return ocsort.update(detections, (orig_h, orig_w), (orig_h, orig_w))
    except TypeError:
        pass
    if detections.size == 0:
        det_tensor = torch.empty((0, 6), dtype=torch.float32)
    else:
        det_tensor = torch.as_tensor(detections, dtype=torch.float32)
    try:
        return ocsort.update(det_tensor, None)
    except TypeError:
        pass
    return ocsort.update(detections)


def attach_detection_scores(tracks: list[dict], detections: np.ndarray) -> list[dict]:
    if detections is None or len(detections) == 0:
        return tracks
    for t in tracks:
        best_iou   = 0.0
        best_score = None
        for d in detections:
            iou = _box_iou(t["box"], (float(d[0]), float(d[1]),
                                      float(d[2]), float(d[3])))
            if iou > best_iou:
                best_iou   = iou
                best_score = float(d[4])
        if best_score is not None and best_iou >= SCORE_MATCH_IOU:
            t["score"] = best_score
    return tracks


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _centre(box: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(bx - ax, by - ay)


def _box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _xywh_to_tlbr(xywh) -> tuple[float, float, float, float]:
    x, y, w, h = xywh
    return float(x), float(y), float(x + w), float(y + h)


def _tlbr_to_xywh(box: tuple) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


# ---------------------------------------------------------------------------
# Primary-target tracker  (OC-SORT primary + NanoTrack coasting)
# ---------------------------------------------------------------------------

class PrimaryTargetTracker:
    """
    Manages the single "laser target" track.

    State machine
    -------------
    TRACKING  : OC-SORT has a confirmed track for the primary ID, OR a short
                constant-velocity prediction is bridging a skipped detection
                frame (issue 2).
    COASTING  : OC-SORT lost the primary; NanoTrack is bridging the gap.
    LOST      : Neither source has a valid target.

    Skip-frame prediction (issue 2)
    -------------------------------
    When the adaptive controller skips a detection frame, OC-SORT is fed an
    empty detection set and therefore returns no track for the primary that
    frame.  Rather than freeze the box, we extrapolate the last confirmed box
    using a smoothed per-frame centre velocity.  This is distinct from
    COASTING: it only bridges a single skipped frame while actively tracking,
    it never spins up NanoTrack, and the prediction horizon is capped by
    MAX_PREDICT_FRAMES.  Velocity is estimated ONLY from real OC-SORT
    confirmations, so predicted boxes never feed back into the velocity model.

    NanoTrack coasting notes
    ------------------------
    • NanoTrack.init() takes the full frame + an (x, y, w, h) ROI.
    • NanoTrack.update() takes the full frame and returns (ok, (x,y,w,h)).
    • No manual crop/offset bookkeeping is needed — the tracker handles
      its own 255×255 search region internally.
    • Because NanoTrack is so fast on GPU, MAX_COAST_FRAMES is raised to
      30 (vs 15 for CSRT) giving more time to re-acquire the target.

    Re-association guard
    --------------------
    An OC-SORT track is accepted for re-association during coast only if:
      (a) its centre is within REACQUIRE_DIST px of the NanoTrack box, AND
      (b) IoU(nano_box, candidate_box) ≥ REACQUIRE_IOU

    Target (re)selection after loss (issue 4)
    -----------------------------------------
    On a clean re-acquisition we first try to re-lock the SAME target by
    proximity to the last known box (within LOST_GATE_DIST).  Only if nothing
    is near do we acquire a new target, and only above AUTO_SELECT_MIN_CONF.
    Switching to a different target is logged, and can be disabled entirely
    with AUTO_SWITCH_ON_LOSS.

    Validity guard
    --------------
    After each NanoTrack update the returned box is rejected if:
      • area has changed by more than BOX_AREA_RATIO_MAX / MIN vs coast start
      • box centre is outside the frame
    """

    def __init__(self, backbone_path: str, neckhead_path: str) -> None:
        self._backbone_path  = backbone_path
        self._neckhead_path  = neckhead_path
        self._nano:          cv2.TrackerNano | None = None
        self._primary_id:    int | None             = None  # raw OC-SORT id
        # Operator-facing stable id: assigned once when a target is first
        # locked, and kept through coasting + re-association so brief flickers
        # (OC-SORT deletes the track and re-issues a new raw id) do NOT change
        # the number the operator sees or has locked.  Only a genuine fresh
        # acquisition or an explicit operator switch advances it.
        self._stable_id:     int | None             = None
        self._stable_counter: int                   = 0
        self._coast_frames:  int                    = 0
        self._last_box:      tuple | None           = None   # (x1,y1,x2,y2)
        self._coast_init_area: float                = 0.0
        self._state:         TrackState             = TrackState.LOST

        # ── Skip-frame constant-velocity predictor state (issue 2) ──
        self._frame_no:             int                          = 0
        self._centre_velocity:      tuple[float, float] | None   = None
        self._last_confirmed_centre: tuple[float, float] | None  = None
        self._last_confirmed_size:  tuple[float, float]          = (0.0, 0.0)
        self._last_confirm_frame_no: int                         = -10 ** 9

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> TrackState:
        return self._state

    @property
    def primary_id(self) -> int | None:
        """Operator-facing stable id (survives brief track losses)."""
        return self._stable_id

    @property
    def raw_primary_id(self) -> int | None:
        """The current underlying OC-SORT id (changes on re-association)."""
        return self._primary_id

    def update(
        self,
        frame: np.ndarray,
        ocsort_tracks: list[dict],
        detection_ran: bool = True,
    ) -> tuple[tuple | None, TrackState]:
        """
        Parameters
        ----------
        detection_ran : whether SAHI detection actually ran this frame.  When
            False (a skipped frame), absence of an OC-SORT track is expected
            and is bridged by prediction rather than treated as a loss.

        Returns
        -------
        (box, state)
            box   : (x1, y1, x2, y2) in pixel coords, or None if LOST.
            state : TrackState
        """
        self._frame_no += 1
        primary = self._find_primary(ocsort_tracks)

        # ── Case 1: OC-SORT has a confirmed track for the primary ─────
        if primary is not None:
            # Accumulate velocity only across uninterrupted TRACKING frames;
            # any transition out of TRACKING (coast/lost/fresh lock) resets it.
            continuing = (self._state == TrackState.TRACKING)
            self._primary_id   = primary["track_id"]
            self._coast_frames = 0
            self._record_confirmed(primary["box"], continuing=continuing)
            self._state = TrackState.TRACKING
            return self._last_box, self._state

        # ── Case 1.5: skipped detection frame — predict for display ───
        # Only bridges a short gap while actively tracking; never coasts.
        if (not detection_ran
                and PREDICT_ON_SKIP
                and self._state == TrackState.TRACKING
                and self._last_confirmed_centre is not None
                and (self._frame_no - self._last_confirm_frame_no)
                    <= MAX_PREDICT_FRAMES):
            self._last_box = self._extrapolate(frame.shape)
            # State remains TRACKING — the target isn't lost, we simply did
            # not re-detect it on this deliberately-skipped frame.
            return self._last_box, self._state

        # ── Case 2: OC-SORT lost the primary — NanoTrack coasting ─────
        if self._last_box is not None and self._coast_frames < MAX_COAST_FRAMES:
            if self._coast_frames == 0:
                # First coast frame: initialise NanoTrack from the last
                # confirmed OC-SORT box.
                self._reinit_nano(frame, self._last_box)
                x1, y1, x2, y2 = self._last_box
                self._coast_init_area = max(1.0, (x2 - x1) * (y2 - y1))

            ok, nano_box = self._update_nano(frame)
            self._coast_frames += 1

            if ok and self._is_nano_valid(nano_box, frame.shape):
                # Try to re-associate with the nearest OC-SORT track
                reacq = self._nearest_track(nano_box, ocsort_tracks)
                if reacq is not None:
                    self._reinit_nano(frame, reacq["box"])
                    self._adopt(reacq, announce=False, fresh=False)
                    return self._last_box, self._state

                self._last_box = nano_box
                self._state    = TrackState.COASTING
                return self._last_box, self._state

        # ── Case 3: re-acquire / fresh-acquire from OC-SORT tracks ────
        if ocsort_tracks:
            # (a) Continuity: prefer re-locking the SAME target by proximity
            #     to the last known box (issue 4).
            if self._last_box is not None:
                cand = self._nearest_within(self._last_box, ocsort_tracks,
                                             LOST_GATE_DIST)
                if cand is not None:
                    self._adopt(cand, announce=True, fresh=False)
                    return self._last_box, self._state
                if not AUTO_SWITCH_ON_LOSS:
                    # Original target gone and auto-switch disabled — wait.
                    self._state = TrackState.LOST
                    return None, self._state

            # (b) Fresh acquisition: highest-confidence track, but only if it
            #     clears the minimum-confidence gate (issue 4).
            best = max(ocsort_tracks, key=self._track_score)
            if self._track_score(best) >= AUTO_SELECT_MIN_CONF:
                self._adopt(best, announce=True, fresh=True)
                return self._last_box, self._state

            # Nothing confident enough — do not latch onto a low-conf blob.
            self._state = TrackState.LOST
            return None, self._state

        # ── Case 4: Truly lost ─────────────────────────────────────────
        self._state = TrackState.LOST
        return None, self._state

    def select(self, track_id: int, ocsort_tracks: list[dict]) -> bool:
        """
        Manually lock onto a specific OC-SORT track id (operator click in the
        command center).  Returns True if the id is currently present and was
        adopted.  Treated as a fresh lock: velocity estimate is cleared and any
        ongoing coast is abandoned.
        """
        for t in ocsort_tracks:
            if t["track_id"] == track_id:
                self._nano         = None
                self._coast_frames = 0
                # Force-adopt regardless of confidence: the operator chose it.
                self._primary_id   = track_id
                self._stable_id    = self._new_stable_id()
                self._record_confirmed(t["box"], continuing=False)
                self._state = TrackState.TRACKING
                print(f"[primary] operator selected raw ID{track_id} "
                      f"-> stable ID{self._stable_id}")
                return True
        return False

    def reset(self) -> None:
        self._nano            = None
        self._primary_id      = None
        self._stable_id       = None
        self._coast_frames    = 0
        self._last_box        = None
        self._coast_init_area = 0.0
        self._state           = TrackState.LOST
        # Predictor state
        self._centre_velocity        = None
        self._last_confirmed_centre  = None
        self._last_confirmed_size    = (0.0, 0.0)
        self._last_confirm_frame_no  = -10 ** 9

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _track_score(t: dict) -> float:
        """
        Confidence used for target selection.  With detection scores now
        attached (issue 3), a real confidence is preferred.  Tracks with no
        score are pushed below any scored track (they are usually internally
        coasted), with a negligible area term only to break ties among them.
        """
        s = t["score"]
        if s is not None:
            return s
        x1, y1, x2, y2 = t["box"]
        area = (x2 - x1) * (y2 - y1)
        return -1.0 + min(area, 1e4) / 1e9

    def _find_primary(self, tracks: list[dict]) -> dict | None:
        if self._primary_id is None:
            return None
        for t in tracks:
            if t["track_id"] == self._primary_id:
                return t
        return None

    def _nearest_track(self, box: tuple, tracks: list[dict]) -> dict | None:
        """
        Return the nearest OC-SORT track only if it satisfies BOTH the
        proximity AND IoU conditions.  Either failing means no re-association
        to avoid latching onto a different nearby object.
        """
        if not tracks:
            return None
        cx, cy       = _centre(box)
        best, best_d = None, float("inf")
        for t in tracks:
            d = _dist(cx, cy, *_centre(t["box"]))
            if d < best_d:
                best, best_d = t, d
        if best is None or best_d > REACQUIRE_DIST:
            return None
        if _box_iou(box, best["box"]) < REACQUIRE_IOU:
            return None
        return best

    def _nearest_within(self, box: tuple, tracks: list[dict],
                        max_dist: float) -> dict | None:
        """
        Distance-only nearest-track gate, used for post-loss continuity
        (issue 4).  No IoU requirement here because after a loss the candidate
        box generally won't overlap the (stale) last known box.
        """
        if not tracks:
            return None
        cx, cy       = _centre(box)
        best, best_d = None, float("inf")
        for t in tracks:
            d = _dist(cx, cy, *_centre(t["box"]))
            if d < best_d:
                best, best_d = t, d
        if best is not None and best_d <= max_dist:
            return best
        return None

    def _new_stable_id(self) -> int:
        """Mint a fresh operator-facing id.  Counts up from 1, independent of
        OC-SORT's churning raw ids."""
        self._stable_counter += 1
        return self._stable_counter

    def _adopt(self, track: dict, *, announce: bool, fresh: bool = True) -> None:
        """
        Adopt `track` as the primary target.

        fresh=True   genuinely new target (auto fresh-acquire, or operator
                     click): mint a NEW stable id.
        fresh=False  the SAME physical target reappearing after a brief loss
                     (coast re-association, or proximity re-lock): KEEP the
                     existing stable id so the operator-facing number doesn't
                     jump.  Only the raw OC-SORT id underneath changes.
        """
        new_id = track["track_id"]
        if announce and self._primary_id is not None and new_id != self._primary_id:
            kind = "switched" if fresh else "re-acquired"
            print(f"[primary] target {kind}: raw ID{self._primary_id} "
                  f"-> raw ID{new_id} (stable ID{self._stable_id})")
        self._primary_id   = new_id
        if fresh or self._stable_id is None:
            self._stable_id = self._new_stable_id()
        self._coast_frames = 0
        self._record_confirmed(track["box"], continuing=False)
        self._state = TrackState.TRACKING

    def _record_confirmed(self, box: tuple, *, continuing: bool) -> None:
        """
        Record a confirmed primary box and, when `continuing`, update the
        smoothed per-frame centre velocity used for skip-frame prediction.
        Velocity is normalised by the number of render frames since the last
        confirmation, so it stays correct whether or not frames were skipped
        in between.
        """
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = (x2 - x1), (y2 - y1)

        if continuing and self._last_confirmed_centre is not None:
            gap    = max(1, self._frame_no - self._last_confirm_frame_no)
            raw_vx = (cx - self._last_confirmed_centre[0]) / gap
            raw_vy = (cy - self._last_confirmed_centre[1]) / gap
            raw_vx = max(-MAX_PREDICT_CENTRE_VEL, min(MAX_PREDICT_CENTRE_VEL, raw_vx))
            raw_vy = max(-MAX_PREDICT_CENTRE_VEL, min(MAX_PREDICT_CENTRE_VEL, raw_vy))
            if self._centre_velocity is None:
                self._centre_velocity = (raw_vx, raw_vy)
            else:
                a = VEL_EMA
                self._centre_velocity = (
                    a * raw_vx + (1.0 - a) * self._centre_velocity[0],
                    a * raw_vy + (1.0 - a) * self._centre_velocity[1],
                )
        else:
            # Fresh lock or transition out of a non-TRACKING state — no
            # trustworthy velocity yet.
            self._centre_velocity = None

        self._last_confirmed_centre = (cx, cy)
        self._last_confirmed_size   = (bw, bh)
        self._last_confirm_frame_no = self._frame_no
        self._last_box              = box

    def _extrapolate(self, shape: tuple) -> tuple[float, float, float, float]:
        """
        Predict the primary box on a skipped frame using the last confirmed
        centre + smoothed velocity, holding the box size fixed.  The centre is
        clamped to the frame so the marker can't fly off-screen.
        """
        h, w = shape[:2]
        cx0, cy0 = self._last_confirmed_centre
        if self._centre_velocity is None:
            cx, cy = cx0, cy0
        else:
            k  = self._frame_no - self._last_confirm_frame_no
            cx = cx0 + self._centre_velocity[0] * k
            cy = cy0 + self._centre_velocity[1] * k
        cx = min(max(cx, 0.0), w - 1.0)
        cy = min(max(cy, 0.0), h - 1.0)
        bw, bh = self._last_confirmed_size
        return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)

    def _is_nano_valid(self, box: tuple, shape: tuple) -> bool:
        """
        Validity guard: reject a NanoTrack output if the box has grown/
        shrunk dramatically or drifted outside the frame.
        """
        h, w  = shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        ratio = (bw * bh) / self._coast_init_area
        if ratio > BOX_AREA_RATIO_MAX or ratio < BOX_AREA_RATIO_MIN:
            print(f"[NanoTrack] area ratio {ratio:.2f} out of bounds "
                  f"[{BOX_AREA_RATIO_MIN}, {BOX_AREA_RATIO_MAX}] "
                  f"— invalidating coast.")
            return False
        cx, cy = _centre(box)
        if not (0 <= cx < w and 0 <= cy < h):
            print("[NanoTrack] box centre outside frame — invalidating coast.")
            return False
        return True

    def _reinit_nano(self, frame: np.ndarray, box: tuple) -> None:
        """
        (Re-)initialise NanoTrack on the full frame with `box` as the
        initial ROI.  NanoTrack expects (x, y, w, h) in integer pixels.
        """
        self._nano = make_nano_tracker(self._backbone_path,
                                       self._neckhead_path)
        xywh = _tlbr_to_xywh(box)
        self._nano.init(frame, xywh)

    def _update_nano(
        self, frame: np.ndarray
    ) -> tuple[bool, tuple[float, float, float, float]]:
        """
        Run one NanoTrack update step on the full frame.
        Returns (ok, (x1, y1, x2, y2)).
        """
        if self._nano is None:
            return False, (0.0, 0.0, 0.0, 0.0)
        ok, xywh = self._nano.update(frame)
        if not ok:
            return False, (0.0, 0.0, 0.0, 0.0)
        return True, _xywh_to_tlbr(xywh)


# ---------------------------------------------------------------------------
# Adaptive frame-skip controller
# ---------------------------------------------------------------------------

class AdaptiveSkipController:
    """
    Tracks rolling SAHI latency and decides whether to run a full
    detection pass on the current frame.

    When latency exceeds SAHI_LATENCY_SKIP_THRESH every other frame is
    skipped; OC-SORT and NanoTrack still run every frame using the last
    detection result, so the display stays smooth.

    Set SAHI_LATENCY_SKIP_THRESH = 0 to disable skipping entirely.
    """
    _WINDOW = 10

    def __init__(self) -> None:
        self._latencies: deque[float] = deque(maxlen=self._WINDOW)
        self._frame_idx: int          = 0

    def record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)

    def should_detect(self) -> bool:
        self._frame_idx += 1
        if SAHI_LATENCY_SKIP_THRESH <= 0 or len(self._latencies) < self._WINDOW:
            return True
        avg = sum(self._latencies) / len(self._latencies)
        if avg > SAHI_LATENCY_SKIP_THRESH:
            return (self._frame_idx % 2) == 0
        return True

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return 1000.0 * sum(self._latencies) / len(self._latencies)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_COL_TRACK    = (0,   255,   0)
_COL_TRACKING = (0,   255, 255)
_COL_COASTING = (0,   180, 255)
_COL_LOST     = (60,  60,  200)
_COL_FPS      = (0,   200, 255)
_COL_STATE    = (255, 255, 255)


def draw_all_tracks(frame: np.ndarray, tracks: list[dict],
                    primary_id: int | None) -> None:
    for t in tracks:
        if t["track_id"] == primary_id:
            continue
        x1, y1, x2, y2 = (int(v) for v in t["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COL_TRACK, 1)
        score_str = f"{t['score']:.2f}" if t["score"] is not None else ""
        cv2.putText(frame, f"ID{t['track_id']} {score_str}",
                    (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COL_TRACK, 1)


def draw_primary_target(frame: np.ndarray, box: tuple | None,
                        state: TrackState, primary_id: int | None) -> None:
    if box is None:
        return
    col = _COL_TRACKING if state == TrackState.TRACKING else _COL_COASTING
    x1, y1, x2, y2 = (int(v) for v in box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    arm = 12
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), col, 2)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), col, 2)
    cv2.circle(frame, (cx, cy), 4, col, -1)
    label = f"TARGET ID{primary_id}" if primary_id is not None else "TARGET"
    cv2.putText(frame, label, (x1, max(y1 - 10, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def draw_hud(frame: np.ndarray, fps: float, state: TrackState,
             n_tracks: int, skip_ctrl: AdaptiveSkipController,
             skipped: bool) -> None:
    h, _ = frame.shape[:2]
    state_col = {
        TrackState.TRACKING: _COL_TRACKING,
        TrackState.COASTING: _COL_COASTING,
        TrackState.LOST:     _COL_LOST,
    }[state]
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, _COL_FPS,   2)
    cv2.putText(frame, f"STATE: {state.name}",
                (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_col,  2)
    cv2.putText(frame, f"TRACKS: {n_tracks}",
                (10, 84),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COL_STATE, 1)
    lat_str  = f"DET: {skip_ctrl.avg_latency_ms:.0f} ms"
    skip_str = "  [SKIP]" if skipped else ""
    cv2.putText(frame, lat_str + skip_str,
                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_STATE, 1)
    cv2.putText(frame, "Q: quit   R: reset primary target",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Runtime-tunable config surface
# ---------------------------------------------------------------------------
#
# The original script mutated module globals from argparse.  The engine keeps
# that mechanism (so the tracking maths is byte-for-byte the same) but routes
# every live change through set_param(), which updates the relevant global and,
# where needed, the already-constructed objects.

def _apply_param(name: str, value, model, ocsort) -> tuple[bool, str]:
    """Apply one tunable. Returns (ok, human_message)."""
    global CONF_THRES, OCS_DET_THRESH, AUTO_SELECT_MIN_CONF
    global SAHI_LATENCY_SKIP_THRESH, PREDICT_ON_SKIP, AUTO_SWITCH_ON_LOSS

    if name == "conf_thres":
        v = float(value)
        if not (0.0 < v < 1.0):
            return False, "conf_thres must be in (0, 1)"
        CONF_THRES = OCS_DET_THRESH = AUTO_SELECT_MIN_CONF = v
        # The TRT engine's threshold is baked at build for the .engine, but the
        # python wrapper filters on it too; update the attribute if present.
        for attr in ("confidence_threshold", "conf_threshold"):
            if hasattr(model, attr):
                try:
                    setattr(model, attr, v)
                except Exception:
                    pass
        if hasattr(ocsort, "det_thresh"):
            try:
                ocsort.det_thresh = v
            except Exception:
                pass
        return True, f"conf_thres = {v:.2f}"

    if name == "latency_skip_ms":
        v = max(0.0, float(value))
        SAHI_LATENCY_SKIP_THRESH = v / 1000.0
        return True, f"latency_skip = {v:.0f} ms" + (" (disabled)" if v == 0 else "")

    if name == "predict_on_skip":
        PREDICT_ON_SKIP = bool(value)
        return True, f"predict_on_skip = {PREDICT_ON_SKIP}"

    if name == "auto_switch_on_loss":
        AUTO_SWITCH_ON_LOSS = bool(value)
        return True, f"auto_switch_on_loss = {AUTO_SWITCH_ON_LOSS}"

    if name == "jpeg_quality":
        return False, "jpeg_quality is handled by the server, not the engine"

    return False, f"unknown parameter: {name}"


# ---------------------------------------------------------------------------
# TrackerEngine — headless, frame-in / (annotated-frame + telemetry)-out
# ---------------------------------------------------------------------------

class TrackerEngine:
    """
    Wraps the full SAHI -> OC-SORT -> NanoTrack pipeline behind a clean,
    headless interface so it can be driven by a network server instead of a
    local OpenCV window.

    Typical use (server side)::

        engine = TrackerEngine(engine_path, backbone, neckhead, source="jetson")
        engine.open()
        while engine.running:
            annotated, telemetry = engine.process_next()
            if annotated is None:
                break
            ...stream annotated + telemetry...

    All operator controls (select_target, reset_target, set_param,
    set_source, pause) are safe to call from another thread; they are applied
    at the top of the next process_next() call under a lock.
    """

    def __init__(self, engine_path: str, backbone: str, neckhead: str,
                 source=None, draw_hud_on_frame: bool = False) -> None:
        self._engine_path = engine_path
        self._backbone    = backbone
        self._neckhead    = neckhead
        self._source      = source
        self._draw_hud    = draw_hud_on_frame

        self._model     = None
        self._ocsort    = None
        self._primary   = None
        self._skip_ctrl = None
        self._cap       = None
        self._is_file   = False

        self._orig_w = 0
        self._orig_h = 0
        self._fps_val = float(CAM_FPS)
        self._prev_t  = time.perf_counter()
        self._frame_count = 0
        self._last_dets = np.empty((0, 6), dtype=np.float32)
        self._last_display_tracks: list[dict] = []
        self._pending = deque()

        # On-target tracking (spec: laser visible on drone >= 80% of the time).
        # "On target" = TRACKING or COASTING with a valid box, i.e. the laser
        # would be pointed at the drone. Rolling window gives a live readout;
        # session counters give the whole-run figure for the demo / report.
        # Counting starts only once a target has first been acquired, so idle
        # time before the operator locks on doesn't drag the percentage down.
        self._ontgt_window: deque[bool] = deque(maxlen=ONTARGET_WINDOW)
        self._ontgt_session_on = 0
        self._ontgt_session_total = 0
        self._ontgt_counting = False

        self._paused = False
        self.running = False

        # Pending operator commands, drained each frame under the lock.
        import threading
        self._lock = threading.Lock()
        self._cmd_queue: list[tuple[str, dict]] = []
        # Latest tracks visible to the operator (for select-by-id validation).
        self._visible_tracks: list[dict] = []
        # Reject masks: normalised (fx1,fy1,fx2,fy2) zones; detections whose
        # centre falls inside any are dropped before tracking. Per-scene, in
        # memory only.
        self._masks: list[tuple] = []
        # Log of recent events (param changes, target switches) to surface.
        self._events: deque[str] = deque(maxlen=50)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> dict:
        """Load models, open capture, read first frame. Returns capability dict."""
        print(f"[engine] loading SAHI-TRT model: {self._engine_path}")
        self._model     = load_sahi_model(self._engine_path)
        self._ocsort    = make_ocsort_tracker()
        self._primary   = PrimaryTargetTracker(self._backbone, self._neckhead)
        self._skip_ctrl = AdaptiveSkipController()

        self._cap, self._is_file = open_capture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open source: {self._source!r}")
        ret, probe = self._cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame.")
        self._orig_h, self._orig_w = probe.shape[:2]
        n_slices = estimate_sahi_slices(self._orig_w, self._orig_h)
        if self._is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._pending.append(probe)
        self._fps_val = nominal_fps(self._source)
        self._prev_t  = time.perf_counter()
        self.running  = True
        return {
            "width": self._orig_w,
            "height": self._orig_h,
            "source": str(self._source),
            "is_file": self._is_file,
            "sahi_slices": n_slices,
            "sahi_max_batch": SAHI_MAX_BATCH,
            "class_name": CLASS_NAME,
        }

    def close(self) -> None:
        self.running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── operator controls (thread-safe) ─────────────────────────────────────

    def select_target(self, track_id: int) -> None:
        with self._lock:
            self._cmd_queue.append(("select", {"track_id": int(track_id)}))

    def reset_target(self) -> None:
        with self._lock:
            self._cmd_queue.append(("reset", {}))

    def set_param(self, name: str, value) -> None:
        with self._lock:
            self._cmd_queue.append(("param", {"name": name, "value": value}))

    def set_source(self, source) -> None:
        with self._lock:
            self._cmd_queue.append(("source", {"source": source}))

    def set_paused(self, value: bool) -> None:
        with self._lock:
            self._paused = bool(value)

    def set_masks(self, masks) -> None:
        """Replace the reject-mask set. masks: list of [fx1,fy1,fx2,fy2]."""
        clean = []
        for m in masks or []:
            try:
                fx1, fy1, fx2, fy2 = (float(v) for v in m[:4])
                clean.append((min(fx1, fx2), min(fy1, fy2),
                              max(fx1, fx2), max(fy1, fy2)))
            except (TypeError, ValueError):
                continue
        with self._lock:
            self._masks = clean
        self._log_event(f"reject masks set: {len(clean)}")

    def pop_events(self) -> list[str]:
        with self._lock:
            ev = list(self._events)
            self._events.clear()
            return ev

    def _log_event(self, msg: str) -> None:
        self._events.append(msg)
        print(f"[engine] {msg}")

    # ── command drain ────────────────────────────────────────────────────────

    def _drain_commands(self) -> None:
        with self._lock:
            queued = self._cmd_queue
            self._cmd_queue = []
        for kind, args in queued:
            if kind == "select":
                ok = self._primary.select(args["track_id"], self._visible_tracks)
                self._log_event(
                    f"target {'locked' if ok else 'select FAILED (gone)'}: "
                    f"ID{args['track_id']}")
            elif kind == "reset":
                self._primary.reset()
                self._log_event("primary target reset")
            elif kind == "param":
                ok, msg = _apply_param(args["name"], args["value"],
                                       self._model, self._ocsort)
                self._log_event(msg)
            elif kind == "source":
                self._switch_source(args["source"])

    def _switch_source(self, source) -> None:
        try:
            new_cap, is_file = open_capture(source)
            ret, probe = new_cap.read()
            if not ret:
                new_cap.release()
                self._log_event(f"source switch FAILED: {source!r}")
                return
        except Exception as exc:
            self._log_event(f"source switch error: {exc}")
            return
        if self._cap is not None:
            self._cap.release()
        self._cap = new_cap
        self._is_file = is_file
        self._source = source
        self._orig_h, self._orig_w = probe.shape[:2]
        if is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._pending.clear()
        self._pending.append(probe)
        self._primary.reset()
        self._last_display_tracks = []
        self._fps_val = nominal_fps(source)
        self._log_event(f"source -> {source} ({self._orig_w}x{self._orig_h})")

    # ── reject-mask filtering ────────────────────────────────────────────────

    def _apply_conf_gate(self, dets: np.ndarray) -> np.ndarray:
        """
        Keep only detections at or above the live CONF_THRES. Makes the conf
        slider authoritative regardless of the engine's baked-in threshold.
        Detection rows are [x1, y1, x2, y2, score, cls]; score is column 4.
        """
        if dets is None or len(dets) == 0:
            return dets
        keep = dets[dets[:, 4] >= CONF_THRES]
        return keep if len(keep) else np.empty((0, 6), dtype=np.float32)

    def _apply_masks(self, dets: np.ndarray) -> np.ndarray:
        """
        Drop detections whose box centre lies inside any reject mask. Masks are
        normalised fractions of the frame, so they hold across resolutions.
        """
        with self._lock:
            masks = self._masks
        if not masks or dets is None or len(dets) == 0:
            return dets
        w = max(1, self._orig_w)
        h = max(1, self._orig_h)
        keep = []
        for d in dets:
            cx = ((d[0] + d[2]) / 2.0) / w
            cy = ((d[1] + d[3]) / 2.0) / h
            masked = any(mx1 <= cx <= mx2 and my1 <= cy <= my2
                         for (mx1, my1, mx2, my2) in masks)
            if not masked:
                keep.append(d)
        if not keep:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(keep, dtype=np.float32)

    # ── live-camera reconnect (CSI ribbon glitch / USB drop) ─────────────────

    def _reconnect_camera(self, max_attempts: int = 5,
                          backoff_s: float = 0.5):
        """
        Attempt to reopen a live camera after a read failure. Returns the first
        good frame on success, or None after `max_attempts` failed reopens.
        File sources never call this (they loop / EOF instead).
        """
        self._log_event("camera read failed — attempting to reconnect")
        for attempt in range(1, max_attempts + 1):
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            time.sleep(backoff_s)
            try:
                new_cap, _ = open_capture(self._source)
            except Exception as exc:
                self._log_event(f"reconnect attempt {attempt} error: {exc}")
                continue
            if new_cap.isOpened():
                ret, frame = new_cap.read()
                if ret:
                    self._cap = new_cap
                    self._log_event(
                        f"camera reconnected on attempt {attempt}")
                    return frame
                new_cap.release()
            self._log_event(f"reconnect attempt {attempt}/{max_attempts} failed")
        self._log_event("camera reconnect gave up — ending run")
        return None

    # ── the per-frame step (the original main-loop body, headless) ───────────

    def process_next(self):
        """
        Process one frame.  Returns (annotated_frame, telemetry_dict).
        Returns (None, None) when the source is exhausted / closed.
        """
        if not self.running or self._cap is None:
            return None, None

        self._drain_commands()

        # FPS EMA, identical to the original loop.
        now = time.perf_counter()
        dt = now - self._prev_t
        self._prev_t = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps_val = (FPS_EMA_ALPHA * inst
                             + (1.0 - FPS_EMA_ALPHA) * self._fps_val)

        if self._paused:
            # Hold on the last frame without advancing capture.
            time.sleep(0.01)
            blank = np.zeros((self._orig_h, self._orig_w, 3), dtype=np.uint8)
            return blank, self._telemetry(TrackState.LOST, 0, True, paused=True)

        if self._pending:
            frame = self._pending.popleft()
        else:
            ret, frame = self._cap.read()
            if not ret:
                if self._is_file:
                    # Loop video files so a demo clip runs continuously.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret:
                        return None, None
                else:
                    # Live camera hiccup (CSI ribbon glitch, USB drop). Try to
                    # reopen rather than killing the whole run — important for a
                    # demo where a momentary camera stall shouldn't end tracking.
                    frame = self._reconnect_camera()
                    if frame is None:
                        return None, None

        self._orig_h, self._orig_w = frame.shape[:2]
        self._frame_count += 1

        # 1. SAHI + TRT detection (adaptive skip) — unchanged logic.
        run_det = self._skip_ctrl.should_detect()
        if run_det:
            t0 = time.perf_counter()
            self._last_dets = run_sahi(self._model, frame)
            self._skip_ctrl.record_latency(time.perf_counter() - t0)
            # Explicit confidence gate on the LIVE threshold. The TRT engine's
            # confidence is baked in at build time and OC-SORT's internal
            # det_thresh may not update on the running object, so we filter here
            # to make the command-center conf slider authoritative every frame.
            self._last_dets = self._apply_conf_gate(self._last_dets)
            # Drop detections whose centre falls in an operator reject mask
            # (fixed-background false positives). Applied here, before OC-SORT,
            # so a masked blob never becomes a track or coasts.
            self._last_dets = self._apply_masks(self._last_dets)
            detections = self._last_dets
        elif PREDICT_ON_SKIP:
            detections = np.empty((0, 6), dtype=np.float32)
        else:
            detections = self._last_dets

        # 2. OC-SORT
        raw_tracks = update_ocsort_compat(self._ocsort, detections,
                                          self._orig_h, self._orig_w)
        ocsort_tracks = parse_ocsort_outputs(raw_tracks, self._orig_h, self._orig_w)
        attach_detection_scores(ocsort_tracks, detections)

        # Expose tracks so an operator's select-by-id is validated against
        # what is actually on screen right now.
        with self._lock:
            self._visible_tracks = ocsort_tracks

        # 3. Primary-target tracker
        target_box, track_state = self._primary.update(
            frame, ocsort_tracks, detection_ran=run_det)

        # On-target accounting: start counting once a target has ever been
        # acquired; thereafter every frame counts, and "on target" means the
        # laser would be pointed at the drone (TRACKING or COASTING + a box).
        on_target = (track_state in (TrackState.TRACKING, TrackState.COASTING)
                     and target_box is not None)
        if on_target:
            self._ontgt_counting = True
        if self._ontgt_counting:
            self._ontgt_window.append(on_target)
            self._ontgt_session_total += 1
            if on_target:
                self._ontgt_session_on += 1

        # 4. Draw boxes + crosshair (HUD optional; the GUI renders its own).
        if run_det or not PREDICT_ON_SKIP:
            display_tracks = ocsort_tracks
            self._last_display_tracks = ocsort_tracks
        else:
            display_tracks = self._last_display_tracks

        # draw_all_tracks excludes the primary's *raw* box from the secondary
        # list; the primary marker shows the *stable* operator-facing id.
        draw_all_tracks(frame, display_tracks, self._primary.raw_primary_id)
        draw_primary_target(frame, target_box, track_state,
                            self._primary.primary_id)
        if self._draw_hud:
            draw_hud(frame, self._fps_val, track_state,
                     len(display_tracks), self._skip_ctrl, skipped=not run_det)

        telem = self._telemetry(track_state, len(display_tracks),
                                not run_det, paused=False,
                                tracks=display_tracks,
                                target_box=target_box)
        return frame, telem

    # ── telemetry assembly ───────────────────────────────────────────────────

    def _telemetry(self, state, n_tracks, skipped, paused,
                   tracks=None, target_box=None) -> dict:
        track_list = []
        if tracks:
            for t in tracks:
                track_list.append({
                    "id": t["track_id"],
                    "box": [round(float(v), 1) for v in t["box"]],
                    "score": (round(float(t["score"]), 3)
                              if t["score"] is not None else None),
                })
        # On-target percentages (rolling = recent, session = whole run).
        if self._ontgt_window:
            ontgt_rolling = round(
                100.0 * sum(self._ontgt_window) / len(self._ontgt_window), 1)
        else:
            ontgt_rolling = None
        if self._ontgt_session_total > 0:
            ontgt_session = round(
                100.0 * self._ontgt_session_on / self._ontgt_session_total, 1)
        else:
            ontgt_session = None
        return {
            "fps": round(self._fps_val, 2),
            "det_latency_ms": round(self._skip_ctrl.avg_latency_ms, 1),
            "width": self._orig_w,
            "height": self._orig_h,
            "state": state.name,
            "n_tracks": n_tracks,
            "frame_no": self._frame_count,
            "skipped": bool(skipped),
            "paused": bool(paused),
            "primary_id": self._primary.primary_id if self._primary else None,
            "raw_primary_id": (self._primary.raw_primary_id
                               if self._primary else None),
            "primary_box": ([round(float(v), 1) for v in target_box]
                            if target_box is not None else None),
            "on_target_pct": ontgt_rolling,        # rolling window
            "on_target_session_pct": ontgt_session,  # whole run
            "tracks": track_list,
            # Live values of the tunables so the GUI can reflect server truth.
            "params": {
                "conf_thres": round(CONF_THRES, 3),
                "latency_skip_ms": round(SAHI_LATENCY_SKIP_THRESH * 1000.0, 1),
                "predict_on_skip": PREDICT_ON_SKIP,
                "auto_switch_on_loss": AUTO_SWITCH_ON_LOSS,
            },
            "events": self.pop_events(),
        }
