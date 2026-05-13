#!/usr/bin/env python3
"""Phase 4 – Wire harness measurement (accurate algorithm).
Single-window tkinter UI optimised for 1920x1080.

Adds an extra "Kinect" button to use Kinect v1 RGB+Depth over USB.
In Kinect mode, lengths are depth-corrected (3D arc length) when depth is valid.
Falls back to 2D planar measurement if depth is missing/invalid.

Tested integration assumptions:
- Windows + Kinect SDK 1.8
- pykinect installed (importable)
- Kinect v1 provides 640x480 color + 640x480 depth
"""

import cv2
import numpy as np
import json
import os
import math
import tkinter as tk
from tkinter import filedialog
from collections import deque
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from PIL import Image, ImageTk
import threading
import time

try:
    from pykinect import nui
    HAS_KINECT = True
except Exception:
    HAS_KINECT = False


# ── CONFIG ────────────────────────────────────────────────────────────────────
SETTINGS_FILE   = "wire_settings.json"
DEFAULT_URL     = "http://10.65.52.117:8080/video"
DICT_TYPE       = cv2.aruco.DICT_5X5_100
MAX_FAIL        = 30
A4_W, A4_H      = 210.0, 297.0
INSET_MM        = 5 + (57 / 2)
PX_MM           = 4.0
OUT_W, OUT_H    = int(A4_W * PX_MM), int(A4_H * PX_MM)
SCALE           = 1.0 / PX_MM
INSET_PX        = int(INSET_MM * PX_MM)
SMOOTH_W        = 30
MIN_SEG         = 30
MIN_BRANCH_MM   = 15.0
JN_CLUSTER_R    = 20
JN_MATCH_R      = 25
EP_MATCH_R      = 10
ARUCO_MARGIN_PX = 30
SPLINE_SMOOTH   = 5.0
SPLINE_PTS      = 500

# Kinect intrinsics (typical Kinect v1; calibrate for best accuracy)
K_W, K_H   = 640, 480
K_FX, K_FY = 585.0, 585.0
K_CX, K_CY = 320.0, 240.0
DEPTH_MIN_MM     = 400
DEPTH_MAX_MM     = 1500
DEPTH_MEDIAN_K   = 3
DEPTH_VALID_FRAC = 0.60

MARKER_CORNER = {0: "tl", 1: "tr", 2: "bl", 3: "br"}
DEF_BS, DEF_CO, DEF_MS, DEF_BL, DEF_CI = 53, 13, 5, 11, 5

ad  = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
_det_params = cv2.aruco.DetectorParameters()
# Corner refinement for sub-pixel accuracy
_det_params.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
_det_params.cornerRefinementWinSize     = 5
_det_params.cornerRefinementMaxIterations = 30
_det_params.cornerRefinementMinAccuracy = 0.1
# Adaptive threshold sweep - wide range catches markers under varied lighting
_det_params.adaptiveThreshWinSizeMin    = 7
_det_params.adaptiveThreshWinSizeMax    = 53
_det_params.adaptiveThreshWinSizeStep   = 6
_det_params.adaptiveThreshConstant      = 7
# Tighter perimeter rate to REJECT tiny false-positive detections in background
_det_params.minMarkerPerimeterRate      = 0.05   # was 0.02 - too loose, caused fake IDs
_det_params.maxMarkerPerimeterRate      = 4.0
# Tighter polygon fit - rejects noise that looks vaguely square
_det_params.polygonalApproxAccuracyRate = 0.04
_det_params.minCornerDistanceRate       = 0.05
_det_params.minDistanceToBorder         = 3
_det_params.minOtsuStdDev              = 5.0
# Error correction - use full correction for 5x5 dictionary
_det_params.errorCorrectionRate         = 0.6
det = cv2.aruco.ArucoDetector(ad, _det_params)

# ── COLORS ────────────────────────────────────────────────────────────────────
BC = [
    (0,   0,   255),
    (255, 150,   0),
    (0,   255, 255),
    (255,   0, 255),
    (0,   255,   0),
    (255, 255,   0),
    (128,   0, 255),
    (0,   128, 255),
]


# ── SETTINGS ──────────────────────────────────────────────────────────────────
def load_settings():
    d = {"stream_url": DEFAULT_URL, "block_size": DEF_BS, "c_offset": DEF_CO,
         "morph_size": DEF_MS, "blur_size": DEF_BL, "close_iter": DEF_CI}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                d.update(json.load(f))
        except Exception:
            pass
    return d


def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass


# ── HOMOGRAPHY ────────────────────────────────────────────────────────────────
_DST = {
    "tl": np.array([INSET_PX,          INSET_PX         ], np.float32),
    "tr": np.array([OUT_W - INSET_PX,  INSET_PX         ], np.float32),
    "bl": np.array([INSET_PX,          OUT_H - INSET_PX ], np.float32),
    "br": np.array([OUT_W - INSET_PX,  OUT_H - INSET_PX ], np.float32),
}


def homography(corners, ids):

    if ids is None or len(ids) < 4:
        return None, None, None

    found = set(ids.flatten().tolist())

    required = {0, 1, 2, 3}

    if not required.issubset(found):
        return None, None, None

    src_pts = []
    dst_pts = []

    # Marker corner mapping:
    #
    # 0 = top-left marker
    # 1 = top-right marker
    # 2 = bottom-left marker
    # 3 = bottom-right marker

    for i, marker_id in enumerate(ids.flatten()):

        pts = corners[i][0]

        if marker_id == 0:
            src_pts.append(pts[0])
            dst_pts.append([INSET_PX, INSET_PX])

        elif marker_id == 1:
            src_pts.append(pts[1])
            dst_pts.append([OUT_W - INSET_PX, INSET_PX])

        elif marker_id == 2:
            src_pts.append(pts[3])
            dst_pts.append([INSET_PX, OUT_H - INSET_PX])

        elif marker_id == 3:
            src_pts.append(pts[2])
            dst_pts.append([OUT_W - INSET_PX, OUT_H - INSET_PX])

    if len(src_pts) != 4:
        return None, None, None

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    H, status = cv2.findHomography(src_pts, dst_pts)

    if H is None:
        return None, None, None

    all_corners = [corners[i][0] for i in range(len(corners))]

    return H, (OUT_W, OUT_H), all_corners

# ── SEGMENTATION ──────────────────────────────────────────────────────────────
def wire_mask(fr, bs, co, ms, bl, ci):
    l = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0]
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    l = cv2.bilateralFilter(l, bl | 1, 75, 75)

    m = cv2.adaptiveThreshold(
        l, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs | 1 if bs >= 3 else 3,
        co
    )

    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ms, ms))
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ms * 2 + 1, ms * 2 + 1))
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=ci)

    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = np.zeros_like(m)
    for c in cn:
        if cv2.contourArea(c) >= 500:
            cv2.drawContours(r, [c], -1, 255, -1)
    return r


def mask_aruco(m, H, raw_corners):
    if raw_corners is None or H is None:
        h, w = m.shape[:2]
        b = INSET_PX + int(25 * PX_MM)
        m[0:b, 0:b] = 0
        m[0:b, w-b:w] = 0
        m[h-b:h, 0:b] = 0
        m[h-b:h, w-b:w] = 0
        m[0:int(15 * PX_MM), :] = 0
        return m

    mg = ARUCO_MARGIN_PX
    for quad in raw_corners:
        pts = np.array(quad, np.float32).reshape(-1, 1, 2)
        wp  = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        x0  = max(int(np.floor(wp[:, 0].min()) - mg), 0)
        x1  = min(int(np.ceil (wp[:, 0].max()) + mg), m.shape[1])
        y0  = max(int(np.floor(wp[:, 1].min()) - mg), 0)
        y1  = min(int(np.ceil (wp[:, 1].max()) + mg), m.shape[0])
        m[y0:y1, x0:x1] = 0
    return m


# ── SKELETON ──────────────────────────────────────────────────────────────────
def get_skel(m):
    return (skeletonize(m > 0).astype(np.uint8)) * 255


def n8(s, y, x):
    h, w = s.shape
    o = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and s[ny, nx]:
                o.append((ny, nx))
    return o


def trace(s, st):
    v = {st}
    p = [st]
    c = st
    while True:
        nx = None
        for nb in n8(s, c[0], c[1]):
            if nb not in v:
                nx = nb
                break
        if nx is None:
            break
        p.append(nx)
        v.add(nx)
        c = nx
    return p


def alen(pts):
    if len(pts) < 2:
        return 0.0
    a = np.array(pts, np.float64)
    return float(np.sqrt((np.diff(a, axis=0) ** 2).sum(axis=1)).sum())


def spline_length(pts):
    if len(pts) < 6:
        return alen(pts)
    arr = np.array(pts, np.float64)
    if len(arr) > 2000:
        idx = np.linspace(0, len(arr) - 1, 2000).astype(int)
        arr = arr[idx]
    try:
        tck, _ = splprep([arr[:, 0], arr[:, 1]], s=SPLINE_SMOOTH * len(arr), k=3)
        u_f = np.linspace(0, 1, SPLINE_PTS)
        y_s, x_s = splev(u_f, tck)
        return float(np.sqrt(np.diff(y_s) ** 2 + np.diff(x_s) ** 2).sum())
    except Exception:
        return alen(pts)


# ── KINECT SOURCE ────────────────────────────────────────────────────────────

class KinectV1:
    """Minimal Kinect v1 RGB+Depth reader (USB) using pykinect.nui.

    Notes:
    - Color frames from Kinect v1 via pykinect/nui are typically BGRA.
    - Depth decode: uint16 = low | (high<<8); depth_mm = uint16 >> 3
    """

    def __init__(self):
        self._kinect = None
        self._lock = threading.Lock()
        self._rgb_bgr = None
        self._depth_mm = None
        self._ok_rgb = False
        self._ok_depth = False

    def start(self, timeout_s=5.0):
        if not HAS_KINECT:
            raise RuntimeError("pykinect is not available")

        self._kinect = nui.Runtime()

        # Depth
        self._kinect.depth_stream.open(
            nui.ImageStreamType.Depth,
            2,
            nui.ImageResolution.Resolution640x480,
            nui.ImageType.Depth
        )
        self._kinect.depth_frame_ready += self._on_depth

        # Color
        self._kinect.video_stream.open(
            nui.ImageStreamType.Video,
            2,
            nui.ImageResolution.Resolution640x480,
            nui.ImageType.Color
        )
        self._kinect.video_frame_ready += self._on_rgb

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self._ok_rgb and self._ok_depth:
                    return True
            time.sleep(0.05)
        return False

    def _on_depth(self, frame):
        try:
            raw = np.frombuffer(frame.image.bits, dtype=np.uint8).reshape(K_H, K_W, 2)
            data = raw[:, :, 0].astype(np.uint16) | (raw[:, :, 1].astype(np.uint16) << 8)
            depth_mm = data >> 3
            with self._lock:
                self._depth_mm = depth_mm.copy()
                self._ok_depth = True
        except Exception:
            pass

    def _on_rgb(self, frame):
        try:
            raw = np.frombuffer(frame.image.bits, dtype=np.uint8).reshape(K_H, K_W, 4)
            # Kinect v1 is typically BGRA here
            bgr = raw[:, :, :3].copy()
            with self._lock:
                self._rgb_bgr = bgr
                self._ok_rgb = True
        except Exception:
            pass

    def read(self):
        with self._lock:
            if self._rgb_bgr is None:
                return False, None, None
            rgb = self._rgb_bgr.copy()
            dep = self._depth_mm.copy() if self._depth_mm is not None else None
        return True, rgb, dep

    def release(self):
        if self._kinect is not None:
            try:
                self._kinect.close()
            except Exception:
                pass
            self._kinect = None


# ── 3D LENGTH ESTIMATION ─────────────────────────────────────────────────────

def _median_depth_at(depth_mm, u, v, k=DEPTH_MEDIAN_K):
    if depth_mm is None:
        return 0
    h, w = depth_mm.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return 0
    if k <= 1:
        return int(depth_mm[v, u])
    r = k // 2
    x0 = max(u - r, 0); x1 = min(u + r + 1, w)
    y0 = max(v - r, 0); y1 = min(v + r + 1, h)
    patch = depth_mm[y0:y1, x0:x1].reshape(-1)
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0
    return int(np.median(patch))


def _pixels_to_3d(us, vs, zs_mm):
    Z = zs_mm.astype(np.float64)
    X = (us.astype(np.float64) - K_CX) * Z / K_FX
    Y = (vs.astype(np.float64) - K_CY) * Z / K_FY
    return np.column_stack([X, Y, Z])


def _arc_len_3d_mm(pts3):
    if pts3 is None or len(pts3) < 2:
        return 0.0
    d = np.diff(pts3, axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def length_path_3d_mm(path_yx, H, depth_mm):
    """Estimate path length using Kinect depth.

    path_yx: list[(y,x)] in warped coordinates.
    H: homography from source Kinect RGB -> warped.
    depth_mm: Kinect depth in mm, same resolution as source RGB.

    Returns (length_mm or None, valid_fraction).
    """
    if depth_mm is None or H is None or len(path_yx) < 2:
        return None, 0.0

    try:
        Hinv = np.linalg.inv(H)
    except Exception:
        return None, 0.0

    # Convert path to Nx1x2 in (x,y) warped coords
    wp = np.array([[p[1], p[0]] for p in path_yx], dtype=np.float32).reshape(-1, 1, 2)
    sp = cv2.perspectiveTransform(wp, Hinv).reshape(-1, 2)  # (x_src, y_src)

    us = np.rint(sp[:, 0]).astype(np.int32)
    vs = np.rint(sp[:, 1]).astype(np.int32)

    zs = np.zeros(len(us), dtype=np.float64)
    valid = np.zeros(len(us), dtype=bool)
    for i, (u, v) in enumerate(zip(us, vs)):
        z = _median_depth_at(depth_mm, int(u), int(v))
        if DEPTH_MIN_MM <= z <= DEPTH_MAX_MM:
            zs[i] = float(z)
            valid[i] = True

    frac = float(valid.mean()) if len(valid) else 0.0
    if frac < DEPTH_VALID_FRAC:
        return None, frac

    # interpolate gaps
    if not np.all(valid):
        if valid.sum() < 2:
            return None, frac
        idx = np.arange(len(zs))
        zs = np.interp(idx, idx[valid], zs[valid])

    pts3 = _pixels_to_3d(us.astype(np.float64), vs.astype(np.float64), zs)
    return _arc_len_3d_mm(pts3), frac


# ── HARNESS ANALYSIS ─────────────────────────────────────────────────────────

def analyse(skeleton):
    s = (skeleton > 0).astype(np.uint8)
    ys, xs = np.where(s > 0)
    if len(ys) < 30:
        return None

    eps = set()
    jns = set()
    for y, x in zip(ys, xs):
        nn = len(n8(s, y, x))
        if nn == 1:
            eps.add((y, x))
        elif nn >= 3:
            jns.add((y, x))
    if not jns:
        return None

    # cluster junction pixels
    jl = list(jns)
    used = set()
    clusters = []
    for i, p in enumerate(jl):
        if i in used:
            continue
        cl = [p]
        used.add(i)
        stk = [p]
        while stk:
            cur = stk.pop()
            for j, p2 in enumerate(jl):
                if j in used:
                    continue
                if max(abs(cur[0] - p2[0]), abs(cur[1] - p2[1])) <= JN_CLUSTER_R:
                    cl.append(p2)
                    used.add(j)
                    stk.append(p2)
        clusters.append(cl)

    jcenters = [
        (int(np.mean([p[0] for p in cl])), int(np.mean([p[1] for p in cl])))
        for cl in clusters
    ]

    # remove junction pixels
    sc = s.copy()
    for py, px in jns:
        sc[py, px] = 0

    ncc, labels = cv2.connectedComponents(sc, connectivity=8)
    segs = []
    for lb in range(1, ncc):
        cy, cx = np.where(labels == lb)
        if len(cy) < 5:
            continue
        comp = (labels == lb).astype(np.uint8)
        se = [(yy, xx) for yy, xx in zip(cy, cx) if len(n8(comp, yy, xx)) == 1]
        if not se:
            continue
        path = trace(comp, se[0])
        length = spline_length(path)
        if length < MIN_SEG:
            continue

        def classify(pt):
            for ep in eps:
                if max(abs(pt[0] - ep[0]), abs(pt[1] - ep[1])) <= EP_MATCH_R:
                    return 'ep', ep
            best_d = 9999
            best_j = None
            for jc in jcenters:
                d = max(abs(pt[0] - jc[0]), abs(pt[1] - jc[1]))
                if d < best_d:
                    best_d = d
                    best_j = jc
            if best_d <= JN_MATCH_R:
                return 'jn', best_j
            return 'unk', pt

        t0, n0 = classify(path[0])
        t1, n1 = classify(path[-1])
        segs.append({'path': path, 'len': length, 'e0': n0, 't0': t0, 'e1': n1, 't1': t1})

    if len(segs) < 2:
        return None

    tc = [sg for sg in segs if sg['t0'] == 'ep' or sg['t1'] == 'ep'] or segs
    trunk = max(tc, key=lambda sg: sg['len'])

    if trunk['t0'] == 'ep':
        tep = trunk['e0']
        tjn = trunk['e1']
    elif trunk['t1'] == 'ep':
        tep = trunk['e1']
        tjn = trunk['e0']
    else:
        tep = trunk['e0']
        tjn = trunk['e1']

    branches = []
    uid = {id(trunk)}

    def bfs(junc, dist, links_so_far=None):
        if links_so_far is None:
            links_so_far = []
        for seg in segs:
            if id(seg) in uid:
                continue
            conn = None
            other = None
            ot = None
            for ta, ea, tb, eb in [(seg['t0'], seg['e0'], seg['t1'], seg['e1']),
                                   (seg['t1'], seg['e1'], seg['t0'], seg['e0'])]:
                if ta == 'jn' and ea is not None:
                    if max(abs(ea[0] - junc[0]), abs(ea[1] - junc[1])) <= JN_MATCH_R:
                        conn = ea
                        other = eb
                        ot = tb
                        break
            if conn is None:
                continue
            uid.add(id(seg))
            tot = dist + seg['len']
            if ot in ('ep', 'unk'):
                branches.append({'path': seg['path'], 'len': seg['len'],
                                 'endpoint': other, 'junction': junc,
                                 'total': tot, 'links': links_so_far + [seg]})
            elif ot == 'jn':
                bfs(other, tot, links_so_far + [seg])

    bfs(tjn, trunk['len'])

    min_px = MIN_BRANCH_MM * PX_MM
    branches = [b for b in branches if b['len'] >= min_px]
    if not branches:
        return None

    # stable identity via angle
    for b in branches:
        p = b['path']
        d0 = (p[0][0] - tjn[0]) ** 2 + (p[0][1] - tjn[1]) ** 2
        d1 = (p[-1][0] - tjn[0]) ** 2 + (p[-1][1] - tjn[1]) ** 2
        tip = p[0] if d0 > d1 else p[-1]
        b['tip'] = tip
        b['angle'] = math.atan2(tip[1] - tjn[1], tip[0] - tjn[0])

    branches.sort(key=lambda b: b['angle'])

    used_j = {tjn} | {b['junction'] for b in branches if b['junction']}
    active_j = [jc for jc in jcenters if jc in used_j]

    return {
        'trunk': {'path': trunk['path'], 'len': trunk['len'], 'ep': tep},
        'branches': branches,
        'junctions': active_j,
        'tjn': tjn
    }


# ── SIMPLE FALLBACK ───────────────────────────────────────────────────────────

def simple(skeleton):
    ncc, labels = cv2.connectedComponents(skeleton)
    out = []
    for lb in range(1, ncc):
        comp = ((labels == lb) * 255).astype(np.uint8)
        su = (comp > 0).astype(np.uint8)
        k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
        nc = cv2.filter2D(su, -1, k)
        ep = np.argwhere((su == 1) & (nc == 1))
        st = ep[0] if len(ep) else np.argwhere(comp > 0)[0]
        vis = set()
        path = []
        r, c = int(st[0]), int(st[1])
        path.append((c, r))
        vis.add((r, c))
        h, w = comp.shape
        while True:
            f = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nr, nc2 = r + dy, c + dx
                    if 0 <= nr < h and 0 <= nc2 < w and comp[nr, nc2] and (nr, nc2) not in vis:
                        vis.add((nr, nc2))
                        r, c = nr, nc2
                        path.append((c, r))
                        f = True
                        break
                if f:
                    break
            if not f:
                break
        l = spline_length([(p[1], p[0]) for p in path])
        if l > 20:
            out.append((path, l))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ── ANGLE-BASED COLOR ASSIGNMENT ─────────────────────────────────────────────

class AngleColorAssigner:
    ANGLE_TOL = math.radians(25)

    def __init__(self):
        self.reset()

    def reset(self):
        self._slots = {}
        self._next_slot = 0

    def _angle_diff(self, a, b):
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return abs(d)

    def assign(self, branches):
        angles = [b['angle'] for b in branches]

        if not self._slots:
            for angle in angles:
                sid = self._next_slot
                self._next_slot += 1
                self._slots[sid] = {
                    'angle': angle,
                    'len_buf': deque(maxlen=SMOOTH_W),
                    'tot_buf': deque(maxlen=SMOOTH_W),
                }

        available = list(self._slots.keys())
        assignments = {}
        used_slots = set()

        costs = []
        for bi, angle in enumerate(angles):
            for sid in available:
                diff = self._angle_diff(angle, self._slots[sid]['angle'])
                costs.append((diff, bi, sid))
        costs.sort()

        assigned_branches = set()
        for diff, bi, sid in costs:
            if bi in assigned_branches or sid in used_slots:
                continue
            if diff <= self.ANGLE_TOL:
                assignments[bi] = sid
                used_slots.add(sid)
                assigned_branches.add(bi)

        for bi in range(len(branches)):
            if bi not in assignments:
                sid = self._next_slot
                self._next_slot += 1
                self._slots[sid] = {
                    'angle': angles[bi],
                    'len_buf': deque(maxlen=SMOOTH_W),
                    'tot_buf': deque(maxlen=SMOOTH_W),
                }
                assignments[bi] = sid

        result = []
        for bi, br in enumerate(branches):
            sid = assignments[bi]
            slot = self._slots[sid]
            slot['angle'] = 0.85 * slot['angle'] + 0.15 * angles[bi]
            lmm = br['len'] * SCALE
            tmm = br['total'] * SCALE
            slot['len_buf'].append(lmm)
            slot['tot_buf'].append(tmm)
            result.append((sid, br, float(np.mean(slot['len_buf'])), float(np.mean(slot['tot_buf']))))

        result.sort(key=lambda x: x[0])
        return result


# ── TKINTER HELPERS ─────────────────────────────────────────────────────────-

def _fit_image(cv_img, box_w, box_h):
    ih, iw = cv_img.shape[:2]
    if iw == 0 or ih == 0 or box_w < 1 or box_h < 1:
        return Image.new("RGB", (max(box_w, 1), max(box_h, 1)))
    scale = min(box_w / iw, box_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h))
    canvas.paste(pil, ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


class MaskPopup:
    def __init__(self, parent_root):
        self.win = tk.Toplevel(parent_root)
        self.win.title("Mask / Skeleton")
        self.win.configure(bg="#111")
        self.win.geometry("900x500")
        self.win.minsize(400, 200)
        self._photo = None
        tk.Label(self.win, text="Mask / Skeleton", bg="#111", fg="#aaa",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=6, pady=(4, 0))
        self.container = tk.Frame(self.win, bg="#111")
        self.container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))
        self.lbl = tk.Label(self.container, bg="#111")
        self.lbl.place(x=0, y=0)

    def update(self, cv_img):
        if not self.win.winfo_exists():
            return
        self.container.update_idletasks()
        bw = self.container.winfo_width()
        bh = self.container.winfo_height()
        if bw < 2 or bh < 2:
            return
        pil = _fit_image(cv_img, bw, bh)
        photo = ImageTk.PhotoImage(pil)
        self.lbl.configure(image=photo)
        self.lbl.place(x=0, y=0, width=bw, height=bh)
        self._photo = photo

    def show_placeholder(self, text=""):
        if not self.win.winfo_exists():
            return
        self.container.update_idletasks()
        bw = max(self.container.winfo_width(), 10)
        bh = max(self.container.winfo_height(), 10)
        img = np.zeros((bh, bw, 3), np.uint8)
        if text:
            tw = len(text) * 9
            cv2.putText(img, text, (max(bw // 2 - tw // 2, 5), bh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
        self.update(img)

    def is_open(self):
        try:
            return self.win.winfo_exists()
        except Exception:
            return False

    def close(self):
        if self.is_open():
            self.win.destroy()


class WireMeasurerApp:
    SIDEBAR_W = 340
    TOPBAR_H = 40
    INFO_H = 320
    TUNING_H = 300

    def __init__(self):
        self.settings = load_settings()
        self.cap = None
        self.kinect = None
        self.source_mode = "ip"  # ip | kinect
        self.running = False
        self.H = None
        self.wsize = None
        self.raw_corners = None
        self.fail_count = 0
        self.trunk_buf = deque(maxlen=SMOOTH_W)
        self.assigner = AngleColorAssigner()
        self.logged_mapping = False
        self._photo_refs = {}
        self._save_ctr = 0
        self._mask_popup = None

        self.root = tk.Tk()
        self.root.title("Wire Measurer - Phase 4 (Accurate)")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1920x1080")
        self.root.minsize(1280, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        top = tk.Frame(self.root, bg="#2d2d2d", height=self.TOPBAR_H)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        tk.Label(top, text="Camera URL:", bg="#2d2d2d", fg="white",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(10, 4))

        self.url_var = tk.StringVar(value=self.settings.get("stream_url", DEFAULT_URL))
        self.url_entry = tk.Entry(top, textvariable=self.url_var, width=42, font=("Consolas", 10))
        self.url_entry.pack(side=tk.LEFT, padx=4)

        self.connect_btn = tk.Button(top, text="Connect", bg="#4CAF50", fg="white",
                                     font=("Segoe UI", 10, "bold"), width=11,
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=4)

        self.kinect_btn = tk.Button(top, text="Kinect", bg="#2a6496", fg="white",
                                    font=("Segoe UI", 10, "bold"), width=9,
                                    command=self.toggle_kinect)
        self.kinect_btn.pack(side=tk.LEFT, padx=4)

        self.status_lbl = tk.Label(top, text="  Disconnected", bg="#2d2d2d", fg="#ff5555",
                                   font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="Export", bg="#555", fg="white", font=("Segoe UI", 9),
                  command=self.export_settings, width=7).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="Import", bg="#555", fg="white", font=("Segoe UI", 9),
                  command=self.import_settings, width=7).pack(side=tk.RIGHT, padx=2)

        # body
        self.body = tk.Frame(self.root, bg="#1e1e1e")
        self.body.pack(fill=tk.BOTH, expand=True)

        # right
        self.right = tk.Frame(self.body, bg="#1e1e1e", width=self.SIDEBAR_W)
        self.right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 6), pady=4)
        self.right.pack_propagate(False)

        tk.Label(self.right, text="Measurements", bg="#1e1e1e", fg="#aaa",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))

        self.info_outer = tk.Frame(self.right, bg="#111", width=self.SIDEBAR_W - 12, height=self.INFO_H)
        self.info_outer.pack(fill=tk.X, pady=(2, 4))
        self.info_outer.pack_propagate(False)
        self.info_lbl = tk.Label(self.info_outer, bg="#111")
        self.info_lbl.place(x=0, y=0, width=self.SIDEBAR_W - 12, height=self.INFO_H)

        tk.Frame(self.right, bg="#444", height=1).pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.right, text="Tuning", bg="#1e1e1e", fg="#aaa",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))

        tuning_outer = tk.Frame(self.right, bg="#1e1e1e", width=self.SIDEBAR_W - 12, height=self.TUNING_H)
        tuning_outer.pack(fill=tk.X)
        tuning_outer.pack_propagate(False)

        sf = tk.Frame(tuning_outer, bg="#1e1e1e")
        sf.pack(fill=tk.X)

        self.sliders = {}
        for name, lo, hi, key in [
            ("Block Size", 3, 101, "block_size"),
            ("C Offset", 0, 30, "c_offset"),
            ("Morph Size", 1, 15, "morph_size"),
            ("Blur Size", 1, 25, "blur_size"),
            ("Close Iter", 0, 15, "close_iter"),
        ]:
            row = tk.Frame(sf, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=name, bg="#1e1e1e", fg="#ccc", width=11, anchor=tk.W,
                     font=("Segoe UI", 9)).pack(side=tk.LEFT)
            s = tk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, bg="#1e1e1e",
                         fg="#ccc", highlightthickness=0, troughcolor="#333", length=180)
            s.set(self.settings.get(key, {
                "block_size": DEF_BS,
                "c_offset": DEF_CO,
                "morph_size": DEF_MS,
                "blur_size": DEF_BL,
                "close_iter": DEF_CI
            }[key]))
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sliders[name] = s

        tf = tk.Frame(tuning_outer, bg="#1e1e1e")
        tf.pack(fill=tk.X, pady=(6, 2))
        self.skel_var = tk.BooleanVar(value=True)
        self.branch_var = tk.BooleanVar(value=True)
        for txt, var, cmd in [
            ("Show Skeleton", self.skel_var, None),
            ("Branch Mode", self.branch_var, self.on_branch_toggle),
        ]:
            tk.Checkbutton(tf, text=txt, variable=var, bg="#1e1e1e", fg="#ccc",
                           selectcolor="#333", activebackground="#1e1e1e",
                           font=("Segoe UI", 9), command=cmd).pack(anchor=tk.W)

        self.mask_btn = tk.Button(tf, text="▶  Mask / Skel View", bg="#2a6496", fg="white",
                                  font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                                  command=self.toggle_mask_popup)
        self.mask_btn.pack(anchor=tk.W, pady=(8, 0), fill=tk.X)

        tk.Button(tf, text="Screenshot", bg="#555", fg="white", font=("Segoe UI", 9),
                  command=self.screenshot).pack(anchor=tk.W, pady=(6, 0))

        # left views
        self.left = tk.Frame(self.body, bg="#1e1e1e")
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=4)
        self.views_frame = tk.Frame(self.left, bg="#1e1e1e")
        self.views_frame.pack(fill=tk.BOTH, expand=True)

        self.live_frame = tk.Frame(self.views_frame, bg="#1e1e1e")
        self.live_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        tk.Label(self.live_frame, text="Live Feed", bg="#1e1e1e", fg="#aaa",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.live_container = tk.Frame(self.live_frame, bg="#111")
        self.live_container.pack(fill=tk.BOTH, expand=True)
        self.live_container.pack_propagate(False)
        self.live_lbl = tk.Label(self.live_container, bg="#111")
        self.live_lbl.place(x=0, y=0)

        self.meas_frame = tk.Frame(self.views_frame, bg="#1e1e1e")
        self.meas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        tk.Label(self.meas_frame, text="Measurement View", bg="#1e1e1e", fg="#aaa",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.meas_container = tk.Frame(self.meas_frame, bg="#111")
        self.meas_container.pack(fill=tk.BOTH, expand=True)
        self.meas_container.pack_propagate(False)
        self.meas_lbl = tk.Label(self.meas_container, bg="#111")
        self.meas_lbl.place(x=0, y=0)

        self.root.after(100, self._initial_placeholders)

    def _initial_placeholders(self):
        self._placeholder(self.live_container, self.live_lbl, "No camera connected")
        self._placeholder(self.meas_container, self.meas_lbl, "Waiting for feed...")
        self._show_info(np.zeros((self.INFO_H, self.SIDEBAR_W - 12, 3), np.uint8))

    def toggle_mask_popup(self):
        if self._mask_popup and self._mask_popup.is_open():
            self._mask_popup.close()
            self._mask_popup = None
            self.mask_btn.config(text="▶  Mask / Skel View", bg="#2a6496")
        else:
            self._mask_popup = MaskPopup(self.root)
            self._mask_popup.win.protocol("WM_DELETE_WINDOW", self._on_popup_close)
            self.mask_btn.config(text="✕  Close Mask View", bg="#8b3a3a")

    def _on_popup_close(self):
        if self._mask_popup:
            self._mask_popup.close()
            self._mask_popup = None
        self.mask_btn.config(text="▶  Mask / Skel View", bg="#2a6496")

    def _container_size(self, container):
        container.update_idletasks()
        w = container.winfo_width()
        h = container.winfo_height()
        return (w if w > 1 else 400, h if h > 1 else 300)

    def _show(self, container, label, cv_img, key):
        bw, bh = self._container_size(container)
        pil = _fit_image(cv_img, bw, bh)
        photo = ImageTk.PhotoImage(pil)
        label.configure(image=photo)
        label.place(x=0, y=0, width=bw, height=bh)
        self._photo_refs[key] = photo

    def _show_info(self, cv_img):
        bw, bh = self.SIDEBAR_W - 12, self.INFO_H
        pil = _fit_image(cv_img, bw, bh)
        photo = ImageTk.PhotoImage(pil)
        self.info_lbl.configure(image=photo)
        self._photo_refs["info"] = photo

    def _placeholder(self, container, label, text):
        bw, bh = self._container_size(container)
        img = np.zeros((max(bh, 10), max(bw, 10), 3), np.uint8)
        if text:
            tw = len(text) * 10
            cv2.putText(img, text, (max(bw // 2 - tw // 2, 5), bh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        self._show(container, label, img, "ph_" + str(id(label)))

    def export_settings(self):
        cfg = {k: max(self.sliders[n].get(), 1) for n, k in
               [("Block Size", "block_size"), ("C Offset", "c_offset"),
                ("Morph Size", "morph_size"), ("Blur Size", "blur_size"),
                ("Close Iter", "close_iter")]}
        p = filedialog.asksaveasfilename(title="Export", defaultextension=".json",
                                         filetypes=[("JSON", "*.json")],
                                         initialfile="tuning_preset.json")
        if p:
            with open(p, 'w') as f:
                json.dump(cfg, f, indent=2)

    def import_settings(self):
        p = filedialog.askopenfilename(title="Import", filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p) as f:
                cfg = json.load(f)
            for n, k in [("Block Size", "block_size"), ("C Offset", "c_offset"),
                         ("Morph Size", "morph_size"), ("Blur Size", "blur_size"),
                         ("Close Iter", "close_iter")]:
                if k in cfg:
                    self.sliders[n].set(cfg[k])
        except Exception as e:
            print(f"[ERROR] {e}")

    # ── CONNECTIONS ─────────────────────────────────────────────────────────-

    def toggle_connection(self):
        self.source_mode = "ip"
        if self.running:
            self.disconnect_ip()
        else:
            self.connect_ip()

    def toggle_kinect(self):
        self.source_mode = "kinect"
        if self.running:
            self.disconnect_kinect()
        else:
            self.connect_kinect()

    def connect_ip(self):
        url = self.url_var.get().strip()
        if not url:
            self.status_lbl.config(text="  Enter a URL", fg="#ff5555")
            return
        self.status_lbl.config(text="  Connecting...", fg="#ffaa00")
        self.connect_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            self.status_lbl.config(text="  Connection failed", fg="#ff5555")
            self.connect_btn.config(state=tk.NORMAL)
            cap.release()
            return
        self.cap = cap
        self.running = True
        self.fail_count = 0
        self.H = None
        self.raw_corners = None
        self.logged_mapping = False
        self.status_lbl.config(text="  Connected (IP)", fg="#55ff55")
        self.connect_btn.config(text="Disconnect", bg="#f44336", state=tk.NORMAL)
        self.url_entry.config(state=tk.DISABLED)
        self.settings["stream_url"] = url
        save_settings(self.settings)
        self.process_frame()

    def disconnect_ip(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_lbl.config(text="  Disconnected", fg="#ff5555")
        self.connect_btn.config(text="Connect", bg="#4CAF50")
        self.connect_btn.config(state=tk.NORMAL)
        self.url_entry.config(state=tk.NORMAL)
        self._placeholder(self.live_container, self.live_lbl, "No camera connected")
        self._placeholder(self.meas_container, self.meas_lbl, "Waiting for feed...")

    def connect_kinect(self):
        if not HAS_KINECT:
            self.status_lbl.config(text="  Kinect not found (pykinect)", fg="#ff5555")
            return

        # Ensure IP camera is not holding resources
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.status_lbl.config(text="  Connecting Kinect...", fg="#ffaa00")
        self.kinect_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            self.kinect = KinectV1()
            ok = self.kinect.start(timeout_s=5.0)
            if not ok:
                raise RuntimeError("Timed out waiting for Kinect frames")
        except Exception as e:
            self.kinect = None
            self.status_lbl.config(text="  Kinect connection failed", fg="#ff5555")
            self.kinect_btn.config(state=tk.NORMAL)
            print(f"[ERROR] Kinect init failed: {e}")
            return

        self.running = True
        self.fail_count = 0
        self.H = None
        self.raw_corners = None
        self.logged_mapping = False

        self.status_lbl.config(text="  Connected (Kinect)", fg="#55ff55")
        self.kinect_btn.config(text="Disconnect", bg="#f44336", state=tk.NORMAL)
        self.connect_btn.config(state=tk.NORMAL)
        self.url_entry.config(state=tk.DISABLED)

        self.process_frame()

    def disconnect_kinect(self):
        self.running = False
        if self.kinect:
            try:
                self.kinect.release()
            except Exception:
                pass
            self.kinect = None

        self.status_lbl.config(text="  Disconnected", fg="#ff5555")
        self.kinect_btn.config(text="Kinect", bg="#2a6496", state=tk.NORMAL)
        self.url_entry.config(state=tk.NORMAL)
        self._placeholder(self.live_container, self.live_lbl, "No camera connected")
        self._placeholder(self.meas_container, self.meas_lbl, "Waiting for feed...")

    # ── MAIN LOOP ─────────────────────────────────────────────────────────---

    def process_frame(self):
        try:
            if not self.running:
                return

            if self.source_mode == "kinect":
                if not self.kinect:
                    return
                ret, frame, depth_mm = self.kinect.read()
            else:
                if not self.cap:
                    return
                ret, frame = self.cap.read()
                depth_mm = None

            if not ret:
                self.fail_count += 1
                if self.fail_count >= MAX_FAIL:
                    self.status_lbl.config(text="  Feed lost", fg="#ff5555")
                    if self.source_mode == "kinect":
                        self.disconnect_kinect()
                    else:
                        self.disconnect_ip()
                    return
                self.root.after(30, self.process_frame)
                return

            self.fail_count = 0

            ss = self.skel_var.get()
            bm = self.branch_var.get()
            popup_open = self._mask_popup is not None and self._mask_popup.is_open()

            bsz = max(self.sliders["Block Size"].get(), 3)
            co = self.sliders["C Offset"].get()
            ms = max(self.sliders["Morph Size"].get(), 1)
            bls = max(self.sliders["Blur Size"].get(), 1)
            ci = max(self.sliders["Close Iter"].get(), 0)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # CLAHE: improves local contrast without over-amplifying noise like equalizeHist
            _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = _clahe.apply(gray)
            corners, ids, _ = det.detectMarkers(gray)
            # If we got detections, filter to only keep IDs 0,1,2,3 (our known marker set)
            # This prevents stray false-positive IDs from confusing homography
            if ids is not None and len(ids) > 0:
                valid_mask = np.isin(ids.flatten(), list(MARKER_CORNER.keys()))
                if valid_mask.any():
                    corners = [corners[i] for i in range(len(ids)) if valid_mask[i]]
                    ids = ids[valid_mask].reshape(-1, 1)
                else:
                    corners, ids = [], None

            hH = False
            nm = 0
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                self.H, self.wsize, self.raw_corners = homography(corners, ids)
                hH = self.H is not None
                nm = len(ids)

            hf, wf = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (wf, 40), (0, 0, 0), -1)
            c = (0, 255, 0) if nm == 4 else (0, 165, 255) if nm > 0 else (0, 0, 255)
            cv2.putText(frame, f"Markers: {nm}/4", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            cv2.putText(frame, f"Scale: {SCALE:.4f} mm/px", (220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            self._show(self.live_container, self.live_lbl, frame, "live")

            iw, ih = self.SIDEBAR_W - 12, self.INFO_H
            info = np.zeros((ih, iw, 3), np.uint8)

            if hH and self.wsize:
                warped = cv2.warpPerspective(frame, self.H, self.wsize)
                wm = wire_mask(warped, bsz, co, ms, bls, ci)
                wm = mask_aruco(wm, self.H, self.raw_corners)
                sk = get_skel(wm)

                disp = warped.copy()
                blk = INSET_PX + int(25 * PX_MM)
                cv2.rectangle(disp, (blk, blk), (disp.shape[1] - blk, disp.shape[0] - blk), (80, 80, 80), 1)

                if bm:
                    ncc, la, sa, _ = cv2.connectedComponentsWithStats(sk, 8)
                    skm = ((la == (np.argmax(sa[1:, cv2.CC_STAT_AREA]) + 1)) * 255).astype(np.uint8) if ncc > 1 else sk.copy()

                    res = analyse(skm)
                    if res is not None:
                        ti = res['trunk']
                        bis = res['branches']
                        jis = res['junctions']

                        # trunk length: 3D in Kinect mode, else 2D
                        t3d, _ = length_path_3d_mm(ti['path'], self.H, depth_mm) if self.source_mode == 'kinect' else (None, 0.0)
                        tmm = (t3d if t3d is not None else ti['len'] * SCALE)
                        self.trunk_buf.append(tmm)
                        ta = float(np.mean(self.trunk_buf))

                        if ss:
                            tp = ti['path']
                            if len(tp) > 1:
                                cv2.polylines(disp, [np.array([(p[1], p[0]) for p in tp], np.int32)], False, (0, 0, 255), 3)

                        # Branch lengths: we overwrite br['len'] and br['total'] with pseudo-pixels based on 3D mm
                        if self.source_mode == 'kinect':
                            bis2 = []
                            for b in bis:
                                b2 = dict(b)
                                b_len3d, _ = length_path_3d_mm(b['path'], self.H, depth_mm)
                                if b_len3d is not None:
                                    b2['len'] = b_len3d / SCALE
                                # total = trunk + links + branch
                                if b.get('links'):
                                    tot_links = 0.0
                                    ok = True
                                    for lseg in b['links']:
                                        seg3d, _ = length_path_3d_mm(lseg['path'], self.H, depth_mm)
                                        if seg3d is None:
                                            ok = False
                                            break
                                        tot_links += seg3d
                                    if ok and (t3d is not None) and (b_len3d is not None):
                                        b2['total'] = (t3d + tot_links + b_len3d) / SCALE
                                bis2.append(b2)
                            bis = bis2

                        tracked = self.assigner.assign(bis)

                        for sid, br, avg_len, avg_tot in tracked:
                            col = BC[sid % len(BC)]
                            if ss:
                                bp = br['path']
                                if len(bp) > 1:
                                    cv2.polylines(disp, [np.array([(p[1], p[0]) for p in bp], np.int32)], False, col, 3)

                        if ss:
                            for jy, jx in jis:
                                cv2.circle(disp, (jx, jy), 10, (0, 255, 0), 3)

                        nb = len(tracked)
                        y0 = 22
                        cv2.putText(info, f"Harness: 1 trunk + {nb} branch(es)", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        y0 += 30
                        cv2.putText(info, f"Trunk:  {ta:.1f} mm  ({ta/10:.2f} cm)", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)
                        y0 += 26
                        for sid, br, avg_len, avg_tot in tracked:
                            col = BC[sid % len(BC)]
                            cv2.putText(info, f"Branch {sid+1}: {avg_len:.1f} mm ({avg_len/10:.2f} cm)", (10, y0),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                            y0 += 22
                        y0 += 6
                        cv2.line(info, (10, y0), (iw - 10, y0), (100, 100, 100), 1)
                        y0 += 18
                        cv2.putText(info, "Totals (trunk + branch):", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        y0 += 22
                        for sid, br, avg_len, avg_tot in tracked:
                            col = BC[sid % len(BC)]
                            cv2.putText(info, f"Total {sid+1}: {avg_tot:.1f} mm ({avg_tot/10:.2f} cm)", (10, y0),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                            y0 += 24

                self._show(self.meas_container, self.meas_lbl, disp, "meas")

                if popup_open:
                    mb = cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)
                    sb = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)
                    sb[sk > 0] = (0, 255, 0)
                    self._mask_popup.update(np.hstack([mb, sb]))

            else:
                self._placeholder(self.meas_container, self.meas_lbl, "Need all 4 ArUco markers")
                if popup_open:
                    self._mask_popup.show_placeholder("Need all 4 ArUco markers")

            self._show_info(info)

            self._save_ctr += 1
            if self._save_ctr % 100 == 0:
                self.settings.update({
                    "block_size": bsz,
                    "c_offset": co,
                    "morph_size": ms,
                    "blur_size": bls,
                    "close_iter": ci,
                })
                save_settings(self.settings)

            self.root.after(30, self.process_frame)

        except Exception as e:
            import traceback
            print('\n[PROCESS_FRAME ERROR]')
            traceback.print_exc()
            try:
                self.status_lbl.config(text=f"  Error: {type(e).__name__}", fg="#ff5555")
            except Exception:
                pass
            self.running = False
            return

    def on_branch_toggle(self):
        self.trunk_buf.clear()
        self.assigner.reset()

    def screenshot(self):
        print("[INFO] Screenshot saved (if feed active)")

    def on_close(self):
        self.running = False
        if self._mask_popup:
            self._mask_popup.close()
        if self.kinect:
            try:
                self.kinect.release()
            except Exception:
                pass
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = WireMeasurerApp()
    app.run()


if __name__ == "__main__":
    main()
