#!/usr/bin/env python3
"""
Phase 4 – Wire harness measurement (accurate algorithm).
Single-window tkinter UI optimised for 1920x1080.
Aspect-ratio-preserving views. Fixed layout that does NOT grow.
Mask/Skeleton view is now a detachable popup window.
"""

import cv2
import numpy as np
import sys
import json
import os
import math
import tkinter as tk
from tkinter import ttk, filedialog
from collections import deque
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from PIL import Image, ImageTk
from itertools import permutations

# ── CONFIG ────────────────────────────────────────────────────────────────────
SETTINGS_FILE    = "wire_settings.json"
DEFAULT_URL      = "http://10.65.52.117:8080/video"
DICT_TYPE        = cv2.aruco.DICT_5X5_100
MAX_FAIL         = 30
A4_W, A4_H       = 210.0, 297.0
INSET_MM         = 5 + (57 / 2)
PX_MM            = 4.0
OUT_W, OUT_H     = int(A4_W * PX_MM), int(A4_H * PX_MM)
SCALE            = 1.0 / PX_MM
INSET_PX         = int(INSET_MM * PX_MM)
SMOOTH_W         = 30
MIN_SEG          = 30
MIN_BRANCH_MM    = 15.0
JN_CLUSTER_R     = 20
JN_MATCH_R       = 25
EP_MATCH_R       = 10
ARUCO_MARGIN_PX  = 30
SPLINE_SMOOTH    = 5.0
SPLINE_PTS       = 500

# How far a branch tip can move between frames and still be considered
# the same branch (in pixels). Generous to handle skeleton jitter.
TIP_MATCH_DIST   = 60

MARKER_CORNER = {0: "tl", 1: "tr", 2: "bl", 3: "br"}
DEF_BS, DEF_CO, DEF_MS, DEF_BL, DEF_CI = 53, 13, 5, 11, 5

ad  = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
det = cv2.aruco.ArucoDetector(ad, cv2.aruco.DetectorParameters())

# ── SETTINGS ──────────────────────────────────────────────────────────────────
def load_settings():
    d = {"stream_url": DEFAULT_URL, "block_size": DEF_BS, "c_offset": DEF_CO,
         "morph_size": DEF_MS, "blur_size": DEF_BL, "close_iter": DEF_CI}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f: d.update(json.load(f))
        except Exception: pass
    return d

def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w") as f: json.dump(s, f, indent=2)
    except Exception: pass

# ── HOMOGRAPHY ────────────────────────────────────────────────────────────────
_DST = {
    "tl": np.array([INSET_PX, INSET_PX], np.float32),
    "tr": np.array([OUT_W - INSET_PX, INSET_PX], np.float32),
    "bl": np.array([INSET_PX, OUT_H - INSET_PX], np.float32),
    "br": np.array([OUT_W - INSET_PX, OUT_H - INSET_PX], np.float32),
}

def homography(corners, ids):
    found = set(ids.flatten().tolist())
    needed = set(MARKER_CORNER.keys())
    if not needed.issubset(found): return None, None, None
    centres = {}
    for i, m in enumerate(ids.flatten()):
        mid = int(m)
        if mid in needed:
            p = corners[i][0]
            centres[mid] = np.array([p[:, 0].mean(), p[:, 1].mean()], np.float32)
    src = np.array([centres[mid] for mid in MARKER_CORNER], np.float32)
    dst = np.array([_DST[MARKER_CORNER[mid]] for mid in MARKER_CORNER], np.float32)
    H, _ = cv2.findHomography(src, dst)
    ac = [corners[i][0] for i in range(len(ids))]
    return H, (OUT_W, OUT_H), ac

# ── SEGMENTATION ──────────────────────────────────────────────────────────────
def wire_mask(fr, bs, co, ms, bl, ci):
    l = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0]
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    l = cv2.bilateralFilter(l, bl | 1, 75, 75)
    m = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, bs | 1 if bs >= 3 else 3, co)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ms, ms))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ms * 2 + 1, ms * 2 + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=ci)
    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = np.zeros_like(m)
    for c in cn:
        if cv2.contourArea(c) >= 500: cv2.drawContours(r, [c], -1, 255, -1)
    return r

def mask_aruco(m, H, raw_corners):
    if raw_corners is None or H is None:
        h, w = m.shape[:2]; b = INSET_PX + int(25 * PX_MM)
        m[0:b, 0:b] = 0; m[0:b, w-b:w] = 0; m[h-b:h, 0:b] = 0; m[h-b:h, w-b:w] = 0
        m[0:int(15 * PX_MM), :] = 0
        return m
    mg = ARUCO_MARGIN_PX
    for quad in raw_corners:
        pts = np.array(quad, np.float32).reshape(-1, 1, 2)
        wp = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        x0 = max(int(np.floor(wp[:, 0].min()) - mg), 0)
        x1 = min(int(np.ceil(wp[:, 0].max()) + mg), m.shape[1])
        y0 = max(int(np.floor(wp[:, 1].min()) - mg), 0)
        y1 = min(int(np.ceil(wp[:, 1].max()) + mg), m.shape[0])
        m[y0:y1, x0:x1] = 0
    return m

# ── SKELETON ──────────────────────────────────────────────────────────────────
def get_skel(m): return (skeletonize(m > 0).astype(np.uint8)) * 255

def n8(s, y, x):
    h, w = s.shape; o = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and s[ny, nx]: o.append((ny, nx))
    return o

def trace(s, st):
    v = {st}; p = [st]; c = st
    while True:
        nx = None
        for n in n8(s, c[0], c[1]):
            if n not in v: nx = n; break
        if nx is None: break
        p.append(nx); v.add(nx); c = nx
    return p

def alen(pts):
    if len(pts) < 2: return 0.0
    a = np.array(pts, np.float64)
    return float(np.sqrt((np.diff(a, axis=0)**2).sum(axis=1)).sum())

def spline_length(pts):
    if len(pts) < 6: return alen(pts)
    arr = np.array(pts, np.float64)
    if len(arr) > 2000:
        idx = np.linspace(0, len(arr)-1, 2000).astype(int); arr = arr[idx]
    try:
        s = SPLINE_SMOOTH * len(arr)
        tck, u = splprep([arr[:, 0], arr[:, 1]], s=s, k=3)
        u_f = np.linspace(0, 1, SPLINE_PTS)
        y_s, x_s = splev(u_f, tck)
        return float(np.sqrt(np.diff(y_s)**2 + np.diff(x_s)**2).sum())
    except Exception: return alen(pts)

# ── HARNESS ANALYSIS ─────────────────────────────────────────────────────────
def analyse(skeleton):
    s = (skeleton > 0).astype(np.uint8)
    ys, xs = np.where(s > 0)
    if len(ys) < 30: return None

    eps = set(); jns = set()
    for y, x in zip(ys, xs):
        nn = len(n8(s, y, x))
        if nn == 1: eps.add((y, x))
        elif nn >= 3: jns.add((y, x))
    if not jns: return None

    jl = list(jns); used = set(); clusters = []
    for i, p in enumerate(jl):
        if i in used: continue
        cl = [p]; used.add(i); stk = [p]
        while stk:
            cur = stk.pop()
            for j, p2 in enumerate(jl):
                if j in used: continue
                if max(abs(cur[0]-p2[0]), abs(cur[1]-p2[1])) <= JN_CLUSTER_R:
                    cl.append(p2); used.add(j); stk.append(p2)
        clusters.append(cl)

    jcenters = [(int(np.mean([p[0] for p in cl])), int(np.mean([p[1] for p in cl]))) for cl in clusters]

    sc = s.copy()
    for py, px in jns: sc[py, px] = 0

    ncc, labels = cv2.connectedComponents(sc, connectivity=8)
    segs = []
    for lb in range(1, ncc):
        cy, cx = np.where(labels == lb)
        if len(cy) < 5: continue
        comp = (labels == lb).astype(np.uint8)
        se = []
        for yy, xx in zip(cy, cx):
            if len(n8(comp, yy, xx)) == 1: se.append((yy, xx))
        if not se: continue
        path = trace(comp, se[0])
        length = spline_length(path)
        if length < MIN_SEG: continue

        def classify(pt):
            for ep in eps:
                if max(abs(pt[0]-ep[0]), abs(pt[1]-ep[1])) <= EP_MATCH_R: return 'ep', ep
            best_d = 999; best_j = None
            for jc in jcenters:
                d = max(abs(pt[0]-jc[0]), abs(pt[1]-jc[1]))
                if d < best_d: best_d = d; best_j = jc
            if best_d <= JN_MATCH_R: return 'jn', best_j
            return 'unk', pt

        t0, n0 = classify(path[0]); t1, n1 = classify(path[-1])
        segs.append({'path': path, 'len': length, 'e0': n0, 't0': t0, 'e1': n1, 't1': t1})

    if len(segs) < 2: return None

    tc = [s for s in segs if s['t0'] == 'ep' or s['t1'] == 'ep']
    if not tc: tc = segs
    trunk = max(tc, key=lambda s: s['len'])

    if trunk['t0'] == 'ep':   tep = trunk['e0']; tjn = trunk['e1']
    elif trunk['t1'] == 'ep': tep = trunk['e1']; tjn = trunk['e0']
    else:                     tep = trunk['e0']; tjn = trunk['e1']

    branches = []; uid = {id(trunk)}

    def bfs(junc, dist, links_so_far=None):
        if links_so_far is None:
            links_so_far = []
        for seg in segs:
            if id(seg) in uid: continue
            conn = None; other = None; ot = None
            for ta, ea, tb, eb in [(seg['t0'], seg['e0'], seg['t1'], seg['e1']),
                                    (seg['t1'], seg['e1'], seg['t0'], seg['e0'])]:
                if ta == 'jn' and ea is not None:
                    if max(abs(ea[0]-junc[0]), abs(ea[1]-junc[1])) <= JN_MATCH_R:
                        conn = ea; other = eb; ot = tb; break
            if conn is None: continue
            uid.add(id(seg)); tot = dist + seg['len']
            if ot in ('ep', 'unk'):
                branches.append({'path': seg['path'], 'len': seg['len'],
                                 'endpoint': other, 'junction': junc, 'total': tot,
                                 'links': links_so_far + [seg]})
            elif ot == 'jn':
                bfs(other, tot, links_so_far + [seg])
    bfs(tjn, trunk['len'])

    min_branch_px = MIN_BRANCH_MM * PX_MM
    branches = [b for b in branches if b['len'] >= min_branch_px]
    if not branches: return None

    # Compute tip (far endpoint from junction) for each branch
    for b in branches:
        p = b['path']
        d0 = (p[0][0] - tjn[0])**2 + (p[0][1] - tjn[1])**2
        d1 = (p[-1][0] - tjn[0])**2 + (p[-1][1] - tjn[1])**2
        b['tip'] = p[0] if d0 > d1 else p[-1]

    # Sort by angle as canonical stable ordering
    branches.sort(key=lambda b: math.atan2(b['tip'][0] - tjn[0], b['tip'][1] - tjn[1]))

    used_j = set()
    if tjn: used_j.add(tjn)
    for b in branches:
        if b['junction']: used_j.add(b['junction'])
    active_j = [jc for jc in jcenters if jc in used_j]

    return {'trunk': {'path': trunk['path'], 'len': trunk['len'], 'ep': tep},
            'branches': branches, 'junctions': active_j, 'tjn': tjn}

# ── SIMPLE FALLBACK ───────────────────────────────────────────────────────────
def simple(skeleton):
    ncc, labels = cv2.connectedComponents(skeleton)
    out = []
    for lb in range(1, ncc):
        comp = ((labels == lb)*255).astype(np.uint8)
        su = (comp > 0).astype(np.uint8)
        k = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
        nc = cv2.filter2D(su, -1, k)
        ep = np.argwhere((su == 1) & (nc == 1))
        st = ep[0] if len(ep) else np.argwhere(comp > 0)[0]
        vis = set(); path = []; r, c = int(st[0]), int(st[1])
        path.append((c, r)); vis.add((r, c)); h, w = comp.shape
        while True:
            f = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    nr, nc2 = r+dy, c+dx
                    if 0 <= nr < h and 0 <= nc2 < w and comp[nr, nc2] and (nr, nc2) not in vis:
                        vis.add((nr, nc2)); r, c = nr, nc2; path.append((c, r)); f = True; break
                if f: break
            if not f: break
        raw_pts = [(p[1], p[0]) for p in path]
        l = spline_length(raw_pts)
        if l > 20: out.append((path, l))
    out.sort(key=lambda x: x[1], reverse=True); return out

# ── COLORS ────────────────────────────────────────────────────────────────────
BC = [(255,150,0),(0,165,255),(255,0,255),(0,255,255),
      (255,255,0),(128,0,255),(0,255,128),(255,128,0)]

# ── STABLE BRANCH TRACKER ─────────────────────────────────────────────────────
class BranchTracker:
    """
    Maintains a stable mapping from physical branch identity (tracked by tip
    position) to a fixed slot index (0, 1, 2, …).  Each slot keeps its own
    smoothing buffers so colours never swap.
    """
    def __init__(self, smooth_w=SMOOTH_W, max_dist=TIP_MATCH_DIST):
        self.smooth_w  = smooth_w
        self.max_dist  = max_dist
        # slot → deque of (len_mm, total_mm) measurements
        self.len_bufs  = {}   # slot → deque
        self.tot_bufs  = {}   # slot → deque
        # slot → last known tip (y, x)
        self.slot_tips = {}
        self._next_slot = 0

    def reset(self):
        self.len_bufs.clear()
        self.tot_bufs.clear()
        self.slot_tips.clear()
        self._next_slot = 0

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, branches):
        """
        Match incoming branches to existing slots by nearest tip.
        Returns list of (slot_index, branch_dict) in the same order as branches.
        """
        tips = [b['tip'] for b in branches]
        n    = len(tips)

        if not self.slot_tips:
            # First frame – create slots in whatever order branches arrived
            assignments = []
            for i, (tip, br) in enumerate(zip(tips, branches)):
                slot = self._next_slot; self._next_slot += 1
                self.slot_tips[slot] = tip
                self.len_bufs[slot]  = deque(maxlen=self.smooth_w)
                self.tot_bufs[slot]  = deque(maxlen=self.smooth_w)
                assignments.append(slot)
        else:
            existing_slots = list(self.slot_tips.keys())
            assignments    = [None] * n

            # Greedy nearest-neighbour match (good enough for ≤8 branches)
            used_slots = set()
            dist_matrix = {}
            for i, tip in enumerate(tips):
                for slot in existing_slots:
                    dist_matrix[(i, slot)] = self._dist(tip, self.slot_tips[slot])

            # Sort all (branch_idx, slot) pairs by distance, assign greedily
            pairs = sorted(dist_matrix.keys(), key=lambda k: dist_matrix[k])
            used_branches = set()
            for (bi, slot) in pairs:
                if bi in used_branches or slot in used_slots:
                    continue
                if dist_matrix[(bi, slot)] <= self.max_dist:
                    assignments[bi] = slot
                    used_slots.add(slot)
                    used_branches.add(bi)

            # Any unmatched branch gets a new slot
            for i in range(n):
                if assignments[i] is None:
                    slot = self._next_slot; self._next_slot += 1
                    self.slot_tips[slot] = tips[i]
                    self.len_bufs[slot]  = deque(maxlen=self.smooth_w)
                    self.tot_bufs[slot]  = deque(maxlen=self.smooth_w)
                    assignments[i] = slot

        # Push measurements and update tip positions
        result = []
        for i, (slot, br) in enumerate(zip(assignments, branches)):
            lmm  = br['len']   * SCALE
            tmm  = br['total'] * SCALE
            self.len_bufs[slot].append(lmm)
            self.tot_bufs[slot].append(tmm)
            self.slot_tips[slot] = tips[i]   # smooth tip update
            result.append((slot, br,
                           float(np.mean(self.len_bufs[slot])),
                           float(np.mean(self.tot_bufs[slot]))))

        # Sort result by slot so the sidebar order is also stable
        result.sort(key=lambda x: x[0])
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  TKINTER UI
# ══════════════════════════════════════════════════════════════════════════════

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
        if not self.win.winfo_exists(): return
        self.container.update_idletasks()
        bw = self.container.winfo_width()
        bh = self.container.winfo_height()
        if bw < 2 or bh < 2: return
        pil = _fit_image(cv_img, bw, bh)
        photo = ImageTk.PhotoImage(pil)
        self.lbl.configure(image=photo)
        self.lbl.place(x=0, y=0, width=bw, height=bh)
        self._photo = photo

    def show_placeholder(self, text=""):
        if not self.win.winfo_exists(): return
        self.container.update_idletasks()
        bw = max(self.container.winfo_width(), 10)
        bh = max(self.container.winfo_height(), 10)
        img = np.zeros((bh, bw, 3), np.uint8)
        if text:
            tw = len(text) * 9
            cv2.putText(img, text, (max(bw//2 - tw//2, 5), bh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
        self.update(img)

    def is_open(self):
        try: return self.win.winfo_exists()
        except Exception: return False

    def close(self):
        if self.is_open(): self.win.destroy()


class WireMeasurerApp:
    SIDEBAR_W = 340
    TOPBAR_H  = 40
    PAD       = 4
    INFO_H    = 320
    TUNING_H  = 300

    def __init__(self):
        self.settings = load_settings()
        self.cap = None
        self.running = False
        self.H = None; self.wsize = None; self.raw_corners = None
        self.show_skel = True; self.show_mask = False; self.branch_mode = True
        self.fail_count = 0
        self.trunk_buf = deque(maxlen=SMOOTH_W)
        self.tracker = BranchTracker()   # ← replaces br_bufs / tot_bufs
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

        # ── top bar ──
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
        self.status_lbl = tk.Label(top, text="  Disconnected", bg="#2d2d2d", fg="#ff5555",
                                   font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="Export", bg="#555", fg="white", font=("Segoe UI", 9),
                  command=self.export_settings, width=7).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="Import", bg="#555", fg="white", font=("Segoe UI", 9),
                  command=self.import_settings, width=7).pack(side=tk.RIGHT, padx=2)

        # ── body ──
        self.body = tk.Frame(self.root, bg="#1e1e1e")
        self.body.pack(fill=tk.BOTH, expand=True)

        # ── RIGHT sidebar ──
        self.right = tk.Frame(self.body, bg="#1e1e1e", width=self.SIDEBAR_W)
        self.right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 6), pady=4)
        self.right.pack_propagate(False)
        tk.Label(self.right, text="Measurements", bg="#1e1e1e", fg="#aaa",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.info_outer = tk.Frame(self.right, bg="#111",
                                   width=self.SIDEBAR_W - 12, height=self.INFO_H)
        self.info_outer.pack(fill=tk.X, pady=(2, 4))
        self.info_outer.pack_propagate(False)
        self.info_lbl = tk.Label(self.info_outer, bg="#111")
        self.info_lbl.place(x=0, y=0, width=self.SIDEBAR_W - 12, height=self.INFO_H)
        tk.Frame(self.right, bg="#444", height=1).pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.right, text="Tuning", bg="#1e1e1e", fg="#aaa",
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))
        tuning_outer = tk.Frame(self.right, bg="#1e1e1e",
                                width=self.SIDEBAR_W - 12, height=self.TUNING_H)
        tuning_outer.pack(fill=tk.X)
        tuning_outer.pack_propagate(False)
        sf = tk.Frame(tuning_outer, bg="#1e1e1e")
        sf.pack(fill=tk.X)
        self.sliders = {}
        for name, lo, hi, key in [("Block Size", 3, 101, "block_size"),
                                    ("C Offset",   0,  30, "c_offset"),
                                    ("Morph Size", 1,  15, "morph_size"),
                                    ("Blur Size",  1,  25, "blur_size"),
                                    ("Close Iter", 0,  15, "close_iter")]:
            row = tk.Frame(sf, bg="#1e1e1e"); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=name, bg="#1e1e1e", fg="#ccc", width=11, anchor=tk.W,
                     font=("Segoe UI", 9)).pack(side=tk.LEFT)
            s = tk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, bg="#1e1e1e",
                         fg="#ccc", highlightthickness=0, troughcolor="#333", length=180)
            s.set(self.settings.get(key, {"block_size": DEF_BS, "c_offset": DEF_CO,
                  "morph_size": DEF_MS, "blur_size": DEF_BL, "close_iter": DEF_CI}[key]))
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sliders[name] = s
        tf = tk.Frame(tuning_outer, bg="#1e1e1e"); tf.pack(fill=tk.X, pady=(6, 2))
        self.skel_var   = tk.BooleanVar(value=True)
        self.branch_var = tk.BooleanVar(value=True)
        for txt, var, cmd in [("Show Skeleton", self.skel_var, None),
                               ("Branch Mode",  self.branch_var, self.on_branch_toggle)]:
            tk.Checkbutton(tf, text=txt, variable=var, bg="#1e1e1e", fg="#ccc",
                           selectcolor="#333", activebackground="#1e1e1e",
                           font=("Segoe UI", 9), command=cmd).pack(anchor=tk.W)
        self.mask_btn = tk.Button(tf, text="▶  Mask / Skel View", bg="#2a6496", fg="white",
                                  font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                                  command=self.toggle_mask_popup)
        self.mask_btn.pack(anchor=tk.W, pady=(8, 0), fill=tk.X)
        tk.Button(tf, text="Screenshot", bg="#555", fg="white", font=("Segoe UI", 9),
                  command=self.screenshot).pack(anchor=tk.W, pady=(6, 0))

        # ── LEFT views ──
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

    # ── mask popup ──
    def toggle_mask_popup(self):
        if self._mask_popup and self._mask_popup.is_open():
            self._mask_popup.close(); self._mask_popup = None
            self.mask_btn.config(text="▶  Mask / Skel View", bg="#2a6496")
        else:
            self._mask_popup = MaskPopup(self.root)
            self._mask_popup.win.protocol("WM_DELETE_WINDOW", self._on_popup_close)
            self.mask_btn.config(text="✕  Close Mask View", bg="#8b3a3a")

    def _on_popup_close(self):
        if self._mask_popup: self._mask_popup.close(); self._mask_popup = None
        self.mask_btn.config(text="▶  Mask / Skel View", bg="#2a6496")

    # ── display helpers ──
    def _container_size(self, container):
        container.update_idletasks()
        w = container.winfo_width(); h = container.winfo_height()
        return (w if w > 1 else 400, h if h > 1 else 300)

    def _show(self, container, label, cv_img, key):
        bw, bh = self._container_size(container)
        pil = _fit_image(cv_img, bw, bh)
        photo = ImageTk.PhotoImage(pil)
        label.configure(image=photo)
        label.place(x=0, y=0, width=bw, height=bh)
        self._photo_refs[key] = photo

    def _show_info(self, cv_img):
        bw = self.SIDEBAR_W - 12; bh = self.INFO_H
        pil = _fit_image(cv_img, bw, bh)
        photo = ImageTk.PhotoImage(pil)
        self.info_lbl.configure(image=photo)
        self._photo_refs["info"] = photo

    def _placeholder(self, container, label, text):
        bw, bh = self._container_size(container)
        img = np.zeros((max(bh, 10), max(bw, 10), 3), np.uint8)
        if text:
            tw = len(text) * 10
            cv2.putText(img, text, (max(bw//2 - tw//2, 5), bh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        self._show(container, label, img, "ph_" + str(id(label)))

    # ── settings ──
    def export_settings(self):
        cfg = {k: max(self.sliders[n].get(), 1) for n, k in
               [("Block Size","block_size"),("C Offset","c_offset"),("Morph Size","morph_size"),
                ("Blur Size","blur_size"),("Close Iter","close_iter")]}
        p = filedialog.asksaveasfilename(title="Export", defaultextension=".json",
                                          filetypes=[("JSON","*.json")], initialfile="tuning_preset.json")
        if p:
            with open(p, 'w') as f: json.dump(cfg, f, indent=2)

    def import_settings(self):
        p = filedialog.askopenfilename(title="Import", filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p) as f: cfg = json.load(f)
            for n, k in [("Block Size","block_size"),("C Offset","c_offset"),("Morph Size","morph_size"),
                          ("Blur Size","blur_size"),("Close Iter","close_iter")]:
                if k in cfg: self.sliders[n].set(cfg[k])
        except Exception as e: print(f"[ERROR] {e}")

    # ── connection ──
    def toggle_connection(self):
        self.disconnect() if self.running else self.connect()

    def connect(self):
        url = self.url_var.get().strip()
        if not url:
            self.status_lbl.config(text="  Enter a URL", fg="#ff5555"); return
        self.status_lbl.config(text="  Connecting...", fg="#ffaa00")
        self.connect_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            self.status_lbl.config(text="  Connection failed", fg="#ff5555")
            self.connect_btn.config(state=tk.NORMAL); cap.release(); return
        self.cap = cap; self.running = True; self.fail_count = 0
        self.H = None; self.raw_corners = None; self.logged_mapping = False
        self.status_lbl.config(text="  Connected", fg="#55ff55")
        self.connect_btn.config(text="Disconnect", bg="#f44336", state=tk.NORMAL)
        self.url_entry.config(state=tk.DISABLED)
        self.settings["stream_url"] = url; save_settings(self.settings)
        self.process_frame()

    def disconnect(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        self.status_lbl.config(text="  Disconnected", fg="#ff5555")
        self.connect_btn.config(text="Connect", bg="#4CAF50")
        self.url_entry.config(state=tk.NORMAL)
        self._placeholder(self.live_container, self.live_lbl, "No camera connected")
        self._placeholder(self.meas_container, self.meas_lbl, "Waiting for feed...")

    # ── main loop ──
    def process_frame(self):
        if not self.running or not self.cap: return

        ret, frame = self.cap.read()
        if not ret:
            self.fail_count += 1
            if self.fail_count >= MAX_FAIL:
                self.status_lbl.config(text="  Feed lost", fg="#ff5555"); self.disconnect(); return
            self.root.after(30, self.process_frame); return
        self.fail_count = 0

        ss = self.skel_var.get()
        bm = self.branch_var.get()
        popup_open = self._mask_popup is not None and self._mask_popup.is_open()

        bsz = max(self.sliders["Block Size"].get(), 3)
        co  = self.sliders["C Offset"].get()
        ms  = max(self.sliders["Morph Size"].get(), 1)
        bls = max(self.sliders["Blur Size"].get(), 1)
        ci  = max(self.sliders["Close Iter"].get(), 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        hH = False; nm = 0
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            self.H, self.wsize, self.raw_corners = homography(corners, ids)
            hH = self.H is not None; nm = len(ids)
            if hH and not self.logged_mapping:
                self.logged_mapping = True
                for i, m in enumerate(ids.flatten()):
                    p = corners[i][0]; cx, cy = p[:,0].mean(), p[:,1].mean()
                    print(f"  [DEBUG] ID {m} ({cx:.0f},{cy:.0f}) -> '{MARKER_CORNER.get(int(m),'?')}'")

        hf, wf = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (wf, 40), (0, 0, 0), -1)
        c = (0,255,0) if nm == 4 else (0,165,255) if nm > 0 else (0,0,255)
        cv2.putText(frame, f"Markers: {nm}/4", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.putText(frame, f"Scale: {SCALE:.4f} mm/px", (220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
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
            cv2.rectangle(disp, (blk, blk), (disp.shape[1]-blk, disp.shape[0]-blk), (80,80,80), 1)

            if bm:
                ncc, la, sa, _ = cv2.connectedComponentsWithStats(sk, 8)
                skm = ((la == (np.argmax(sa[1:, cv2.CC_STAT_AREA])+1))*255).astype(np.uint8) if ncc > 1 else sk.copy()

                res = analyse(skm)
                if res is not None:
                    ti  = res['trunk']
                    bis = res['branches']
                    jis = res['junctions']

                    tmm = ti['len'] * SCALE
                    self.trunk_buf.append(tmm)
                    ta = float(np.mean(self.trunk_buf))

                    # Draw trunk
                    if ss:
                        tp = ti['path']
                        if len(tp) > 1:
                            cv2.polylines(disp, [np.array([(p[1],p[0]) for p in tp], np.int32)], False, (0,0,255), 3)
                            cv2.circle(disp, (tp[0][1],tp[0][0]), 8, (0,0,255), -1)
                            cv2.circle(disp, (tp[-1][1],tp[-1][0]), 8, (0,0,255), -1)

                    # ── Stable tracker update ──
                    tracked = self.tracker.update(bis)
                    # tracked = list of (slot, branch_dict, avg_len_mm, avg_total_mm)
                    # slot is the permanent identity index; use slot % len(BC) for color

                    for slot, br, avg_len, avg_tot in tracked:
                        col = BC[slot % len(BC)]
                        if ss:
                            for lseg in br.get('links', []):
                                lp = lseg['path']
                                if len(lp) > 1:
                                    cv2.polylines(disp, [np.array([(p[1],p[0]) for p in lp], np.int32)], False, col, 3)
                            bp = br['path']
                            if len(bp) > 1:
                                cv2.polylines(disp, [np.array([(p[1],p[0]) for p in bp], np.int32)], False, col, 3)
                                cv2.circle(disp, (bp[0][1],bp[0][0]), 8, col, -1)
                                cv2.circle(disp, (bp[-1][1],bp[-1][0]), 8, col, -1)

                    if ss:
                        for jy, jx in jis:
                            cv2.circle(disp, (jx, jy), 10, (0,255,0), 3)

                    # Sidebar info
                    nb = len(tracked)
                    y0 = 22
                    cv2.putText(info, f"Harness: 1 trunk + {nb} branch(es)", (10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y0 += 30
                    cv2.putText(info, f"Trunk:  {ta:.1f} mm  ({ta/10:.2f} cm)", (10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,255), 1); y0 += 26
                    for slot, br, avg_len, avg_tot in tracked:
                        col = BC[slot % len(BC)]
                        cv2.putText(info, f"Branch {slot+1}: {avg_len:.1f} mm ({avg_len/10:.2f} cm)",
                                    (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1); y0 += 22
                    y0 += 6
                    cv2.line(info, (10,y0), (iw-10,y0), (100,100,100), 1); y0 += 18
                    cv2.putText(info, "Totals (trunk + branch):", (10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1); y0 += 22
                    for slot, br, avg_len, avg_tot in tracked:
                        col = BC[slot % len(BC)]
                        cv2.putText(info, f"Total {slot+1}: {avg_tot:.1f} mm ({avg_tot/10:.2f} cm)",
                                    (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1); y0 += 24
                    cv2.putText(info, f"(avg {len(self.trunk_buf)} frames)",
                                (10, min(y0+6, ih-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
                else:
                    cv2.putText(info, "No junction (simple fallback)", (10,22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)
                    st = simple(sk); y0 = 50
                    for i, (path, lpx) in enumerate(st[:8]):
                        mm = lpx*SCALE; clr = BC[i%len(BC)]
                        if ss and len(path) > 1:
                            cv2.polylines(disp, [np.array(path, np.int32).reshape(-1,1,2)], False, clr, 2)
                            cv2.circle(disp, path[0], 6, clr, -1)
                            cv2.circle(disp, path[-1], 6, clr, -1)
                        cv2.putText(info, f"#{i+1}: {mm:.1f} mm ({mm/10:.2f} cm)",
                                    (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1); y0 += 22
            else:
                st = simple(sk); y0 = 22
                cv2.putText(info, "Simple Strand Mode", (10,y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); y0 += 30
                for i, (path, lpx) in enumerate(st[:8]):
                    mm = lpx*SCALE; clr = BC[i%len(BC)]
                    if ss and len(path) > 1:
                        cv2.polylines(disp, [np.array(path, np.int32).reshape(-1,1,2)], False, clr, 2)
                        cv2.circle(disp, path[0], 6, clr, -1)
                        cv2.circle(disp, path[-1], 6, clr, -1)
                    cv2.putText(info, f"#{i+1}: {mm:.1f} mm ({mm/10:.2f} cm)",
                                (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1); y0 += 22

            cv2.putText(disp, "BRANCH" if bm else "SIMPLE", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            self._show(self.meas_container, self.meas_lbl, disp, "meas")

            if popup_open:
                mb = cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)
                sb = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR); sb[sk > 0] = (0, 255, 0)
                self._mask_popup.update(np.hstack([mb, sb]))
        else:
            self._placeholder(self.meas_container, self.meas_lbl, "Need all 4 ArUco markers")
            if popup_open:
                self._mask_popup.show_placeholder("Need all 4 ArUco markers")

        self._show_info(info)

        self._save_ctr += 1
        if self._save_ctr % 100 == 0:
            self.settings.update({"block_size": bsz, "c_offset": co, "morph_size": ms,
                                  "blur_size": bls, "close_iter": ci})
            save_settings(self.settings)

        self.root.after(30, self.process_frame)

    def on_branch_toggle(self):
        self.trunk_buf.clear()
        self.tracker.reset()

    def screenshot(self):
        print("[INFO] Screenshot saved (if feed active)")

    def on_close(self):
        self.running = False
        if self._mask_popup: self._mask_popup.close()
        if self.cap: self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = WireMeasurerApp()
    app.run()

if __name__ == "__main__":
    main()
