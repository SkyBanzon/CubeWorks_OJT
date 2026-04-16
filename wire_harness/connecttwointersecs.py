#!/usr/bin/env python3
"""
Phase 4 – Dynamic branch-aware wire harness measurement.
Any number of branches, any number of junctions.
Trunk = longest path between any two endpoints (through junctions).
Each branch = junction to tip.  Total = trunk-to-junction + branch.
"""

import cv2
import numpy as np
import sys
import json
import os
import math
import tkinter as tk
from tkinter import filedialog
from collections import deque
from skimage.morphology import skeletonize

# ── CONFIG ────────────────────────────────────────────────────────────────────
STREAM_URL      = "http://10.40.71.2:8080/video"
DICT_TYPE       = cv2.aruco.DICT_5X5_100
MAX_FAIL        = 30
A4_W, A4_H      = 210.0, 297.0
INSET_MM        = 5 + (57 / 2)
PX_MM           = 4.0
OUT_W, OUT_H    = int(A4_W*PX_MM), int(A4_H*PX_MM)
SCALE           = 1.0/PX_MM
INSET_PX        = int(INSET_MM*PX_MM)
SMOOTH_W        = 30
MIN_SEG         = 15
JN_CLUSTER_R    = 20
JN_MATCH_R      = 25
EP_MATCH_R      = 10
ARUCO_MARGIN_PX = 30
TARGET_MASK_W   = 8
PATH_SMOOTH_WIN = 7
SKEL_BRIDGE_GAP = 12

MARKER_CORNER = {0: "tl", 1: "tr", 2: "bl", 3: "br"}

DEF_BS, DEF_CO, DEF_MS, DEF_BL, DEF_CI = 53, 13, 5, 11, 5

ad = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
det = cv2.aruco.ArucoDetector(ad, cv2.aruco.DetectorParameters())

_tk_root = tk.Tk()
_tk_root.withdraw()

# ── SETTINGS EXPORT / IMPORT ─────────────────────────────────────────────────
def _read_trackbars():
    return {
        "block_size": max(cv2.getTrackbarPos("Block Size", "Tuning"), 3),
        "c_offset":   cv2.getTrackbarPos("C Offset", "Tuning"),
        "morph_size": max(cv2.getTrackbarPos("Morph Size", "Tuning"), 1),
        "blur_size":  max(cv2.getTrackbarPos("Blur Size", "Tuning"), 1),
        "close_iter": max(cv2.getTrackbarPos("Close Iter", "Tuning"), 0),
    }

def _apply_settings(cfg):
    cv2.setTrackbarPos("Block Size", "Tuning", cfg.get("block_size", DEF_BS))
    cv2.setTrackbarPos("C Offset",   "Tuning", cfg.get("c_offset",  DEF_CO))
    cv2.setTrackbarPos("Morph Size", "Tuning", cfg.get("morph_size",DEF_MS))
    cv2.setTrackbarPos("Blur Size",  "Tuning", cfg.get("blur_size", DEF_BL))
    cv2.setTrackbarPos("Close Iter", "Tuning", cfg.get("close_iter",DEF_CI))

def export_settings():
    cfg = _read_trackbars()
    path = filedialog.asksaveasfilename(
        title="Export Tuning Settings",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile="tuning_preset.json",
    )
    if not path:
        print("[INFO] Export cancelled.")
        return
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Settings exported to {path}: {cfg}")

def import_settings():
    path = filedialog.askopenfilename(
        title="Import Tuning Settings",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not path:
        print("[INFO] Import cancelled.")
        return
    try:
        with open(path, 'r') as f:
            cfg = json.load(f)
        _apply_settings(cfg)
        print(f"[INFO] Settings imported from {path}: {cfg}")
    except Exception as e:
        print(f"[ERROR] Failed to import settings: {e}")

# ── HOMOGRAPHY ────────────────────────────────────────────────────────────────
_DST_MAP = {
    "tl": np.array([INSET_PX,           INSET_PX],           dtype=np.float32),
    "tr": np.array([OUT_W - INSET_PX,   INSET_PX],           dtype=np.float32),
    "bl": np.array([INSET_PX,           OUT_H - INSET_PX],   dtype=np.float32),
    "br": np.array([OUT_W - INSET_PX,   OUT_H - INSET_PX],   dtype=np.float32),
}

def homography(corners, ids):
    found = set(ids.flatten().tolist())
    needed = set(MARKER_CORNER.keys())
    if not needed.issubset(found):
        return None, None, None
    centres = {}
    for i, m in enumerate(ids.flatten()):
        mid = int(m)
        if mid in needed:
            p = corners[i][0]
            centres[mid] = np.array([p[:, 0].mean(), p[:, 1].mean()], np.float32)
    src_pts = []
    dst_pts = []
    for mid, corner_label in MARKER_CORNER.items():
        src_pts.append(centres[mid])
        dst_pts.append(_DST_MAP[corner_label])
    src = np.array(src_pts, np.float32)
    dst = np.array(dst_pts, np.float32)
    H, _ = cv2.findHomography(src, dst)
    all_corners = []
    for i in range(len(ids)):
        all_corners.append(corners[i][0])
    return H, (OUT_W, OUT_H), all_corners

# ── SEGMENTATION ──────────────────────────────────────────────────────────────
def wire_mask(fr, bs, co, ms, bl, ci):
    l=cv2.cvtColor(fr,cv2.COLOR_BGR2LAB)[:,:,0]
    l=cv2.createCLAHE(3.0,(8,8)).apply(l)
    l=cv2.bilateralFilter(l,bl|1,75,75)
    m=cv2.adaptiveThreshold(l,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV,bs|1 if bs>=3 else 3,co)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ms,ms))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k,iterations=1)
    k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ms*2+1,ms*2+1))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k2,iterations=ci)
    k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ms*3+1,ms*3+1))
    m=cv2.dilate(m,k3,iterations=1)
    m=cv2.erode(m,k3,iterations=1)
    cn,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    r=np.zeros_like(m)
    for c in cn:
        if cv2.contourArea(c)>=500: cv2.drawContours(r,[c],-1,255,-1)
    return r

def mask_aruco(m, H, raw_corners):
    if raw_corners is None or H is None:
        h,w=m.shape[:2]; b=INSET_PX+int(25*PX_MM)
        m[0:b,0:b]=0; m[0:b,w-b:w]=0; m[h-b:h,0:b]=0; m[h-b:h,w-b:w]=0
        m[0:int(15*PX_MM),:]=0
        return m
    mg = ARUCO_MARGIN_PX
    for quad in raw_corners:
        pts = np.array(quad, dtype=np.float32).reshape(-1,1,2)
        warped_pts = cv2.perspectiveTransform(pts, H).reshape(-1,2)
        x_min = int(np.floor(warped_pts[:,0].min()) - mg)
        x_max = int(np.ceil(warped_pts[:,0].max()) + mg)
        y_min = int(np.floor(warped_pts[:,1].min()) - mg)
        y_max = int(np.ceil(warped_pts[:,1].max()) + mg)
        h,w = m.shape[:2]
        x_min = max(x_min, 0); y_min = max(y_min, 0)
        x_max = min(x_max, w); y_max = min(y_max, h)
        m[y_min:y_max, x_min:x_max] = 0
    return m

# ── ADAPTIVE EROSION ──────────────────────────────────────────────────────────
def adaptive_erode(mask):
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = dt.max()
    if max_d <= TARGET_MASK_W + 2:
        return mask
    ridge = dt[dt > 0]
    if len(ridge) == 0:
        return mask
    hw_low = float(np.percentile(ridge, 25))
    erode_r = max(int(hw_low - TARGET_MASK_W), 0)
    if erode_r < 1:
        return mask
    erode_r = min(erode_r, max(int(hw_low * 0.4), 1))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * erode_r + 1, 2 * erode_r + 1))
    eroded = cv2.erode(mask, k, iterations=1)
    n_before = cv2.connectedComponents(mask, connectivity=8)[0]
    n_after  = cv2.connectedComponents(eroded, connectivity=8)[0]
    if n_after > n_before + 1:
        return mask
    return eroded

# ── SKELETON GAP BRIDGING ─────────────────────────────────────────────────────
def bridge_skeleton_gaps(sk):
    s = (sk > 0).astype(np.uint8)
    ys, xs = np.where(s > 0)
    if len(ys) < 10:
        return sk
    endpoints = []
    for y, x in zip(ys, xs):
        nn = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < s.shape[0] and 0 <= nx < s.shape[1] and s[ny, nx]:
                    nn += 1
        if nn == 1:
            endpoints.append((y, x))
    if len(endpoints) < 2:
        return sk
    ncc, labels = cv2.connectedComponents(s, connectivity=8)
    bridged = sk.copy()
    used = set()
    for i, (y1, x1) in enumerate(endpoints):
        if i in used: continue
        best_j = -1; best_d = SKEL_BRIDGE_GAP + 1
        for j, (y2, x2) in enumerate(endpoints):
            if j <= i or j in used: continue
            if labels[y1, x1] == labels[y2, x2]: continue
            d = math.hypot(y2 - y1, x2 - x1)
            if d < best_d:
                best_d = d; best_j = j
        if best_j >= 0:
            y2, x2 = endpoints[best_j]
            cv2.line(bridged, (x1, y1), (x2, y2), 255, 1)
            used.add(i); used.add(best_j)
    return bridged

# ── PATH SMOOTHING ────────────────────────────────────────────────────────────
def smooth_path(pts, win=PATH_SMOOTH_WIN):
    if len(pts) < win:
        return pts
    arr = np.array(pts, dtype=np.float64)
    hw = win // 2
    smoothed = arr.copy()
    for i in range(hw, len(arr) - hw):
        smoothed[i] = arr[i - hw:i + hw + 1].mean(axis=0)
    return [tuple(p.astype(int)) for p in smoothed]

# ── SKELETON ──────────────────────────────────────────────────────────────────
def skel(m): return (skeletonize(m>0).astype(np.uint8))*255

def n8(s,y,x):
    h,w=s.shape; o=[]
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            if dy==0 and dx==0: continue
            ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w and s[ny,nx]: o.append((ny,nx))
    return o

def trace(s,st):
    v={st}; p=[st]; c=st
    while True:
        nx=None
        for n in n8(s,c[0],c[1]):
            if n not in v: nx=n; break
        if nx is None: break
        p.append(nx); v.add(nx); c=nx
    return p

def alen(pts):
    if len(pts)<2: return 0.0
    a=np.array(pts,np.float64); return float(np.sqrt((np.diff(a,axis=0)**2).sum(axis=1)).sum())

# ── GRAPH-BASED HARNESS ANALYSIS ─────────────────────────────────────────────
def analyse(skeleton):
    """Build a full graph of the skeleton: nodes = endpoints + junction-clusters,
    edges = skeleton segments between them.  Then find the trunk as the longest
    endpoint-to-endpoint path through the graph (Dijkstra on negative weights).
    Everything else hanging off the trunk path is a branch."""

    s = (skeleton > 0).astype(np.uint8)
    ys, xs = np.where(s > 0)
    if len(ys) < 30:
        return None

    # ── 1. classify pixels ────────────────────────────────────────────────
    eps = set(); jns = set()
    for y, x in zip(ys, xs):
        nn = len(n8(s, y, x))
        if nn == 1:   eps.add((y, x))
        elif nn >= 3: jns.add((y, x))
    if not jns:
        return None

    # ── 2. cluster junction pixels ────────────────────────────────────────
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

    jcenters = [(int(np.mean([p[0] for p in cl])),
                 int(np.mean([p[1] for p in cl]))) for cl in clusters]

    # ── 3. remove junction pixels, find segments ──────────────────────────
    sc = s.copy()
    for py, px in jns:
        sc[py, px] = 0

    ncc, labels = cv2.connectedComponents(sc, connectivity=8)
    segs = []
    for lb in range(1, ncc):
        cy, cx = np.where(labels == lb)
        if len(cy) < 5: continue
        comp = (labels == lb).astype(np.uint8)
        se = []
        for yy, xx in zip(cy, cx):
            if len(n8(comp, yy, xx)) == 1:
                se.append((yy, xx))
        if not se: continue
        path = trace(comp, se[0])
        path = smooth_path(path)
        length = alen(path)
        if length < MIN_SEG: continue

        def classify(pt):
            for ep in eps:
                if max(abs(pt[0]-ep[0]), abs(pt[1]-ep[1])) <= EP_MATCH_R:
                    return 'ep', ep
            best_d = 999; best_j = None
            for jc in jcenters:
                d = max(abs(pt[0]-jc[0]), abs(pt[1]-jc[1]))
                if d < best_d:
                    best_d = d; best_j = jc
            if best_d <= JN_MATCH_R:
                return 'jn', best_j
            return 'unk', pt

        t0, n0 = classify(path[0])
        t1, n1 = classify(path[-1])
        segs.append({'path': path, 'len': length,
                     'e0': n0, 't0': t0, 'e1': n1, 't1': t1})

    if len(segs) < 2:
        return None

    # ── 4. build adjacency graph ──────────────────────────────────────────
    # Nodes: each unique endpoint or junction-center (as tuple)
    # Edges: segments connecting two nodes
    def node_key(typ, pt):
        """Return a hashable node id."""
        return pt  # (y, x) tuple — junctions already merged to centers

    # Collect all nodes
    nodes = set()
    for seg in segs:
        nodes.add(node_key(seg['t0'], seg['e0']))
        nodes.add(node_key(seg['t1'], seg['e1']))

    # Adjacency list: node -> [(neighbor_node, seg_index, length)]
    adj = {n: [] for n in nodes}
    for si, seg in enumerate(segs):
        na = node_key(seg['t0'], seg['e0'])
        nb = node_key(seg['t1'], seg['e1'])
        if na in adj and nb in adj:
            adj[na].append((nb, si, seg['len']))
            adj[nb].append((na, si, seg['len']))

    # ── 5. find trunk = longest path between any two endpoints ────────────
    ep_nodes = [node_key(seg['t0'], seg['e0']) for seg in segs if seg['t0'] == 'ep']
    ep_nodes += [node_key(seg['t1'], seg['e1']) for seg in segs if seg['t1'] == 'ep']
    ep_nodes = list(set(ep_nodes))

    if len(ep_nodes) < 2:
        # Fall back: use any two nodes
        ep_nodes = list(nodes)

    # BFS/DFS to find the longest simple path between endpoint pairs
    # For small graphs this is fine (typically < 20 nodes)
    def longest_path_from(start):
        """Return (total_length, [seg_indices], end_node) for longest path
        from start to any other endpoint."""
        best = (0, [], start)
        stack = [(start, 0, [], set())]  # (node, dist, seg_list, visited_nodes)
        while stack:
            cur, dist, path_segs, visited = stack.pop()
            if cur != start and cur in set(ep_nodes) and dist > best[0]:
                best = (dist, list(path_segs), cur)
            for nb, si, length in adj.get(cur, []):
                if nb not in visited and si not in set(path_segs):
                    new_vis = visited | {cur}
                    stack.append((nb, dist + length, path_segs + [si], new_vis))
        return best

    best_trunk = (0, [], None, None)  # (length, seg_indices, start, end)
    for ep in ep_nodes:
        length, seg_ids, end = longest_path_from(ep)
        if length > best_trunk[0]:
            best_trunk = (length, seg_ids, ep, end)

    trunk_len, trunk_seg_ids, trunk_start, trunk_end = best_trunk
    if not trunk_seg_ids:
        return None

    trunk_seg_set = set(trunk_seg_ids)

    # Build combined trunk path (concatenate segment paths in order)
    # Walk the trunk from trunk_start to trunk_end through the segments
    def build_trunk_path(start, seg_indices):
        """Walk segments in order from start node, concatenating paths."""
        remaining = list(seg_indices)
        full_path = []
        cur = start
        while remaining:
            found = False
            for idx in remaining:
                seg = segs[idx]
                na = node_key(seg['t0'], seg['e0'])
                nb = node_key(seg['t1'], seg['e1'])
                if na == cur:
                    full_path.extend(seg['path'])
                    cur = nb; remaining.remove(idx); found = True; break
                elif nb == cur:
                    full_path.extend(seg['path'][::-1])
                    cur = na; remaining.remove(idx); found = True; break
            if not found:
                break
        return full_path

    trunk_path = build_trunk_path(trunk_start, list(trunk_seg_ids))

    # ── 6. collect junctions that lie on the trunk ────────────────────────
    trunk_junctions = set()
    for si in trunk_seg_ids:
        seg = segs[si]
        if seg['t0'] == 'jn': trunk_junctions.add(seg['e0'])
        if seg['t1'] == 'jn': trunk_junctions.add(seg['e1'])

    # ── 7. find branches = non-trunk segments reachable from trunk junctions
    branches = []

    def collect_branches(junc, dist_to_junc, visited_segs):
        """Recursively collect branches from a junction."""
        for si, seg in enumerate(segs):
            if si in visited_segs:
                continue
            na = node_key(seg['t0'], seg['e0'])
            nb = node_key(seg['t1'], seg['e1'])

            conn_node = None; far_node = None; far_type = None
            if seg['t0'] == 'jn' and seg['e0'] == junc:
                conn_node = na; far_node = nb; far_type = seg['t1']
            elif seg['t1'] == 'jn' and seg['e1'] == junc:
                conn_node = nb; far_node = na; far_type = seg['t0']
            # Also check by proximity (in case node_key doesn't match exactly)
            if conn_node is None:
                if seg['t0'] == 'jn' and seg['e0'] is not None:
                    if max(abs(seg['e0'][0]-junc[0]), abs(seg['e0'][1]-junc[1])) <= JN_MATCH_R:
                        conn_node = na; far_node = nb; far_type = seg['t1']
                if conn_node is None and seg['t1'] == 'jn' and seg['e1'] is not None:
                    if max(abs(seg['e1'][0]-junc[0]), abs(seg['e1'][1]-junc[1])) <= JN_MATCH_R:
                        conn_node = nb; far_node = na; far_type = seg['t0']

            if conn_node is None:
                continue

            new_visited = visited_segs | {si}
            total = dist_to_junc + seg['len']

            if far_type in ('ep', 'unk'):
                # Terminal branch
                far_pt = seg['e1'] if far_node == nb else seg['e0']
                branches.append({
                    'path': seg['path'], 'len': seg['len'],
                    'endpoint': far_pt, 'junction': junc, 'total': total
                })
            elif far_type == 'jn':
                # Junction-to-junction segment that's NOT on the trunk
                # -> this is a continuation; recurse from the far junction
                far_jn = seg['e1'] if far_node == nb else seg['e0']
                if far_jn is not None:
                    collect_branches(far_jn, total, new_visited)

    # For each junction on the trunk, find distance along trunk to that junction
    # (from trunk_start), then collect branches
    for junc in trunk_junctions:
        # Compute distance along trunk from trunk_start to this junction
        dist = 0
        for si in trunk_seg_ids:
            seg = segs[si]
            na = node_key(seg['t0'], seg['e0'])
            nb = node_key(seg['t1'], seg['e1'])
            if seg['e0'] == junc or seg['e1'] == junc:
                # Junction is at the start or end of this segment
                # Distance is up to this point
                if seg['e1'] == junc:
                    dist += seg['len']
                break
            dist += seg['len']

        collect_branches(junc, dist, trunk_seg_set)

    if not branches:
        return None

    # ── 8. sort branches by angle from first trunk junction ───────────────
    first_jn = None
    for jc in jcenters:
        if jc in trunk_junctions:
            first_jn = jc; break
    if first_jn is None and jcenters:
        first_jn = jcenters[0]

    def branch_angle(b):
        p = b['path']
        jn = b['junction']
        d0 = (p[0][0]-jn[0])**2 + (p[0][1]-jn[1])**2
        d1 = (p[-1][0]-jn[0])**2 + (p[-1][1]-jn[1])**2
        tip = p[0] if d0 > d1 else p[-1]
        return math.atan2(tip[0]-jn[0], tip[1]-jn[1])
    branches.sort(key=branch_angle)

    return {'trunk': {'path': trunk_path, 'len': trunk_len, 'ep': trunk_start},
            'branches': branches, 'junctions': jcenters}

# ── SIMPLE FALLBACK ───────────────────────────────────────────────────────────
def simple(skeleton):
    ncc,labels=cv2.connectedComponents(skeleton)
    out=[]
    for lb in range(1,ncc):
        comp=((labels==lb)*255).astype(np.uint8)
        su=(comp>0).astype(np.uint8)
        k=np.array([[1,1,1],[1,0,1],[1,1,1]],np.uint8)
        nc=cv2.filter2D(su,-1,k)
        ep=np.argwhere((su==1)&(nc==1))
        st=ep[0] if len(ep) else np.argwhere(comp>0)[0]
        vis=set(); path=[]; r,c=int(st[0]),int(st[1])
        path.append((c,r)); vis.add((r,c)); h,w=comp.shape
        while True:
            f=False
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    if dy==0 and dx==0: continue
                    nr,nc2=r+dy,c+dx
                    if 0<=nr<h and 0<=nc2<w and comp[nr,nc2] and (nr,nc2) not in vis:
                        vis.add((nr,nc2)); r,c=nr,nc2; path.append((c,r)); f=True; break
                if f: break
            if not f: break
        raw_pts = [(p[1],p[0]) for p in path]
        smoothed = smooth_path(raw_pts)
        l = alen(smoothed)
        if l>20:
            draw_pts = [(p[1],p[0]) for p in smoothed]
            out.append((draw_pts, l))
    out.sort(key=lambda x:x[1],reverse=True); return out

# ── COLORS ────────────────────────────────────────────────────────────────────
BC=[(255,150,0),(0,165,255),(255,0,255),(0,255,255),(255,255,0),(128,0,255),(0,255,128),(255,128,0)]

def hud(fr,nm,hH):
    h,w=fr.shape[:2]; cv2.rectangle(fr,(0,0),(w,50),(0,0,0),-1)
    c=(0,255,0) if nm==4 else (0,165,255) if nm>0 else (0,0,255)
    cv2.putText(fr,f"Markers: {nm}/4",(10,33),cv2.FONT_HERSHEY_SIMPLEX,0.9,c,2)
    cv2.putText(fr,f"Scale: {SCALE:.4f} mm/px",(250,33),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(fr,"Homography: ON" if hH else "Need all 4 markers",(10,h-15),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0) if hH else (100,100,255),2)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def nothing(x): pass

def main():
    print(f"[INFO] Connecting to {STREAM_URL} ...")
    print(f"[INFO] Marker-to-corner mapping: {MARKER_CORNER}")
    print(f"[INFO] If homography looks wrong, edit MARKER_CORNER dict at top of file.")
    cap=cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened(): print("[ERROR] Cannot connect."); sys.exit(1)
    print("[INFO] Connected.")

    cv2.namedWindow("Wire Measurer - Measurement",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wire Measurer - Live Feed",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Measurements",cv2.WINDOW_NORMAL); cv2.resizeWindow("Measurements",600,500)
    cv2.namedWindow("Tuning",cv2.WINDOW_NORMAL); cv2.resizeWindow("Tuning",500,250)
    cv2.createTrackbar("Block Size","Tuning",DEF_BS,101,nothing)
    cv2.createTrackbar("C Offset","Tuning",DEF_CO,30,nothing)
    cv2.createTrackbar("Morph Size","Tuning",DEF_MS,15,nothing)
    cv2.createTrackbar("Blur Size","Tuning",DEF_BL,25,nothing)
    cv2.createTrackbar("Close Iter","Tuning",DEF_CI,15,nothing)

    H=None; ws=None; fc=0; ss=True; sm=False; bm=True
    tb=deque(maxlen=SMOOTH_W); bb={}; tob={}
    raw_corners=None
    logged_mapping=False

    while True:
        ret,frame=cap.read()
        if not ret:
            fc+=1
            if fc>=MAX_FAIL: break
            continue
        fc=0

        bs=max(cv2.getTrackbarPos("Block Size","Tuning"),3)
        co=cv2.getTrackbarPos("C Offset","Tuning")
        ms=max(cv2.getTrackbarPos("Morph Size","Tuning"),1)
        bl=max(cv2.getTrackbarPos("Blur Size","Tuning"),1)
        ci=max(cv2.getTrackbarPos("Close Iter","Tuning"),0)

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_=det.detectMarkers(gray)
        hH=False; nm=0
        if ids is not None and len(ids)>0:
            cv2.aruco.drawDetectedMarkers(frame,corners,ids)
            result=homography(corners,ids)
            H,ws,raw_corners=result
            hH=H is not None; nm=len(ids)
            if hH and not logged_mapping:
                logged_mapping=True
                for i,m in enumerate(ids.flatten()):
                    p=corners[i][0]
                    cx,cy=p[:,0].mean(),p[:,1].mean()
                    label=MARKER_CORNER.get(int(m),"?")
                    print(f"  [DEBUG] Marker ID {m} at camera ({cx:.0f},{cy:.0f}) -> corner '{label}'")

        hud(frame,nm,hH); cv2.imshow("Wire Measurer - Live Feed",frame)

        iw,ih=600,500; info=np.zeros((ih,iw,3),np.uint8)

        if hH and ws:
            warped=cv2.warpPerspective(frame,H,ws)
            wm=wire_mask(warped,bs,co,ms,bl,ci)
            wm=mask_aruco(wm, H, raw_corners)
            wm=adaptive_erode(wm)
            sk=skel(wm)
            sk=bridge_skeleton_gaps(sk)

            disp=warped.copy()
            blk=INSET_PX+int(25*PX_MM); dh,dw=disp.shape[:2]
            cv2.rectangle(disp,(blk,blk),(dw-blk,dh-blk),(80,80,80),1)

            if bm:
                ncc,la,sa,_=cv2.connectedComponentsWithStats(sk,8)
                if ncc>1:
                    big=np.argmax(sa[1:,cv2.CC_STAT_AREA])+1
                    skm=((la==big)*255).astype(np.uint8)
                else: skm=sk.copy()

                res=analyse(skm)
                if res is not None:
                    ti=res['trunk']; bis=res['branches']; jis=res['junctions']
                    tmm=ti['len']*SCALE; tb.append(tmm); ta=float(np.mean(tb))

                    if ss:
                        tp=ti['path']
                        if len(tp)>1:
                            cv2.polylines(disp,[np.array([(p[1],p[0]) for p in tp],np.int32)],False,(0,0,255),3)
                            cv2.circle(disp,(tp[0][1],tp[0][0]),8,(0,0,255),-1)
                            cv2.circle(disp,(tp[-1][1],tp[-1][0]),8,(0,0,255),-1)

                    for i,br in enumerate(bis):
                        bmm=br['len']*SCALE; totmm=br['total']*SCALE
                        if i not in bb: bb[i]=deque(maxlen=SMOOTH_W); tob[i]=deque(maxlen=SMOOTH_W)
                        bb[i].append(bmm); tob[i].append(totmm)
                        col=BC[i%len(BC)]
                        if ss:
                            bp=br['path']
                            if len(bp)>1:
                                cv2.polylines(disp,[np.array([(p[1],p[0]) for p in bp],np.int32)],False,col,3)
                                cv2.circle(disp,(bp[0][1],bp[0][0]),8,col,-1)
                                cv2.circle(disp,(bp[-1][1],bp[-1][0]),8,col,-1)

                    if ss:
                        for jy,jx in jis:
                            cv2.circle(disp,(jx,jy),10,(0,255,0),3)

                    y0=30; nb=len(bis)
                    cv2.putText(info,f"Harness: 1 trunk + {nb} branch(es)",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); y0+=40
                    cv2.putText(info,f"Trunk (red):  {ta:.1f} mm  ({ta/10:.2f} cm)",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2); y0+=30
                    for i in range(nb):
                        col=BC[i%len(BC)]; ba=float(np.mean(bb[i])) if i in bb else 0
                        cv2.putText(info,f"Branch {i+1}:     {ba:.1f} mm  ({ba/10:.2f} cm)",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1); y0+=25
                    y0+=10; cv2.line(info,(10,y0),(iw-10,y0),(100,100,100),1); y0+=25
                    cv2.putText(info,"Totals (trunk + branch):",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1); y0+=30
                    for i in range(nb):
                        col=BC[i%len(BC)]; tota=float(np.mean(tob[i])) if i in tob else 0
                        cv2.putText(info,f"Total {i+1}:  {tota:.1f} mm  ({tota/10:.2f} cm)",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2); y0+=30
                    cv2.putText(info,f"(avg over {len(tb)} frames)",(10,min(y0+10,ih-10)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)
                else:
                    cv2.putText(info,"No junction found (simple fallback)",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),1)
                    st=simple(sk); y0=60
                    for i,(path,lpx) in enumerate(st[:6]):
                        mm=lpx*SCALE; clr=BC[i%len(BC)]
                        if ss and len(path)>1:
                            cv2.polylines(disp,[np.array(path,np.int32).reshape(-1,1,2)],False,clr,2)
                            cv2.circle(disp,path[0],6,clr,-1); cv2.circle(disp,path[-1],6,clr,-1)
                        cv2.putText(info,f"#{i+1}: {mm:.1f} mm ({mm/10:.2f} cm)",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,1); y0+=25
            else:
                st=simple(sk); y0=30
                cv2.putText(info,"Simple Strand Mode",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); y0+=35
                for i,(path,lpx) in enumerate(st[:8]):
                    mm=lpx*SCALE; clr=BC[i%len(BC)]
                    if ss and len(path)>1:
                        cv2.polylines(disp,[np.array(path,np.int32).reshape(-1,1,2)],False,clr,2)
                        cv2.circle(disp,path[0],6,clr,-1); cv2.circle(disp,path[-1],6,clr,-1)
                    cv2.putText(info,f"#{i+1}: {mm:.1f} mm ({mm/10:.2f} cm)",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,1); y0+=25

            cv2.putText(disp,"BRANCH mode (B to toggle)" if bm else "SIMPLE mode (B to toggle)",
                        (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.imshow("Wire Measurer - Measurement",disp)

            if sm:
                mb=cv2.cvtColor(wm,cv2.COLOR_GRAY2BGR); sb=cv2.cvtColor(sk,cv2.COLOR_GRAY2BGR); sb[sk>0]=(0,255,0)
                cv2.namedWindow("Mask | Skeleton",cv2.WINDOW_NORMAL)
                cv2.imshow("Mask | Skeleton",np.hstack([mb,sb]))

        cv2.imshow("Measurements",info)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('t'): ss=not ss
        elif key==ord('m'): sm=not sm
        elif key==ord('b'):
            bm=not bm; tb.clear(); bb.clear(); tob.clear()
            print(f"[INFO] {'BRANCH' if bm else 'SIMPLE'}")
        elif key==ord('s'):
            cv2.imwrite("screenshot_feed.png",frame)
            if hH: cv2.imwrite("screenshot_measurement.png",disp)
            print("[INFO] Saved!")
        elif key==ord('e'):
            export_settings()
        elif key==ord('i'):
            import_settings()

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__": main()
