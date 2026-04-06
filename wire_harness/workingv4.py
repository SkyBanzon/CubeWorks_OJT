#!/usr/bin/env python3
"""
Phase 4 – Branch-aware wire harness measurement with ArUco homography.
Uses proven homography from working code + new Y-harness branch detection.
Measures: trunk (red) + branch1 (blue) + branch2 (orange).
Reports total1 = trunk+branch1, total2 = trunk+branch2.
Measurements shown in a SEPARATE info window.
"""

import cv2
import numpy as np
import sys
from collections import deque
from skimage.morphology import skeletonize

# ── CONFIG ────────────────────────────────────────────────────────────────────
STREAM_URL      = "http://10.40.71.46:8080/video"
MARKER_BLACK_MM = 49.0
DICT_TYPE       = cv2.aruco.DICT_5X5_100
MAX_FAIL_FRAMES = 30

# A4 dimensions
A4_W_MM = 210.0
A4_H_MM = 297.0

# Marker centers are at: margin(15mm) + half total marker(35mm) = 50mm from each edge
MARKER_INSET_MM = 50.0

# Output resolution
PX_PER_MM = 4.0
OUT_W = int(A4_W_MM * PX_PER_MM)    # 840 px
OUT_H = int(A4_H_MM * PX_PER_MM)    # 1188 px
SCALE_MM_PER_PX = 1.0 / PX_PER_MM   # 0.25 mm/px

INSET_PX = int(MARKER_INSET_MM * PX_PER_MM)  # 200 px from edge

SMOOTH_WINDOW = 30

DEFAULT_BLOCK_SIZE  = 53
DEFAULT_C_OFFSET    = 13
DEFAULT_MORPH_SIZE  = 5
DEFAULT_BLUR_SIZE   = 11
DEFAULT_CLOSE_ITER  = 5

# Branch detection
MIN_BRANCH_PX = 40   # ignore skeleton spurs shorter than this
# ─────────────────────────────────────────────────────────────────────────────

aruco_dict   = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ── ARUCO HELPERS (from working code) ─────────────────────────────────────────

def get_marker_centers(corners, ids):
    centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]
        centers[int(marker_id)] = np.array([pts[:,0].mean(), pts[:,1].mean()], dtype=np.float32)
    return centers

def compute_homography(corners, ids):
    if not {0,1,2,3}.issubset(set(ids.flatten().tolist())):
        return None, None
    centers = get_marker_centers(corners, ids)
    src = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)
    dst = np.array([
        [INSET_PX,            INSET_PX           ],   # ID0 top-left
        [OUT_W - INSET_PX,    INSET_PX           ],   # ID1 top-right
        [INSET_PX,            OUT_H - INSET_PX   ],   # ID2 bottom-left
        [OUT_W - INSET_PX,    OUT_H - INSET_PX   ],   # ID3 bottom-right
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H, (OUT_W, OUT_H)

# ── SEGMENTATION (from working code) ──────────────────────────────────────────

def create_wire_mask(frame, block_size, c_offset, morph_size, blur_size, close_iter):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l_channel)
    blur_d = blur_size if blur_size % 2 == 1 else blur_size + 1
    blur_d = max(blur_d, 1)
    l_smooth = cv2.bilateralFilter(l_norm, blur_d, 75, 75)
    bs = block_size if block_size % 2 == 1 else block_size + 1
    bs = max(bs, 3)
    mask = cv2.adaptiveThreshold(
        l_smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, bs, c_offset)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (morph_size * 2 + 1, morph_size * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 500:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    return clean_mask

def mask_out_aruco_regions(mask):
    h, w = mask.shape[:2]
    block = INSET_PX + int(25 * PX_PER_MM)
    mask[0:block, 0:block] = 0
    mask[0:block, w-block:w] = 0
    mask[h-block:h, 0:block] = 0
    mask[h-block:h, w-block:w] = 0
    top_text = int(15 * PX_PER_MM)
    mask[0:top_text, :] = 0
    return mask

# ── SKELETON HELPERS ──────────────────────────────────────────────────────────

def get_skeleton(mask):
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton

def find_neighbors(skel, y, x):
    h, w = skel.shape
    nbrs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                nbrs.append((ny, nx))
    return nbrs

def trace_from(skel, start):
    """Trace skeleton pixels from start, return ordered list of (y,x)."""
    visited = set()
    ordered = [start]
    visited.add(start)
    cur = start
    while True:
        nbrs = find_neighbors(skel, cur[0], cur[1])
        next_pt = None
        for n in nbrs:
            if n not in visited:
                next_pt = n
                break
        if next_pt is None:
            break
        ordered.append(next_pt)
        visited.add(next_pt)
        cur = next_pt
    return ordered

def arc_length_px(pts):
    if len(pts) < 2:
        return 0.0
    arr = np.array(pts, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

# ── BRANCH-AWARE ANALYSIS ─────────────────────────────────────────────────────

def analyse_harness(skeleton):
    """
    Detect Y-shaped harness in skeleton.
    Returns dict with keys: 'trunk', 'branch1', 'branch2', 'junction',
    each containing list of (y,x) pixel coords.
    Returns None if no Y-shape found — falls back to simple strand measurement.
    """
    skel = (skeleton > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(ys) < MIN_BRANCH_PX:
        return None

    # Classify pixels
    endpoints = []
    junctions = []
    for y, x in zip(ys, xs):
        n = len(find_neighbors(skel, y, x))
        if n == 1:
            endpoints.append((y, x))
        elif n >= 3:
            junctions.append((y, x))

    if len(junctions) == 0 or len(endpoints) < 3:
        return None

    # Cluster junction pixels (adjacent junction pixels are one logical junction)
    junction_clusters = []
    used = set()
    for i, jp in enumerate(junctions):
        if i in used:
            continue
        cluster = [jp]
        used.add(i)
        stack = [jp]
        while stack:
            cur = stack.pop()
            for j, jp2 in enumerate(junctions):
                if j in used:
                    continue
                dist = abs(cur[0]-jp2[0]) + abs(cur[1]-jp2[1])
                if dist <= 5:
                    cluster.append(jp2)
                    used.add(j)
                    stack.append(jp2)
        junction_clusters.append(cluster)

    # Use largest cluster as THE junction
    junction_cluster = max(junction_clusters, key=len)
    jy = int(np.mean([p[0] for p in junction_cluster]))
    jx = int(np.mean([p[1] for p in junction_cluster]))

    # Cut skeleton at junction to separate branches
    skel_cut = skel.copy()
    # Remove junction pixels + small radius
    for py, px in junction_cluster:
        skel_cut[py, px] = 0
    # Also blank a small radius around junction center
    cv2.circle(skel_cut, (jx, jy), 4, 0, -1)

    # Find connected components in cut skeleton
    n_cc, labels_cc = cv2.connectedComponents(skel_cut, connectivity=8)

    # For each component, find its endpoints and trace
    branches = []
    for lbl in range(1, n_cc):
        comp_mask = (labels_cc == lbl).astype(np.uint8)
        comp_ys, comp_xs = np.where(comp_mask > 0)
        if len(comp_ys) < MIN_BRANCH_PX:
            continue

        # Find endpoints of this component
        comp_endpoints = []
        for cy, cx in zip(comp_ys, comp_xs):
            nbrs = find_neighbors(comp_mask, cy, cx)
            if len(nbrs) == 1:
                comp_endpoints.append((cy, cx))

        if len(comp_endpoints) == 0:
            # Use pixel farthest from junction
            dists = (comp_ys - jy)**2 + (comp_xs - jx)**2
            far_idx = np.argmax(dists)
            start = (comp_ys[far_idx], comp_xs[far_idx])
        else:
            # Use endpoint farthest from junction
            dists = [(ep[0]-jy)**2 + (ep[1]-jx)**2 for ep in comp_endpoints]
            start = comp_endpoints[np.argmax(dists)]

        path = trace_from(comp_mask, start)
        length = arc_length_px(path)
        if length >= MIN_BRANCH_PX:
            branches.append((path, length))

    if len(branches) < 2:
        return None

    # Sort by length descending — longest is trunk
    branches.sort(key=lambda b: b[1], reverse=True)

    trunk_path = branches[0][0]
    branch1_path = branches[1][0]
    branch2_path = branches[2][0] if len(branches) >= 3 else None

    return {
        'trunk': trunk_path,
        'branch1': branch1_path,
        'branch2': branch2_path,
        'junction': (jy, jx),
    }

# ── SIMPLE STRAND MEASUREMENT (fallback) ──────────────────────────────────────

def find_endpoints(skeleton):
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    return np.argwhere((skel == 1) & (neighbor_count == 1))

def trace_skeleton_path(skeleton, start_point):
    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    path = []
    neighbors_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    r, c = start_point
    path.append((c, r))
    visited[r, c] = True
    while True:
        found_next = False
        for dr, dc in neighbors_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                r, c = nr, nc
                path.append((c, r))
                found_next = True
                break
        if not found_next:
            break
    return path

def measure_all_strands(skeleton):
    num_labels, labels = cv2.connectedComponents(skeleton)
    strands = []
    for label_id in range(1, num_labels):
        component = ((labels == label_id) * 255).astype(np.uint8)
        eps = find_endpoints(component)
        if len(eps) == 0:
            pts = np.argwhere(component > 0)
            if len(pts) == 0:
                continue
            start = pts[0]
        else:
            start = eps[0]
        path = trace_skeleton_path(component, start)
        length_px_val = arc_length_px([(p[1], p[0]) for p in path])  # convert (x,y) to (y,x)
        if length_px_val > 20:
            strands.append((path, length_px_val))
    strands.sort(key=lambda x: x[1], reverse=True)
    return strands

# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_hud(frame, n_markers, has_H):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
    color = (0,255,0) if n_markers==4 else (0,165,255) if n_markers>0 else (0,0,255)
    cv2.putText(frame, f"Markers: {n_markers}/4", (10,33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Scale: {SCALE_MM_PER_PX:.4f} mm/px", (250,33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    hc = (0,255,0) if has_H else (100,100,255)
    ht = "Homography: ON" if has_H else "Need all 4 markers"
    cv2.putText(frame, ht, (10,h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hc, 2)
    cv2.putText(frame, "Q=quit  T=toggle  M=mask  S=save  B=branch mode",
                (w//3, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

def draw_measurement_zone(display):
    block = INSET_PX + int(25 * PX_PER_MM)
    h, w = display.shape[:2]
    cv2.rectangle(display, (block, block), (w - block, h - block), (80, 80, 80), 1)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def nothing(x): pass

def main():
    print(f"[INFO] Connecting to {STREAM_URL} ...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("[ERROR] Could not connect to stream.")
        print(f"[ERROR] URL tried: {STREAM_URL}")
        print("[ERROR] Possible causes:")
        print("        - IP Webcam app is not running on your phone")
        print("        - Phone and laptop are not on the same WiFi")
        print(f"        - Try opening {STREAM_URL} in your browser to verify")
        sys.exit(1)

    print("[INFO] Stream connected!")
    print(f"[INFO] Full A4 output: {OUT_W}x{OUT_H} px ({A4_W_MM}x{A4_H_MM} mm)")
    print(f"[INFO] Scale: {SCALE_MM_PER_PX:.4f} mm/px")
    print("[INFO] Controls: Q=quit  T=skeleton  M=mask  S=screenshot  B=branch mode")

    cv2.namedWindow("Wire Measurer - Measurement", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wire Measurer - Live Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Measurements", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Measurements", 550, 350)
    cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuning", 500, 250)
    cv2.createTrackbar("Block Size",  "Tuning", DEFAULT_BLOCK_SIZE, 101, nothing)
    cv2.createTrackbar("C Offset",    "Tuning", DEFAULT_C_OFFSET, 30, nothing)
    cv2.createTrackbar("Morph Size",  "Tuning", DEFAULT_MORPH_SIZE, 15, nothing)
    cv2.createTrackbar("Blur Size",   "Tuning", DEFAULT_BLUR_SIZE, 25, nothing)
    cv2.createTrackbar("Close Iter",  "Tuning", DEFAULT_CLOSE_ITER, 15, nothing)

    H           = None
    wsize       = None
    fail_count  = 0
    show_skel   = True
    show_mask   = False
    branch_mode = True  # start in branch mode

    # Smoothing buffers for branch mode
    trunk_buf  = deque(maxlen=SMOOTH_WINDOW)
    b1_buf     = deque(maxlen=SMOOTH_WINDOW)
    b2_buf     = deque(maxlen=SMOOTH_WINDOW)

    # Smoothing for simple mode (fallback)
    class SimpleSmoother:
        def __init__(self):
            self.buffers = {}
            self.last_paths = {}
            self.next_id = 0
        def update(self, strands_raw, scale):
            current = [(p, l * scale) for p, l in strands_raw]
            used_existing = set()
            matched = []
            for path, length_mm in current:
                best_id = None
                best_diff = float('inf')
                for sid, buf in self.buffers.items():
                    if sid in used_existing:
                        continue
                    avg = np.mean(buf)
                    diff = abs(length_mm - avg)
                    if diff < best_diff and diff < avg * 0.5:
                        best_diff = diff
                        best_id = sid
                if best_id is not None:
                    used_existing.add(best_id)
                    self.buffers[best_id].append(length_mm)
                    self.last_paths[best_id] = path
                    matched.append(best_id)
                else:
                    new_id = self.next_id; self.next_id += 1
                    self.buffers[new_id] = deque(maxlen=SMOOTH_WINDOW)
                    self.buffers[new_id].append(length_mm)
                    self.last_paths[new_id] = path
                    matched.append(new_id)
                    used_existing.add(new_id)
            stale = [s for s in self.buffers if s not in used_existing]
            for s in list(self.buffers.keys()):
                if s not in used_existing:
                    self.buffers[s].popleft()
                    if len(self.buffers[s]) == 0:
                        del self.buffers[s]
                        if s in self.last_paths:
                            del self.last_paths[s]
            result = []
            for sid in matched:
                avg_mm = float(np.mean(self.buffers[sid]))
                result.append((sid, self.last_paths[sid], avg_mm, len(self.buffers[sid])))
            result.sort(key=lambda x: x[2], reverse=True)
            return result

    simple_smoother = SimpleSmoother()

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count >= MAX_FAIL_FRAMES:
                print("[ERROR] Stream lost.")
                break
            continue
        else:
            if fail_count > 0:
                print("[INFO] Stream recovered.")
            fail_count = 0

        block_size = max(cv2.getTrackbarPos("Block Size", "Tuning"), 3)
        c_offset   = cv2.getTrackbarPos("C Offset", "Tuning")
        morph_size = max(cv2.getTrackbarPos("Morph Size", "Tuning"), 1)
        blur_size  = max(cv2.getTrackbarPos("Blur Size", "Tuning"), 1)
        close_iter = max(cv2.getTrackbarPos("Close Iter", "Tuning"), 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        has_H = False
        n_markers = 0

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            H, wsize = compute_homography(corners, ids)
            has_H = H is not None
            n_markers = len(ids)

        draw_hud(frame, n_markers, has_H)
        cv2.imshow("Wire Measurer - Live Feed", frame)

        # Info panel
        info_h, info_w = 350, 550
        info_img = np.zeros((info_h, info_w, 3), dtype=np.uint8)

        if has_H and wsize:
            warped = cv2.warpPerspective(frame, H, wsize)
            wire_mask = create_wire_mask(warped, block_size, c_offset,
                                         morph_size, blur_size, close_iter)
            wire_mask = mask_out_aruco_regions(wire_mask)
            skeleton = get_skeleton(wire_mask)

            display = warped.copy()
            draw_measurement_zone(display)

            if branch_mode:
                # ── BRANCH-AWARE MODE ──
                # Keep only largest connected component in skeleton
                n_cc_all, labels_all, stats_all, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
                if n_cc_all > 1:
                    areas = stats_all[1:, cv2.CC_STAT_AREA]
                    largest_lbl = np.argmax(areas) + 1
                    skeleton_main = ((labels_all == largest_lbl) * 255).astype(np.uint8)
                else:
                    skeleton_main = skeleton.copy()

                result = analyse_harness(skeleton_main)

                if result is not None:
                    trunk_path = result['trunk']
                    branch1_path = result['branch1']
                    branch2_path = result['branch2']
                    junction = result['junction']

                    trunk_mm = arc_length_px(trunk_path) * SCALE_MM_PER_PX
                    b1_mm = arc_length_px(branch1_path) * SCALE_MM_PER_PX
                    b2_mm = arc_length_px(branch2_path) * SCALE_MM_PER_PX if branch2_path else 0.0

                    trunk_buf.append(trunk_mm)
                    b1_buf.append(b1_mm)
                    if branch2_path:
                        b2_buf.append(b2_mm)

                    trunk_avg = float(np.mean(trunk_buf))
                    b1_avg = float(np.mean(b1_buf))
                    b2_avg = float(np.mean(b2_buf)) if b2_buf else 0.0
                    total1 = trunk_avg + b1_avg
                    total2 = trunk_avg + b2_avg

                    if show_skel:
                        # Draw trunk (RED) — convert (y,x) to (x,y) for drawing
                        if len(trunk_path) > 1:
                            pts_draw = np.array([(p[1], p[0]) for p in trunk_path], dtype=np.int32)
                            cv2.polylines(display, [pts_draw], False, (0, 0, 255), 3)
                            cv2.circle(display, (trunk_path[0][1], trunk_path[0][0]), 8, (0, 0, 255), -1)
                            cv2.circle(display, (trunk_path[-1][1], trunk_path[-1][0]), 8, (0, 0, 255), -1)

                        # Draw branch1 (BLUE)
                        if len(branch1_path) > 1:
                            pts_draw = np.array([(p[1], p[0]) for p in branch1_path], dtype=np.int32)
                            cv2.polylines(display, [pts_draw], False, (255, 150, 0), 3)
                            cv2.circle(display, (branch1_path[0][1], branch1_path[0][0]), 8, (255, 150, 0), -1)
                            cv2.circle(display, (branch1_path[-1][1], branch1_path[-1][0]), 8, (255, 150, 0), -1)

                        # Draw branch2 (ORANGE)
                        if branch2_path and len(branch2_path) > 1:
                            pts_draw = np.array([(p[1], p[0]) for p in branch2_path], dtype=np.int32)
                            cv2.polylines(display, [pts_draw], False, (0, 165, 255), 3)
                            cv2.circle(display, (branch2_path[0][1], branch2_path[0][0]), 8, (0, 165, 255), -1)
                            cv2.circle(display, (branch2_path[-1][1], branch2_path[-1][0]), 8, (0, 165, 255), -1)

                        # Draw junction (GREEN circle)
                        cv2.circle(display, (junction[1], junction[0]), 10, (0, 255, 0), 3)

                    # ── Info panel ──
                    y0 = 30
                    cv2.putText(info_img, "Y-Harness Branch Measurements", (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y0 += 15
                    cv2.line(info_img, (10, y0), (info_w - 10, y0), (100, 100, 100), 1)
                    y0 += 30

                    cv2.putText(info_img,
                                f"Trunk (red):      {trunk_avg:.1f} mm  ({trunk_avg/10:.2f} cm)",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    y0 += 30
                    cv2.putText(info_img,
                                f"Branch 1 (blue):  {b1_avg:.1f} mm  ({b1_avg/10:.2f} cm)",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 2)
                    y0 += 30
                    if branch2_path:
                        cv2.putText(info_img,
                                    f"Branch 2 (orange): {b2_avg:.1f} mm  ({b2_avg/10:.2f} cm)",
                                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                        y0 += 30

                    y0 += 10
                    cv2.line(info_img, (10, y0), (info_w - 10, y0), (100, 100, 100), 1)
                    y0 += 30

                    cv2.putText(info_img,
                                f"Total 1 (trunk+blue):   {total1:.1f} mm  ({total1/10:.2f} cm)",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
                    y0 += 35
                    if branch2_path:
                        cv2.putText(info_img,
                                    f"Total 2 (trunk+orange): {total2:.1f} mm  ({total2/10:.2f} cm)",
                                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 2)
                        y0 += 35

                    y0 += 10
                    cv2.putText(info_img, f"(avg over {len(trunk_buf)} frames)",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                else:
                    cv2.putText(info_img, "No Y-junction found — showing simple strands", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    # Fallback to simple mode
                    strands_raw = measure_all_strands(skeleton)
                    smoothed = simple_smoother.update(strands_raw, SCALE_MM_PER_PX)
                    if show_skel:
                        STRAND_COLORS = [(0,255,0),(255,100,0),(0,255,255),(255,0,255),
                                         (0,165,255),(255,255,0),(128,0,255),(0,128,255)]
                        for i, (sid, path, avg_mm, ns) in enumerate(smoothed):
                            color = STRAND_COLORS[i % len(STRAND_COLORS)]
                            if len(path) > 1:
                                pts_arr = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
                                cv2.polylines(display, [pts_arr], False, color, 2)
                            if len(path) > 0:
                                cv2.circle(display, path[0], 6, color, -1)
                                cv2.circle(display, path[-1], 6, color, -1)
                    y0 = 60
                    for i, (sid, path, avg_mm, ns) in enumerate(smoothed[:8]):
                        cv2.putText(info_img, f"#{i+1}: {avg_mm:.1f} mm ({avg_mm/10:.2f} cm)",
                                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y0 += 25
            else:
                # ── SIMPLE MODE ──
                strands_raw = measure_all_strands(skeleton)
                smoothed = simple_smoother.update(strands_raw, SCALE_MM_PER_PX)
                if show_skel:
                    STRAND_COLORS = [(0,255,0),(255,100,0),(0,255,255),(255,0,255),
                                     (0,165,255),(255,255,0),(128,0,255),(0,128,255)]
                    for i, (sid, path, avg_mm, ns) in enumerate(smoothed):
                        color = STRAND_COLORS[i % len(STRAND_COLORS)]
                        if len(path) > 1:
                            pts_arr = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(display, [pts_arr], False, color, 2)
                        if len(path) > 0:
                            cv2.circle(display, path[0], 6, color, -1)
                            cv2.circle(display, path[-1], 6, color, -1)
                y0 = 30
                cv2.putText(info_img, "Simple Strand Mode", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y0 += 35
                for i, (sid, path, avg_mm, ns) in enumerate(smoothed[:10]):
                    cv2.putText(info_img, f"#{i+1}: {avg_mm:.1f} mm ({avg_mm/10:.2f} cm)",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y0 += 25

            mode_txt = "BRANCH mode (B to toggle)" if branch_mode else "SIMPLE mode (B to toggle)"
            cv2.putText(display, mode_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Wire Measurer - Measurement", display)

            if show_mask:
                mask_bgr = cv2.cvtColor(wire_mask, cv2.COLOR_GRAY2BGR)
                skel_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                skel_bgr[skeleton > 0] = (0, 255, 0)
                combined = np.hstack([mask_bgr, skel_bgr])
                cv2.namedWindow("Mask | Skeleton", cv2.WINDOW_NORMAL)
                cv2.imshow("Mask | Skeleton", combined)

        cv2.imshow("Measurements", info_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit.")
            break
        elif key == ord('t'):
            show_skel = not show_skel
            print(f"[INFO] Skeleton overlay: {'ON' if show_skel else 'OFF'}")
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"[INFO] Mask view: {'ON' if show_mask else 'OFF'}")
        elif key == ord('b'):
            branch_mode = not branch_mode
            trunk_buf.clear(); b1_buf.clear(); b2_buf.clear()
            print(f"[INFO] Mode: {'BRANCH' if branch_mode else 'SIMPLE'}")
        elif key == ord('s'):
            cv2.imwrite("screenshot_feed.png", frame)   
            if has_H and wsize:
                cv2.imwrite("screenshot_warped.png", warped)
                cv2.imwrite("screenshot_mask.png", wire_mask)
                cv2.imwrite("screenshot_skeleton.png", skeleton)
                cv2.imwrite("screenshot_measurement.png", display)
            print("[INFO] Screenshots saved!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
