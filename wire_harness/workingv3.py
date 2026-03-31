import cv2
import numpy as np
import sys
from collections import deque
from skimage.morphology import skeletonize

# ── CONFIG ────────────────────────────────────────────────────────────────────
STREAM_URL      = "http://192.168.50.145:8080/video"
MARKER_BLACK_MM = 49.0
DICT_TYPE       = cv2.aruco.DICT_5X5_100
MAX_FAIL_FRAMES = 30

# A4 dimensions
A4_W_MM = 210.0
A4_H_MM = 297.0

# Marker centers are at: margin(15mm) + half total marker(35mm) = 50mm from each edge
MARKER_INSET_MM = 50.0   # distance from A4 edge to marker center

# Output resolution
PX_PER_MM = 4.0
OUT_W = int(A4_W_MM * PX_PER_MM)    # 840 px — full A4 width
OUT_H = int(A4_H_MM * PX_PER_MM)    # 1188 px — full A4 height
SCALE_MM_PER_PX = 1.0 / PX_PER_MM   # 0.25 mm/px

# Where marker centers land in the output image (in pixels)
INSET_PX = int(MARKER_INSET_MM * PX_PER_MM)  # 200 px from edge

SMOOTH_WINDOW = 30

DEFAULT_BLOCK_SIZE  = 53
DEFAULT_C_OFFSET    = 13
DEFAULT_MORPH_SIZE  = 5
DEFAULT_BLUR_SIZE   = 11
DEFAULT_CLOSE_ITER  = 5
# ─────────────────────────────────────────────────────────────────────────────

aruco_dict   = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ── SMOOTHING ─────────────────────────────────────────────────────────────────

class StrandSmoother:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.buffers = {}
        self.last_paths = {}
        self.next_id = 0

    def update(self, strands_raw, scale):
        current = []
        for path, length_px in strands_raw:
            current.append((path, length_px * scale))

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
                new_id = self.next_id
                self.next_id += 1
                self.buffers[new_id] = deque(maxlen=self.window_size)
                self.buffers[new_id].append(length_mm)
                self.last_paths[new_id] = path
                matched.append(new_id)
                used_existing.add(new_id)

        stale = []
        for sid in self.buffers:
            if sid not in used_existing:
                if len(self.buffers[sid]) > 0:
                    self.buffers[sid].popleft()
                if len(self.buffers[sid]) == 0:
                    stale.append(sid)
        for sid in stale:
            del self.buffers[sid]
            del self.last_paths[sid]

        result = []
        for sid in matched:
            avg_mm = float(np.mean(self.buffers[sid]))
            path = self.last_paths[sid]
            n_samples = len(self.buffers[sid])
            result.append((sid, path, avg_mm, n_samples))

        result.sort(key=lambda x: x[2], reverse=True)
        return result

# ── ARUCO HELPERS ─────────────────────────────────────────────────────────────

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

    # Map marker centers to their REAL position on the full A4 output
    # Marker centers are INSET_PX from each edge
    dst = np.array([
        [INSET_PX,            INSET_PX           ],   # ID0 top-left
        [OUT_W - INSET_PX,    INSET_PX           ],   # ID1 top-right
        [INSET_PX,            OUT_H - INSET_PX   ],   # ID2 bottom-left
        [OUT_W - INSET_PX,    OUT_H - INSET_PX   ],   # ID3 bottom-right
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    return H, (OUT_W, OUT_H)

# ── SEGMENTATION ──────────────────────────────────────────────────────────────

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
        l_smooth, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs, c_offset
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size * 2 + 1, morph_size * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 500:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask

def mask_out_aruco_regions(mask):
    """
    Black out the 4 corners where ArUco markers + text + borders are.
    Each marker center is at INSET_PX from edges.
    Marker total size = 70mm = 280px. Mask a generous square around each.
    """
    h, w = mask.shape[:2]
    # Each corner block: from edge to INSET_PX + half marker + some text margin
    block = INSET_PX + int(25 * PX_PER_MM)  # marker center + 25mm extra for text

    mask[0:block, 0:block] = 0                  # top-left
    mask[0:block, w-block:w] = 0                # top-right
    mask[h-block:h, 0:block] = 0                # bottom-left
    mask[h-block:h, w-block:w] = 0              # bottom-right

    # Also mask the printed title text at the very top
    top_text = int(15 * PX_PER_MM)  # top 15mm has the title text
    mask[0:top_text, :] = 0

    return mask

# ── SKELETON + MEASUREMENT ────────────────────────────────────────────────────

def get_skeleton(mask):
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton

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
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    r, c = start_point
    path.append((c, r))
    visited[r, c] = True

    while True:
        found_next = False
        for dr, dc in neighbors:
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

def compute_arc_length_px(path):
    if len(path) < 2:
        return 0.0
    pts = np.array(path, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

def measure_all_strands(skeleton):
    num_labels, labels = cv2.connectedComponents(skeleton)
    strands = []
    for label_id in range(1, num_labels):
        component = ((labels == label_id) * 255).astype(np.uint8)
        endpoints = find_endpoints(component)
        if len(endpoints) == 0:
            pts = np.argwhere(component > 0)
            if len(pts) == 0:
                continue
            start = pts[0]
        else:
            start = endpoints[0]
        path = trace_skeleton_path(component, start)
        length_px = compute_arc_length_px(path)
        if length_px > 20:
            strands.append((path, length_px))
    strands.sort(key=lambda x: x[1], reverse=True)
    return strands

# ── DRAWING ───────────────────────────────────────────────────────────────────

STRAND_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 255, 255), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]

def draw_measurement_zone(display):
    """Draw the safe measurement area — between the ArUco corners."""
    block = INSET_PX + int(25 * PX_PER_MM)
    h, w = display.shape[:2]
    # Draw dashed-style zone boundary
    cv2.rectangle(display, (block, block), (w - block, h - block), (80, 80, 80), 1)

def draw_strands(display, smoothed_strands):
    for i, (sid, path, avg_mm, n_samples) in enumerate(smoothed_strands):
        color = STRAND_COLORS[i % len(STRAND_COLORS)]

        if len(path) > 1:
            pts_arr = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts_arr], False, color, 2, cv2.LINE_AA)

        mid_idx = len(path) // 2
        mx, my = path[mid_idx]

        if avg_mm >= 10:
            label = f"#{i+1}: {avg_mm:.1f}mm ({avg_mm/10:.2f}cm)"
        else:
            label = f"#{i+1}: {avg_mm:.1f}mm"

        conf = min(n_samples / SMOOTH_WINDOW, 1.0)
        label += f" [{int(conf*100)}%]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        # Clamp label position to stay inside the image
        lx = min(max(mx, 5), display.shape[1] - tw - 5)
        ly = min(max(my, th + 5), display.shape[0] - 5)
        cv2.rectangle(display, (lx-2, ly-th-4), (lx+tw+4, ly+4), (0,0,0), -1)
        cv2.putText(display, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if len(path) > 0:
            cv2.circle(display, path[0], 6, color, -1)
            cv2.circle(display, path[-1], 6, color, -1)

def draw_summary(display, smoothed_strands):
    h, w = display.shape[:2]
    panel_h = 30 + len(smoothed_strands) * 25
    cv2.rectangle(display, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    y = h - panel_h + 20
    cv2.putText(display, f"Detected {len(smoothed_strands)} strand(s)  (avg over {SMOOTH_WINDOW} frames)",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for i, (sid, path, avg_mm, n_samples) in enumerate(smoothed_strands):
        y += 25
        color = STRAND_COLORS[i % len(STRAND_COLORS)]
        txt = f"  #{i+1}: {avg_mm:.1f} mm  ({avg_mm/10:.2f} cm)"
        cv2.putText(display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_hud(frame, n_markers, has_H):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
    color = (0,255,0) if n_markers==4 else (0,165,255) if n_markers>0 else (0,0,255)
    cv2.putText(frame, f"Markers: {n_markers}/4", (10,33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Scale: {SCALE_MM_PER_PX:.4f} mm/px", (250,33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    hc = (0,255,0) if has_H else (100,100,255)
    ht = "Homography: ON" if has_H else "Need all 4 markers"
    cv2.putText(frame, ht, (10,h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hc, 2)
    cv2.putText(frame, "Q=quit T=toggle M=mask S=save", (w//3,h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

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
    print("[INFO] Controls: Q=quit  T=skeleton  M=mask  S=screenshot")

    cv2.namedWindow("Wire Measurer - Measurement", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuning", 500, 250)
    cv2.createTrackbar("Block Size",  "Tuning", DEFAULT_BLOCK_SIZE, 101, nothing)
    cv2.createTrackbar("C Offset",    "Tuning", DEFAULT_C_OFFSET, 30, nothing)
    cv2.createTrackbar("Morph Size",  "Tuning", DEFAULT_MORPH_SIZE, 15, nothing)
    cv2.createTrackbar("Blur Size",   "Tuning", DEFAULT_BLUR_SIZE, 25, nothing)
    cv2.createTrackbar("Close Iter",  "Tuning", DEFAULT_CLOSE_ITER, 15, nothing)

    smoother    = StrandSmoother(window_size=SMOOTH_WINDOW)
    H           = None
    wsize       = None
    fail_count  = 0
    show_skel   = True
    show_mask   = False

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            print(f"[WARN] Failed to read frame ({fail_count}/{MAX_FAIL_FRAMES})...")
            if fail_count >= MAX_FAIL_FRAMES:
                print("[ERROR] Stream lost. Shutting down.")
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
            H, wsize  = compute_homography(corners, ids)
            has_H     = H is not None
            n_markers = len(ids)

        draw_hud(frame, n_markers, has_H)
        cv2.imshow("Wire Measurer - Live Feed", frame)

        if has_H and wsize:
            warped = cv2.warpPerspective(frame, H, wsize)
            wire_mask = create_wire_mask(warped, block_size, c_offset, morph_size, blur_size, close_iter)
            wire_mask = mask_out_aruco_regions(wire_mask)

            skeleton = get_skeleton(wire_mask)
            strands_raw = measure_all_strands(skeleton)
            smoothed = smoother.update(strands_raw, SCALE_MM_PER_PX)

            display = warped.copy()
            draw_measurement_zone(display)

            if show_skel and len(smoothed) > 0:
                draw_strands(display, smoothed)
                draw_summary(display, smoothed)

            cv2.putText(display, "Phase 4: Measurement (smoothed)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Wire Measurer - Measurement", display)

            if show_mask:
                mask_bgr = cv2.cvtColor(wire_mask, cv2.COLOR_GRAY2BGR)
                skel_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                skel_bgr[skeleton > 0] = (0, 255, 0)
                combined = np.hstack([mask_bgr, skel_bgr])
                cv2.namedWindow("Mask | Skeleton", cv2.WINDOW_NORMAL)
                cv2.imshow("Mask | Skeleton", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit by user.")
            break
        elif key == ord('t'):
            show_skel = not show_skel
            print(f"[INFO] Skeleton overlay: {'ON' if show_skel else 'OFF'}")
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"[INFO] Mask view: {'ON' if show_mask else 'OFF'}")
        elif key == ord('s'):
            cv2.imwrite("screenshot_feed.png", frame)
            if has_H and wsize:
                cv2.imwrite("screenshot_warped.png", warped)
                cv2.imwrite("screenshot_mask.png", wire_mask)
                cv2.imwrite("screenshot_skeleton.png", skeleton)
                cv2.imwrite("screenshot_measurement.png", display)
            print("[INFO] Screenshots saved!")
            if smoothed:
                print("[INFO] Smoothed measurements:")
                for i, (sid, path, avg_mm, ns) in enumerate(smoothed):
                    print(f"         #{i+1}: {avg_mm:.1f} mm ({avg_mm/10:.2f} cm) [{ns} samples]")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
