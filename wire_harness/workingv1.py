import cv2
import numpy as np
import sys
from skimage.morphology import skeletonize

# ── CONFIG ────────────────────────────────────────────────────────────────────
STREAM_URL      = "http://10.40.71.26:8080/video"
MARKER_BLACK_MM = 49.0
DICT_TYPE       = cv2.aruco.DICT_5X5_100
MAX_FAIL_FRAMES = 30

# A4 real dimensions between marker CENTERS (mm)
# Markers are at 15mm margin + 35mm (half of 70mm total marker) from each edge
# A4 = 210 x 297 mm
# Horizontal distance between centers: 210 - 2*(15+35) = 210 - 100 = 110 mm
# Vertical distance between centers:   297 - 2*(15+35) = 297 - 100 = 197 mm
BOARD_W_MM = 110.0
BOARD_H_MM = 197.0

# Output warped image size — scaled from real mm (4 px per mm for good resolution)
PX_PER_MM = 4.0
OUT_W = int(BOARD_W_MM * PX_PER_MM)   # 440 px
OUT_H = int(BOARD_H_MM * PX_PER_MM)   # 788 px

DEFAULT_BLOCK_SIZE  = 53
DEFAULT_C_OFFSET    = 13
DEFAULT_MORPH_SIZE  = 5
DEFAULT_BLUR_SIZE   = 11
DEFAULT_CLOSE_ITER  = 15
# ─────────────────────────────────────────────────────────────────────────────

aruco_dict   = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

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
    # Destination maps to real-world mm scaled to pixels
    dst = np.array([
        [0,     0    ],   # ID0 top-left
        [OUT_W, 0    ],   # ID1 top-right
        [0,     OUT_H],   # ID2 bottom-left
        [OUT_W, OUT_H],   # ID3 bottom-right
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H, (OUT_W, OUT_H)

# Scale is now fixed: 1 px = 1/PX_PER_MM mm in the warped image
SCALE_MM_PER_PX = 1.0 / PX_PER_MM   # 0.25 mm/px

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
    min_area = 500
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask

def mask_out_aruco_regions(mask, warped_shape):
    """
    Aggressively mask out the 4 corner regions where ArUco markers,
    borders, and printed text labels live.
    Since we know the warped image maps marker centers to the 4 corners,
    we just black out generous rectangular regions at each corner.
    """
    h, w = warped_shape[:2]

    # Markers are at the corners of the warped image (0,0), (W,0), (0,H), (W,H)
    # The marker + border + text extends ~35mm from each corner in real space
    # In pixels: 35mm * PX_PER_MM
    margin_px = int(40 * PX_PER_MM)  # 40mm generous margin to catch text + borders

    # Black out 4 corners
    mask[0:margin_px, 0:margin_px] = 0                   # top-left
    mask[0:margin_px, w-margin_px:w] = 0                  # top-right
    mask[h-margin_px:h, 0:margin_px] = 0                  # bottom-left
    mask[h-margin_px:h, w-margin_px:w] = 0                # bottom-right

    # Also black out the very edges (paper border, shadows at edge)
    edge_px = int(5 * PX_PER_MM)  # 5mm from all edges
    mask[0:edge_px, :] = 0
    mask[h-edge_px:h, :] = 0
    mask[:, 0:edge_px] = 0
    mask[:, w-edge_px:w] = 0

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
    endpoints = np.argwhere((skel == 1) & (neighbor_count == 1))
    return endpoints

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
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    return float(dists.sum())

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

def draw_strands(display, strands, scale):
    for i, (path, length_px) in enumerate(strands):
        color = STRAND_COLORS[i % len(STRAND_COLORS)]
        length_mm = length_px * scale

        if len(path) > 1:
            pts_arr = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts_arr], False, color, 2, cv2.LINE_AA)

        mid_idx = len(path) // 2
        mx, my = path[mid_idx]

        if length_mm >= 10:
            label = f"#{i+1}: {length_mm:.1f}mm ({length_mm/10:.2f}cm)"
        else:
            label = f"#{i+1}: {length_mm:.1f}mm"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(display, (mx-2, my-th-4), (mx+tw+4, my+4), (0,0,0), -1)
        cv2.putText(display, label, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if len(path) > 0:
            cv2.circle(display, path[0], 6, color, -1)
            cv2.circle(display, path[-1], 6, color, -1)

def draw_summary(display, strands, scale):
    h, w = display.shape[:2]
    panel_h = 30 + len(strands) * 25
    cv2.rectangle(display, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    y = h - panel_h + 20
    cv2.putText(display, f"Detected {len(strands)} strand(s)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, (path, length_px) in enumerate(strands):
        y += 25
        color = STRAND_COLORS[i % len(STRAND_COLORS)]
        length_mm = length_px * scale
        txt = f"  #{i+1}: {length_mm:.1f} mm  ({length_mm/10:.2f} cm)"
        cv2.putText(display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_hud(frame, n_markers, has_H):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
    color = (0,255,0) if n_markers==4 else (0,165,255) if n_markers>0 else (0,0,255)
    cv2.putText(frame, f"Markers: {n_markers}/4", (10,33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Scale: {SCALE_MM_PER_PX:.4f} mm/px (fixed)", (250,33),
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
    print(f"[INFO] Fixed scale: {SCALE_MM_PER_PX:.4f} mm/px")
    print(f"[INFO] Board area: {BOARD_W_MM} x {BOARD_H_MM} mm between markers")
    print(f"[INFO] Warped output: {OUT_W} x {OUT_H} px")
    print("[INFO] Controls: Q=quit  T=skeleton  M=mask  S=screenshot")

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

            # Aggressively mask out corner regions (markers + text + borders)
            wire_mask = mask_out_aruco_regions(wire_mask, warped.shape)

            skeleton = get_skeleton(wire_mask)
            strands = measure_all_strands(skeleton)

            display = warped.copy()

            # Draw measurement zone boundary (where we actually measure)
            margin_px = int(40 * PX_PER_MM)
            edge_px = int(5 * PX_PER_MM)
            cv2.rectangle(display, (margin_px, edge_px), (OUT_W - margin_px, edge_px + 2), (100,100,100), 1)
            cv2.rectangle(display, (edge_px, margin_px), (edge_px + 2, OUT_H - margin_px), (100,100,100), 1)

            if show_skel and len(strands) > 0:
                draw_strands(display, strands, SCALE_MM_PER_PX)
                draw_summary(display, strands, SCALE_MM_PER_PX)

            cv2.putText(display, "Phase 4: Measurement", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
            cv2.imshow("Wire Measurer - Measurement", display)

            if show_mask:
                mask_bgr = cv2.cvtColor(wire_mask, cv2.COLOR_GRAY2BGR)
                skel_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                skel_bgr[skeleton > 0] = (0, 255, 0)
                combined = np.hstack([mask_bgr, skel_bgr])
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
            if strands:
                print("[INFO] Strand measurements:")
                for i, (path, lpx) in enumerate(strands):
                    lmm = lpx * SCALE_MM_PER_PX
                    print(f"         #{i+1}: {lmm:.1f} mm ({lmm/10:.2f} cm)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[RESULT] Tuning values:")
    print(f"         Block Size: {block_size}")
    print(f"         C Offset:   {c_offset}")
    print(f"         Morph Size: {morph_size}")
    print(f"         Blur Size:  {blur_size}")
    print(f"         Close Iter: {close_iter}")

if __name__ == "__main__":
    main()
