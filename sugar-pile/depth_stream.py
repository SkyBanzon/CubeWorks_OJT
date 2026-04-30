import numpy as np
import cv2
from pykinect import nui

W, H   = 640, 480
FX, FY = 585.0, 585.0

def measure_volume(frame):
    Z = frame.astype(np.float32) / 1000.0
    Z[(Z > 3.5) | (Z < 0.5)] = 0

    valid_z = Z[Z > 0]
    if len(valid_z) < 1000:
        print("Not enough data - aim at the pile")
        return

    floor_z   = np.percentile(valid_z, 95)
    heights   = floor_z - Z
    pile_mask = (heights > 0.05) & (Z > 0)

    if not np.any(pile_mask):
        print("No pile detected")
        return

    pile_z      = Z[pile_mask]
    pixel_areas = (pile_z / FX) * (pile_z / FY)
    total_vol   = float(np.sum(pixel_areas * heights[pile_mask]))
    mass        = total_vol * 850

    print(f"\n=== RESULTS ===")
    print(f"Volume : {total_vol:.4f} m3")
    print(f"Mass   : {mass:.2f} kg  ({mass/1000:.3f} tonnes)")

depth_frame = None

def on_depth_ready(frame):
    global depth_frame
    raw = np.frombuffer(frame.image.bits, dtype=np.uint8)
    raw = raw.reshape(H, W, 2)
    data = raw[:,:,0].astype(np.uint16) | (raw[:,:,1].astype(np.uint16) << 8)
    depth_frame = data >> 3

print("Starting Kinect v1...")
with nui.Runtime() as kinect:
    kinect.depth_stream.open(
        nui.ImageStreamType.Depth,
        2,
        nui.ImageResolution.Resolution640x480,
        nui.ImageType.Depth
    )
    kinect.depth_frame_ready += on_depth_ready
    print("Kinect ONLINE — SPACE=Measure  Q=Quit")

    while True:
        if depth_frame is not None:
            vis = np.clip(depth_frame, 500, 1500)
            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(255 - vis, cv2.COLORMAP_JET)
            vis = cv2.medianBlur(vis, 5)
            cv2.putText(vis, "SPACE=Measure  Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)
            cv2.imshow("Sugar Mountain", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and depth_frame is not None:
            measure_volume(depth_frame)

cv2.destroyAllWindows()
print("Done.")
