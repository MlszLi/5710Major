import cv2
import numpy as np
import os, json, time
from glob import glob
from typing import Dict, Any, Tuple, List, Optional

# === Adjustable parameters ===
PATTERN_SIZE = (9, 6)     # Number of inner corners (cols, rows)
SQUARE_SIZE = 1.0         # Physical side length of a square (used only for scaling)
SAVE_DIR = "calib_images"
MIN_OK_IMAGES = 2
os.makedirs(SAVE_DIR, exist_ok=True)


def _detect_corners(gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect chessboard corners."""
    try:
        # More stable new algorithm (OpenCV >= 4.5)
        ret, corners = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret:
            return False, None
        return True, corners.astype(np.float32)
    except AttributeError:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
        if not ret:
            return False, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        return True, corners


def capture_once(camera_index: int = 0) -> Dict[str, Any]:
    """Capture one image, detect chessboard, and save it."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"ok": False, "msg": f"Failed to open camera {camera_index}"}

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"ok": False, "msg": "Failed to read frame"}

    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(SAVE_DIR, f"img_{ts}.jpg")
    cv2.imwrite(img_path, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, corners = _detect_corners(gray)
    vis_path = None
    if ok:
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, ok)
        vis_path = img_path.replace(".jpg", "_corners.jpg")
        cv2.imwrite(vis_path, vis)

    return {
        "ok": True,
        "msg": "Captured successfully",
        "image_path": img_path,
        "chessboard_detected": bool(ok),
        "corners_preview_path": vis_path,
    }


def calibrate_all() -> Dict[str, Any]:
    """Read all captured images and perform camera calibration."""
    image_paths = sorted(
        [p for p in glob(os.path.join(SAVE_DIR, "*.jpg")) if "_corners" not in p] +
        [p for p in glob(os.path.join(SAVE_DIR, "*.png")) if "_corners" not in p]
    )

    if not image_paths:
        return {"ok": False, "msg": "No images found"}

    objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints, reports = [], [], []
    im_size = None

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            reports.append({"image": p, "ok": False, "msg": "Failed to read"})
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if im_size is None:
            im_size = (gray.shape[1], gray.shape[0])
        ok, corners = _detect_corners(gray)
        if ok:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            vis = img.copy()
            cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, ok)
            cv2.imwrite(p.replace(".jpg", "_corners.jpg"), vis)
            reports.append({"image": p, "ok": True, "msg": "Chessboard detected"})
        else:
            reports.append({"image": p, "ok": False, "msg": "Chessboard not found"})

    ok_count = sum(1 for r in reports if r["ok"])
    if ok_count < MIN_OK_IMAGES or im_size is None:
        return {"ok": False, "msg": f"Too few valid images ({ok_count})", "per_image_report": reports}

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, im_size, None, None)

    mean_err = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        mean_err += err
    mean_err /= len(objpoints)

    result = {
        "ok": True,
        "image_size": {"width": im_size[0], "height": im_size[1]},
        "pattern_size": {"cols": PATTERN_SIZE[0], "rows": PATTERN_SIZE[1]},
        "rms": float(rms),
        "mean_reprojection_error": float(mean_err),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "num_images_total": len(image_paths),
        "num_images_ok": int(ok_count),
        "per_image_report": reports,
    }

    with open("calibration_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    # Example: manual test
    print("Capture one frame:")
    for _ in range(10):
        print(capture_once())
    print("Run calibration:")
    print(calibrate_all())
