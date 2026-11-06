"""Puts the six stages together.  scan_image() is used by the command line
program and by every analysis script in tools/."""
import time

import cv2
import numpy as np

from .preprocess import resize_for_work, preprocess
from .corners import detect_corners
from .rectify import rectify_all
from .recognize import read_grid, orientation_score
from .validate import verdict


def scan_image(img_bgr, harris_params=None, read_all_models=False,
               prepare=True):
    """Run the whole pipeline on one BGR image.

    Returns a dict. If no puzzle is found, result["verdict"] == "not_found".
    Corners are always in ORIGINAL image pixels.
    """
    t0 = time.time()

    # Stage 1
    work, scale = resize_for_work(img_bgr)
    gray, norm, binary = preprocess(work)

    # Stage 2
    det = detect_corners(gray, norm, binary, harris_params)
    result = {"work_img": work, "scale": scale, "detection": det}
    if det["corners"] is None:
        result["verdict"] = "not_found"
        result["runtime_s"] = time.time() - t0
        return result
    corners = det["corners"] / scale            # back to original pixels
    result["corners"] = corners

    # Stage 3 (warp the full-resolution image for the sharpest cells)
    rect = rectify_all(img_bgr, corners)

    # Orientation check (bonus B4): the diagonal corner rule only knows where
    # the corners are in the IMAGE, not which one is the puzzle's top-left.
    # If the page is turned by 90, 180 or 270 degrees, the correct order is
    # the detected order shifted by 1, 2 or 3 places. We try all four,
    # read 6 digits from each H4 rectification and keep the order whose
    # digits Tesseract reads with the highest confidence. A shift is only
    # accepted if it is clearly better (+10) than the detected order.
    scores = [orientation_score(rect["rect"]["H4"])]
    for shift in (1, 2, 3):
        h4_shift = rectify_all(img_bgr, np.roll(corners, shift, axis=0))["rect"]["H4"]
        scores.append(orientation_score(h4_shift))
    best = int(np.argmax(scores))
    if best != 0 and scores[best] < scores[0] + 10:
        best = 0
    result["orientation_shift"] = best           # 0, 1, 2 or 3 quarter turns
    result["rotated_180"] = best == 2
    if best != 0:
        corners = np.roll(corners, best, axis=0)
        rect = rectify_all(img_bgr, corners)
    result["corners"] = corners
    result["rect"] = rect

    # Stage 5 (grid from H4; optionally also from A3 and A4 for Stage 4)
    models = ["A3", "A4", "H4"] if read_all_models else ["H4"]
    result["grids"] = {}
    for m in models:
        grid, filled, crops = read_grid(rect["rect"][m], prepare=prepare)
        result["grids"][m] = {"grid": grid, "filled": filled, "crops": crops}

    # Stage 6
    grid = result["grids"]["H4"]["grid"]
    result["verdict"], result["conflicts"] = verdict(grid.tolist())
    result["runtime_s"] = time.time() - t0
    return result


def load_image(path):
    """cv2.imread that also works with non-ASCII paths (e.g. Turkish names)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
