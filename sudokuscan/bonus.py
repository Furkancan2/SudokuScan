"""Optional bonus parts.

B1 - backtracking solver that also checks the solution is unique.
B2 - AR overlay: draw the solution in the rectified 450x450 frame and map it
     back into the original photo with the inverse homography.
"""
import time

import cv2
import numpy as np

from .rectify import SIZE, CELL


def candidates(grid, r, c):
    """Digits that can go into cell (r, c)."""
    used = set(grid[r]) | {grid[i][c] for i in range(9)}
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used |= {grid[br + i][bc + j] for i in range(3) for j in range(3)}
    return [d for d in range(1, 10) if d not in used]


def solve(grid, limit=2):
    """Backtracking. Always fills the empty cell with the FEWEST candidates
    first (much faster than going left to right). Stops after `limit`
    solutions; finding 2 means the puzzle is not unique.
    Returns (list_of_solutions, seconds)."""
    t0 = time.time()
    g = [list(row) for row in grid]
    solutions = []

    def search():
        best, best_cands = None, None
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cands = candidates(g, r, c)
                    if best is None or len(cands) < len(best_cands):
                        best, best_cands = (r, c), cands
        if best is None:                     # no empty cell: solved
            solutions.append([row[:] for row in g])
            return len(solutions) >= limit
        r, c = best
        for d in best_cands:
            g[r][c] = d
            if search():
                return True
            g[r][c] = 0
        return False

    search()
    return solutions, time.time() - t0


def many_point_homography(candidates, H4, gray=None):
    """B3 - homography from up to 100 grid intersections.

    candidates : Harris candidate points in ORIGINAL image pixels
    H4         : the 4-corner homography (photo -> square)

    1. Use H4^-1 to predict where each of the 10x10 ideal intersections
       (i*CELL, j*CELL) should be in the photo.
    2. Match each prediction to the nearest Harris candidate, if it is
       closer than 30% of a cell.
    3. cv2.findHomography with RANSAC on all matched pairs: wrong matches
       (e.g. a digit stroke) are rejected as outliers.
    Returns (H, number of matches, number of RANSAC inliers) or None.
    """
    from .rectify import apply_homography
    ideal = np.array([(i * CELL, j * CELL) for j in range(10) for i in range(10)], float)
    predicted = apply_homography(np.linalg.inv(H4), ideal)

    # size of one cell in the photo, to set the matching tolerance
    cell_px = np.mean(np.linalg.norm(predicted[1:10] - predicted[0:9], axis=1))
    tol = 0.3 * cell_px

    src, dst = [], []
    for p, q in zip(predicted, ideal):
        d = np.linalg.norm(candidates - p, axis=1)
        k = np.argmin(d)
        if d[k] < tol:
            src.append(candidates[k])
            dst.append(q)
    if len(src) < 4:
        return None
    src = np.float32(src)
    if gray is not None:
        # candidates come from the downscaled image; refine them on the
        # full-resolution photo for sub-pixel accuracy
        win = max(3, int(0.08 * cell_px))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        pts = src.reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
        src = pts.reshape(-1, 2)
    H, inliers = cv2.findHomography(np.float32(src), np.float32(dst),
                                    cv2.RANSAC, ransacReprojThreshold=0.1 * CELL)
    if H is None:
        return None
    return H, len(src), int(inliers.sum())


def ar_overlay(original_bgr, H4, givens, solution):
    """Write the missing digits on the photo as if printed on the paper."""
    layer = np.zeros((SIZE, SIZE, 3), np.uint8)
    for r in range(9):
        for c in range(9):
            if givens[r][c] == 0:
                cv2.putText(layer, str(solution[r][c]), (int(c * CELL + 0.28 * CELL), int(r * CELL + 0.76 * CELL)),
                            cv2.FONT_HERSHEY_SIMPLEX, CELL / 45, (255, 80, 0), 3)
    h, w = original_bgr.shape[:2]
    # H4 maps photo -> square, so its inverse maps square -> photo
    back = cv2.warpPerspective(layer, np.linalg.inv(H4), (w, h))
    mask = back.sum(axis=2) > 0
    out = original_bgr.copy()
    out[mask] = back[mask]
    return out
