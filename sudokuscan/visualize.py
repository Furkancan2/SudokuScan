"""Debug images required by Section 6."""
import cv2
import numpy as np

from .recognize import CELL

NAMES = ["TL", "TR", "BR", "BL"]


def harris_image(work_img, response, candidates, kept, corners):
    """Harris response map (log scale, colour) with all candidates (grey),
    candidates inside the puzzle region (yellow) and the 4 corners (red)."""
    r = np.maximum(response, 0)
    r = np.log1p(r / (r.max() + 1e-12) * 1000)
    r = (255 * r / (r.max() + 1e-12)).astype(np.uint8)
    heat = cv2.applyColorMap(r, cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(heat, 0.75, work_img, 0.25, 0)

    s = max(2, work_img.shape[1] // 300)
    for x, y in candidates:
        cv2.circle(out, (int(x), int(y)), s, (160, 160, 160), 1)
    for x, y in kept:
        cv2.circle(out, (int(x), int(y)), s, (0, 255, 255), 1)
    if corners is not None:
        cv2.polylines(out, [corners.astype(np.int32).reshape(-1, 1, 2)], True,
                      (0, 255, 0), max(1, s // 2))
        for name, (x, y) in zip(NAMES, corners):
            cv2.circle(out, (int(x), int(y)), 4 * s, (0, 0, 255), 2)
            cv2.putText(out, name, (int(x) + 5 * s, int(y) - 5 * s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25 * s, (0, 0, 255), 2)
    cv2.putText(out, f"candidates: {len(candidates)}  in region: {len(kept)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def debug_image(rect_h4, grid, filled, conflicts, verdict):
    """Rectified H4 image, upscaled 2x, with cell boundaries, the recognised
    digit in each filled cell (green) and conflicting cells (red)."""
    scale = 2
    img = cv2.resize(rect_h4, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    c = CELL * scale
    for i in range(10):
        cv2.line(img, (i * c, 0), (i * c, 9 * c), (255, 120, 0), 1)
        cv2.line(img, (0, i * c), (9 * c, i * c), (255, 120, 0), 1)
    bad = set(map(tuple, conflicts))
    for r in range(9):
        for col in range(9):
            x, y = col * c, r * c
            if (r, col) in bad:
                cv2.rectangle(img, (x + 2, y + 2), (x + c - 2, y + c - 2),
                              (0, 0, 255), 3)
            if filled[r, col]:
                label = str(grid[r, col]) if grid[r, col] else "?"
                cv2.putText(img, label, (x + c - 24, y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)
    bar = np.full((40, img.shape[1], 3), 255, np.uint8)
    colour = (0, 150, 0) if verdict == "valid" else (0, 0, 255)
    cv2.putText(bar, f"verdict: {verdict}  conflicts: {len(bad)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return np.vstack([img, bar])
