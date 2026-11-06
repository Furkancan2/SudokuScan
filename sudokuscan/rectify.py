"""Stage 3 - Rectification with affine (A3, A4) and projective (H4) models.

Target: a 450 x 450 square (50 px per cell). The corners map to
TL=(0,0), TR=(450,0), BR=(450,450), BL=(0,450).

Degrees of freedom
  affine      [a b tx; c d ty]          -> 6 unknowns -> 3 point pairs
  homography  3x3 matrix up to scale    -> 8 unknowns -> 4 point pairs
An affine map keeps parallel lines parallel, so it cannot model the
vanishing points that a perspective (oblique) view creates.
"""
import cv2
import numpy as np

SIZE = 450                       # target side in pixels (change here only)
CELL = SIZE // 9                 # 50 px per cell
TARGET = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])  # TL TR BR BL


def apply_affine(A, pts):
    """Map (N,2) points with a 2x3 affine matrix."""
    pts = np.asarray(pts, np.float64)
    return pts @ A[:, :2].T + A[:, 2]


def apply_homography(H, pts):
    """Map (N,2) points with a 3x3 homography (divide by w)."""
    pts = np.asarray(pts, np.float64)
    ones = np.ones((len(pts), 1))
    p = np.hstack([pts, ones]) @ H.T
    return p[:, :2] / p[:, 2:3]


def fit_a3(corners):
    """Affine from exactly 3 points: TL, TR, BL. BR is not used."""
    src = np.float32(corners[[0, 1, 3]])
    dst = TARGET[[0, 1, 3]]
    A3 = cv2.getAffineTransform(src, dst)
    # Where does the unused 4th corner (BR) land? Ideally at (450, 450).
    br_mapped = apply_affine(A3, corners[2:3])[0]
    err = float(np.linalg.norm(br_mapped - TARGET[2]))
    return A3, br_mapped, err


def fit_a4(corners):
    """Least-squares affine from all four corners.

    cv2.estimateAffine2D uses RANSAC; with a huge reprojection threshold
    every point is an inlier, and the final refinement step then solves the
    ordinary least-squares problem over all four points.
    """
    A4, _ = cv2.estimateAffine2D(np.float32(corners), TARGET,
                                 method=cv2.RANSAC,
                                 ransacReprojThreshold=1e6,
                                 refineIters=50)
    mapped = apply_affine(A4, corners)
    rms = float(np.sqrt(np.mean(np.sum((mapped - TARGET) ** 2, axis=1))))
    return A4, rms


def fit_h4(corners):
    """Homography from the four corners (exact fit)."""
    H4 = cv2.getPerspectiveTransform(np.float32(corners), TARGET)
    mapped = apply_homography(H4, corners)
    max_err = float(np.max(np.linalg.norm(mapped - TARGET, axis=1)))
    return H4, max_err


def rectify_all(img_bgr, corners):
    """Produce the three 450x450 rectified images and their statistics.

    corners are TL, TR, BR, BL in the coordinates of img_bgr.
    """
    corners = np.asarray(corners, np.float64)
    A3, br_a3, a3_err = fit_a3(corners)
    A4, a4_rms = fit_a4(corners)
    H4, h4_err = fit_h4(corners)

    size = (SIZE, SIZE)
    flags = cv2.INTER_LINEAR
    border = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rect_a3 = cv2.warpAffine(img_bgr, A3, size, flags=flags, **border)
    rect_a4 = cv2.warpAffine(img_bgr, A4, size, flags=flags, **border)
    rect_h4 = cv2.warpPerspective(img_bgr, H4, size, flags=flags, **border)

    return {
        "A3": A3, "A4": A4, "H4": H4,
        "a3_br_mapped": br_a3, "a3_corner_err_px": a3_err,
        "a4_rms_px": a4_rms, "h4_max_err_px": h4_err,
        "rect": {"A3": rect_a3, "A4": rect_a4, "H4": rect_h4},
    }

