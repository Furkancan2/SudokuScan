"""My Stage 2 puzzle-region finder - Furkan Can."""
import cv2
import numpy as np
from scipy.signal import find_peaks

try:
    from .my_corners import my_pick_corners
except ImportError:          # when this file is run directly
    from my_corners import my_pick_corners


def four_sided_candidates(binary):
    """Return a list of (hull, quad) for every big four-sided shape."""

    candidates = []

    # Find all contours, including nested contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate image area
    image_area = binary.shape[0] * binary.shape[1]

    for contour in contours:

        # Calculate convex hull
        hull = cv2.convexHull(contour)

        # Skip small shapes
        hull_area = cv2.contourArea(hull)

        if hull_area < 0.02 * image_area:
            continue

        # Approximate the hull
        perimeter = cv2.arcLength(hull, True)
        epsilon = 0.03 * perimeter

        approx = cv2.approxPolyDP(hull, epsilon, True)

        # Keep only four-sided shapes
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            candidates.append((hull, quad))

    return candidates


def grid_score_lines(binary, quad):
    """Score a quadrilateral using detected grid lines."""

    S = 180

    # Order the four corners
    src = my_pick_corners(quad).astype(np.float32)

    dst = np.float32([
        [0, 0],
        [S - 1, 0],
        [S - 1, S - 1],
        [0, S - 1]
    ])

    # Transform the quadrilateral into a square
    H_matrix = cv2.getPerspectiveTransform(src, dst)
    square = cv2.warpPerspective(binary, H_matrix, (S, S))

    grid = (square > 0).astype(np.float32)

    rows = grid.mean(axis=1)
    cols = grid.mean(axis=0)

    # Minimum peak height and distance
    height = 0.50
    distance = max(1, round(0.6 * S / 9))

    def line_score(profile):

        peaks, _ = find_peaks(
            profile,
            height=height,
            distance=distance
        )

        # Check possible border lines
        if profile[0] >= height:
            peaks = np.append(peaks, 0)

        if profile[-1] >= height:
            peaks = np.append(peaks, S - 1)

        peaks = np.unique(np.sort(peaks))

        # Merge peaks that are closer than `distance` (one line counted twice)
        merged = []

        for peak in peaks:
            if len(merged) == 0 or peak - merged[-1] >= distance:
                merged.append(peak)

        peaks = np.array(merged, dtype=int)

        # Need at least two lines to calculate spacing
        if len(peaks) < 2:
            return 0.0

        # Score for the number of lines
        count_score = max(
            0.0,
            1.0 - abs(len(peaks) - 10) / 10
        )

        # Calculate spacing regularity
        gaps = np.diff(peaks)

        mean_gap = np.mean(gaps)
        if mean_gap <= 0:
            return 0.0

        cv = np.std(gaps) / mean_gap

        regularity_score = 1.0 / (1.0 + cv)

        return float(count_score * regularity_score)

    row_score = line_score(rows)
    col_score = line_score(cols)

    # Both directions must have a good grid
    return min(row_score, col_score)


def my_find_puzzle_region(binary):
    """Return the Sudoku region or None if not found."""

    candidates = four_sided_candidates(binary)

    if len(candidates) == 0:
        return None

    best_region = None
    best_score = 0.50

    for hull, quad in candidates:
        score = grid_score_lines(binary, quad)

        if score > best_score:
            best_score = score
            best_region = hull

    return best_region


def my_filter_points(points, region):
    """Keep Harris points inside or near the Sudoku region."""

    if region is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Find the image dimensions from the region and points
    all_x = np.concatenate([
        points[:, 0],
        region[:, 0, 0]
    ])

    all_y = np.concatenate([
        points[:, 1],
        region[:, 0, 1]
    ])

    width = int(np.ceil(np.max(all_x))) + 2
    height = int(np.ceil(np.max(all_y))) + 2

    # Create a black mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the Sudoku region with white
    cv2.drawContours(mask, [region], -1, 255, -1)

    # Expand the region by 1% of its approximate side length
    side = np.sqrt(cv2.contourArea(region))
    k = max(3, int(round(side * 0.01)))

    if k % 2 == 0:
        k += 1

    mask = cv2.dilate(
        mask,
        np.ones((k, k), dtype=np.uint8)
    )

    # Keep only points inside the expanded mask
    x = points[:, 0].astype(int)
    y = points[:, 1].astype(int)

    inside = (
        (x >= 0) & (x < width) &
        (y >= 0) & (y < height)
    )

    kept = points[inside]

    x = kept[:, 0].astype(int)
    y = kept[:, 1].astype(int)

    return kept[mask[y, x] > 0]


# ---- Test: draws your region on all 13 photos ----
if __name__ == "__main__":
    import glob, os, sys
    sys.path.insert(0, ".")
    from sudokuscan.pipeline import load_image
    from sudokuscan.preprocess import resize_for_work, preprocess

    os.makedirs("out/region", exist_ok=True)
    for path in sorted(glob.glob("data/personal/*.JPG")):
        name = os.path.basename(path)[:-4]
        work, _ = resize_for_work(load_image(path))
        gray, norm, binary = preprocess(work)
        for hull, quad in four_sided_candidates(binary):
            score = grid_score_lines(binary, quad)
            cv2.drawContours(work, [hull], -1, (255, 0, 0), 2)
            x, y = int(quad[0][0]), int(quad[0][1])
            cv2.putText(work, f"{score:.3f}", (x, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        region = my_find_puzzle_region(binary)
        if region is not None:
            cv2.drawContours(work, [region], -1, (0, 255, 0), 3)
        cv2.imwrite(f"out/region/{name}.png", work)
        print(name, "region:", "None" if region is None
              else f"area = {int(cv2.contourArea(region))}")
