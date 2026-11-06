"""My Stage 4 measures - Furkan Can."""

import numpy as np
from scipy.signal import find_peaks
from .recognize import binarize_rectified

IDEAL = np.arange(10) * 50


def my_obliqueness(corners):
    """0 for a frontal view, larger for a stronger keystone."""

    corners = np.asarray(corners, dtype=np.float32)

    tl, tr, br, bl = corners

    # Calculate the lengths of the four sides
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    # Compare opposite sides
    horizontal = abs(top - bottom) / max(top, bottom)
    vertical = abs(left - right) / max(left, right)

    # Use the larger difference
    return float(max(horizontal, vertical))



def my_measure_grid(rect_bgr):
    """Measure grid position errors and line bending."""

    _, ink = binarize_rectified(rect_bgr)
    ink = (ink > 0).astype(np.float32)

    height, width = ink.shape

    def detect_lines(profile):
        # A grid line should cover a noticeable part
        # of the image in its direction.
        threshold = max(0.20, float(np.mean(profile) + np.std(profile)))

        peaks, _ = find_peaks(
            profile,
            height=threshold,
            distance=20,
            prominence=0.05
        )

        # find_peaks does not detect the first or last pixel
        # Border lines can be partially cut off after rectification
        border_threshold = max(0.15, float(np.median(profile) + 0.05))

        # Check a small area near both image borders
        border_width = 5

        if np.max(profile[:border_width]) >= border_threshold:
            border_peak = np.argmax(profile[:border_width])
            peaks = np.append(peaks, border_peak)

        if np.max(profile[-border_width:]) >= border_threshold:
            border_peak = len(profile) - border_width + np.argmax(
                profile[-border_width:]
            )
            peaks = np.append(peaks, border_peak)

        peaks = np.unique(np.sort(peaks))

        # Merge peaks that belong to the same thick line
        merged = []
        for peak in peaks:
            if not merged or peak - merged[-1] >= 20:
                merged.append(int(peak))

        return np.array(merged, dtype=int)

    def match_lines(profile):
        peaks = detect_lines(profile)

        matched = [None] * len(IDEAL)
        available = list(peaks)

        max_distance = 25

        for i, ideal in enumerate(IDEAL):
            if not available:
                break

            nearest = min(available, key=lambda p: abs(p - ideal))
            error = abs(nearest - ideal)

            if error <= max_distance:
                matched[i] = nearest
                available.remove(nearest)

        return matched

    # Full-image projection profiles
    rows = ink.mean(axis=1)
    cols = ink.mean(axis=0)

    row_lines = match_lines(rows)
    col_lines = match_lines(cols)

    errors = []

    for ideal, found in zip(IDEAL, row_lines):
        if found is None:
            errors.append(50.0)
        else:
            errors.append(abs(found - ideal))

    for ideal, found in zip(IDEAL, col_lines):
        if found is None:
            errors.append(50.0)
        else:
            errors.append(abs(found - ideal))

    # Measure straightness in three sections
    bends = []

    def measure_bending(full_lines, horizontal=True):
        if horizontal:
            sections = np.array_split(ink, 3, axis=1)
            profiles = [part.mean(axis=1) for part in sections]
        else:
            sections = np.array_split(ink, 3, axis=0)
            profiles = [part.mean(axis=0) for part in sections]

        section_peaks = [detect_lines(p) for p in profiles]

        for found in full_lines:
            if found is None:
                continue

            positions = []

            for peaks in section_peaks:
                if len(peaks) == 0:
                    continue

                nearest = peaks[np.argmin(np.abs(peaks - found))]

                # Avoid matching an unrelated distant line
                if abs(nearest - found) <= 25:
                    positions.append(float(nearest))

            if len(positions) >= 2:
                bends.append(max(positions) - min(positions))

    measure_bending(row_lines, horizontal=True)
    measure_bending(col_lines, horizontal=False)

    return {
        "mean_spacing_err_px": float(np.mean(errors)),
        "max_spacing_err_px": float(np.max(errors)),
        "mean_bend_px": float(np.mean(bends)) if bends else 0.0
    }


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, ".")
    from sudokuscan.my_stage4 import my_obliqueness, my_measure_grid
    from sudokuscan.pipeline import load_image
    from sudokuscan.rectify import rectify_all

    img = load_image("data/personal/IMG_0841.JPG")
    corners = json.load(open("data/personal/IMG_0841.json"))["corners"]
    print("obliqueness:", round(my_obliqueness(corners), 3))
    rect = rectify_all(img, corners)
    for m in ("A3", "A4", "H4"):
        print(m, my_measure_grid(rect["rect"][m]))
