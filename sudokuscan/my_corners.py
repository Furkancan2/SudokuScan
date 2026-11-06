"""My Stage 2 corner selection - Furkan Can."""

import numpy as np
import cv2


def top_left(points):
    """Return the top-left point from an (N, 2) array of [x, y] points."""

    scores = points[:, 0] + points[:, 1]
    index = np.argmin(scores)

    return points[index]


def my_pick_corners(points):
    """Return the 4 corners in the order TL, TR, BR, BL."""

    tl = points[np.argmin(points[:, 0] + points[:, 1])]
    tr = points[np.argmax(points[:, 0] - points[:, 1])]
    br = points[np.argmax(points[:, 0] + points[:, 1])]
    bl = points[np.argmax(points[:, 1] - points[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def my_is_good_quad(quad, region):
    """Check whether the selected quadrilateral is valid."""

    quad = quad.reshape(-1, 1, 2).astype(np.float32)

    # Check if the quadrilateral is convex
    if not cv2.isContourConvex(quad):
        return False

    # Calculate the areas
    quad_area = cv2.contourArea(quad)
    region_area = cv2.contourArea(region)

    # Avoid division by zero
    if region_area <= 0:
        return False

    # Check whether the quadrilateral covers at least 80% of the region
    area_ratio = quad_area / region_area

    if area_ratio < 0.80:
        return False

    return True


# ---- Tests ----
if __name__ == "__main__":

    points = np.array([
        [120, 95],
        [480, 102],
        [115, 470],
        [300, 300],
        [490, 465]
    ])

    print("top_left:", top_left(points), " expected: [120  95]")
    print("corners:\n", my_pick_corners(points))

    region = np.array([
        [[100, 100]],
        [[500, 100]],
        [[500, 500]],
        [[100, 500]]
    ], dtype=np.int32)

    good = np.float32([
        [105, 102],
        [495, 98],
        [498, 497],
        [102, 495]
    ])

    bowtie = np.float32([
        [105, 102],
        [498, 497],
        [495, 98],
        [102, 495]
    ])

    tiny = np.float32([
        [300, 300],
        [320, 300],
        [320, 320],
        [300, 320]
    ])

    print("good:  ", my_is_good_quad(good, region),
          " expected: True")

    print("bowtie:", my_is_good_quad(bowtie, region),
          " expected: False")

    print("tiny:  ", my_is_good_quad(tiny, region),
          " expected: False")
