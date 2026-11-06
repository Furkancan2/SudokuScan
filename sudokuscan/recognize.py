"""Stage 5 - Cell extraction, empty-cell detection and digit recognition.

Steps for every one of the 81 cells:
  1. cut the 50x50 cell from the rectified image and drop an inner margin
     so the grid lines do not reach the OCR engine;
  2. decide EMPTY / FILLED with an explicit ink rule (before any OCR);
  3. for filled cells, prepare the digit (remove border remnants, crop,
     stretch contrast, scale, centre, pad with white) and run Tesseract in
     single-character mode with a 1-9 whitelist.

Why MARGIN = 5 px (10% of a 50 px cell): the grid lines are 2-6 px thick
in the rectified image and the rectification can be off by 1-2 px, so 5 px
removes the lines, while printed digits sit in the middle ~60% of a cell and
are not cut. Line pieces that still get through are removed in
digit_components().
"""
import os
import shutil

import cv2
import numpy as np
import pytesseract

# Find the Tesseract program: TESSERACT_CMD environment variable, the PATH,
# or the default Windows install folder.
_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif shutil.which("tesseract") is None and os.path.exists(_WIN_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _WIN_DEFAULT

from .rectify import CELL           # 50 px for the 450 px target
MARGIN = round(0.10 * CELL)         # pixels removed on every side (10%)

# Empty-cell rule: the biggest ink blob that is not a border remnant must be
# at least this big, otherwise the cell is empty.
MIN_INK_AREA_FRAC = 0.025   # of the cropped cell area
MIN_INK_HEIGHT_FRAC = 0.30  # of the cropped cell height

TESS_CONFIG = "--psm 10 --oem 1 -c tessedit_char_whitelist=123456789"


def binarize_rectified(rect_bgr):
    """Returns (norm, ink) for the whole 450x450 image.
    norm = greyscale with shading removed (paper ~255),
    ink  = binary mask, 255 where there is ink.

    Divide by a closed background (removes leftover shading), then a local
    (adaptive) threshold: a pixel is ink if it is clearly darker than its
    31x31 neighbourhood. A single global Otsu threshold lost faint, blurry
    digits on strongly oblique photos because the dark grid lines pull the
    threshold down.
    """
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)          # against sensor noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(smooth, bg, scale=255)
    local = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 12)
    # ...and it must also be clearly darker than white paper (255), so that
    # noise in dark, low-light photos is not taken as ink.
    dark = (norm < 210).astype(np.uint8) * 255
    ink = cv2.bitwise_and(local, dark)
    # remove isolated noise pixels (opening = erosion followed by dilation)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return norm, ink


def cell_crop(img, row, col, margin=MARGIN):
    """Inner part of cell (row, col)."""
    y0, x0 = row * CELL + margin, col * CELL + margin
    return img[y0:y0 + CELL - 2 * margin, x0:x0 + CELL - 2 * margin]


def digit_components(ink_cell):
    """Keep the ink blobs that look like part of a digit.

    A blob is treated as a grid-line remnant and removed if
      - it is a long thin line along the border, or
      - it touches the crop border AND its centre is far from the cell
        centre, or
      - it touches the crop border AND spans more than 80% of the cell
        width or height. A printed digit is never that big; an L-shaped
        piece of a thick frame in a corner cell is (its centre can still be
        near the middle, so the rule above misses it - seen on a real photo).
    """
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ink_cell, 8)
    h, w = ink_cell.shape
    keep = np.zeros_like(ink_cell)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx, cy = cents[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        off_centre = abs(cx - w / 2) > 0.3 * w or abs(cy - h / 2) > 0.3 * h
        thin_line = (bw > 0.8 * w and bh < 0.2 * h) or (bh > 0.8 * h and bw < 0.2 * w)
        too_big = bw > 0.8 * w or bh > 0.8 * h
        if area < 4 or thin_line or (touches and (off_centre or too_big)):
            continue
        keep[labels == i] = 255
    return keep


def is_filled(digit_mask):
    """Explicit empty/filled rule on the cleaned ink mask."""
    h, w = digit_mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(digit_mask, 8)
    if n <= 1:
        return False
    biggest = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    area_ok = biggest[cv2.CC_STAT_AREA] >= MIN_INK_AREA_FRAC * h * w
    height_ok = biggest[cv2.CC_STAT_HEIGHT] >= MIN_INK_HEIGHT_FRAC * h
    return bool(area_ok and height_ok)


DIGIT_HEIGHT = 24     # px; chosen with tools/ocr_study.py (24 beat 48)



def prepare_for_ocr(digit_mask, norm_cell, digit_height=DIGIT_HEIGHT):
    """Prepare one digit for Tesseract.

    digit_mask : cleaned binary mask of the digit (from digit_components)
    norm_cell  : the same cell in grey levels, shading removed

    The binary mask is only used to find WHERE the digit is and to erase
    everything else (border remnants, dirt). The grey levels are kept,
    because on blurry or strongly oblique photos thin strokes are faint and
    a hard threshold breaks them (an 8 turns into '||').
    Then: crop, stretch the contrast, scale to a fixed height, centre on a
    white square.
    """
    # 1. keep only the pixels close to the digit
    near = cv2.dilate(digit_mask, np.ones((3, 3), np.uint8), iterations=2)
    img = np.where(near > 0, norm_cell, 255).astype(np.uint8)

    # 2. crop to the digit's bounding box
    ys, xs = np.nonzero(digit_mask)
    y0, y1 = max(0, ys.min() - 1), min(img.shape[0], ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(img.shape[1], xs.max() + 2)
    digit = img[y0:y1, x0:x1].astype(np.float32)

    # 3. contrast stretch: darkest pixel -> 0, paper stays 255
    lo = digit.min()
    digit = np.clip((digit - lo) / max(1.0, 255.0 - lo) * 255.0, 0, 255)

    # 4. fixed height, centred on a white canvas
    h, w = digit.shape
    size = 2 * digit_height                       # canvas = 2 x digit height
    scale = min(digit_height / h, 0.75 * size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    digit = cv2.resize(digit, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size), 255, np.float32)
    cy, cx = (size - new_h) // 2, (size - new_w) // 2
    canvas[cy:cy + new_h, cx:cx + new_w] = digit
    return np.clip(canvas, 0, 255).astype(np.uint8)       # black on white


TESS_FALLBACK = "--psm 13 --oem 1 -c tessedit_char_whitelist=123456789"


def ocr_digit(img):
    """Run Tesseract on one prepared image; returns 1..9 or 0 if unreadable.
    psm 10 = 'single character'; if it returns nothing we try psm 13
    ('raw line'), which sometimes reads digits that psm 10 skips."""
    for config in (TESS_CONFIG, TESS_FALLBACK):
        text = pytesseract.image_to_string(img, config=config)
        for ch in text:
            if ch in "123456789":
                return int(ch)
    return 0


def ocr_confidence(img):
    """Tesseract's confidence (0-100) for the single character in img."""
    data = pytesseract.image_to_data(img, config=TESS_CONFIG,
                                     output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"])
             if t.strip() and float(c) >= 0]
    return max(confs) if confs else 0.0


def orientation_score(rect_bgr, n_cells=6):
    """Mean OCR confidence over the first n filled cells.

    Upright digits are read with high confidence; upside-down digits look
    like unknown shapes and get low confidence. Comparing this score for the
    image and the image turned by 180 degrees tells which way is up.
    """
    norm, ink = binarize_rectified(rect_bgr)
    scores = []
    for r in range(9):
        for c in range(9):
            mask = digit_components(cell_crop(ink, r, c))
            if is_filled(mask):
                scores.append(ocr_confidence(prepare_for_ocr(mask, cell_crop(norm, r, c))))
                if len(scores) == n_cells:
                    return float(np.mean(scores))
    return float(np.mean(scores)) if scores else 0.0


def read_grid(rect_bgr, prepare=True):
    """Read the 9x9 grid from a rectified image.

    prepare=False runs OCR on the raw grey cell (no margin, no cleaning)
    to show the effect of the preparation step.
    Returns (grid, filled, crops) where crops[r][c] is what OCR saw.
    """
    norm, ink = binarize_rectified(rect_bgr)
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    grid = np.zeros((9, 9), int)
    filled = np.zeros((9, 9), bool)
    crops = [[None] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            mask = digit_components(cell_crop(ink, r, c))
            filled[r, c] = is_filled(mask)
            if not filled[r, c]:
                continue
            if prepare:
                ocr_img = prepare_for_ocr(mask, cell_crop(norm, r, c))
                grid[r, c] = ocr_digit(ocr_img)
                if grid[r, c] == 0:          # unreadable: retry at 32 px
                    ocr_img = prepare_for_ocr(mask, cell_crop(norm, r, c), 32)
                    grid[r, c] = ocr_digit(ocr_img)
            else:
                ocr_img = cell_crop(gray, r, c, margin=0)   # raw 50x50 cell
                grid[r, c] = ocr_digit(ocr_img)
            crops[r][c] = ocr_img
    return grid, filled, crops
