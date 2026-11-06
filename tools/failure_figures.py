"""Make the failure-case figures for the report (Appendix B, section 7).

  python tools/failure_figures.py --out analysis/failures

Each case reproduces a real failure seen during development:
  F1  OCR: halftone grain around a '1' in a shaded box (IMG_0835)
  F2  Orientation: upside-down page read without the orientation check (IMG_0842)
  F3  Cell cleaning: L-shaped frame remnant in a corner cell, old rule
  F4  Affine model: A3 rectification of the most oblique image (IMG_0842)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sudokuscan.pipeline import load_image, scan_image
from sudokuscan.preprocess import resize_for_work, preprocess
from sudokuscan.corners import detect_corners
from sudokuscan.rectify import rectify_all
from sudokuscan import recognize as R
from sudokuscan.validate import verdict
from sudokuscan.visualize import debug_image

PERSONAL = ROOT / "data" / "personal"


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    return img


def big(gray, size=180):
    if gray.ndim == 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)


def f1(out):
    img = load_image(PERSONAL / "IMG_0835.JPG")
    res = scan_image(img)
    rect = res["rect"]["rect"]["H4"]
    norm, ink = R.binarize_rectified(rect)
    r, c = 7, 4
    crop = rect[r * 50 - 25:(r + 1) * 50 + 25, c * 50 - 25:(c + 1) * 50 + 25]
    mask = R.digit_components(R.cell_crop(ink, r, c))
    ocr_in = R.prepare_for_ocr(mask, R.cell_crop(norm, r, c))
    pred = res["grids"]["H4"]["grid"][r, c]
    panel = np.hstack([big(crop), big(R.cell_crop(ink, r, c)), big(ocr_in)])
    cv2.imwrite(str(out / "F1_halftone.png"), label(panel, f"F1: true 1, read {pred}"))


def f2(out):
    img = load_image(PERSONAL / "IMG_0842.JPG")
    work, scale = resize_for_work(img)
    det = detect_corners(*preprocess(work))
    rect = rectify_all(img, det["corners"] / scale)["rect"]["H4"]   # no orientation check
    grid, filled, _ = R.read_grid(rect)
    v, conf = verdict(grid.tolist())
    truth = np.array(json.loads((PERSONAL / "IMG_0842.json").read_text())["grid"])
    acc = np.mean(grid == truth)
    dbg = debug_image(rect, grid, filled, conf, v)
    dbg = cv2.resize(dbg, None, fx=0.5, fy=0.5)
    cv2.imwrite(str(out / "F2_upside_down.png"), label(dbg, f"F2: no orientation check, acc {acc:.2f}"))


def f3(out):
    """Old cleaning rule (without the 'too_big' test) on a corner cell."""
    src = ROOT / "analysis" / "failure_src" / "C_showthrough.jpg"
    if not src.exists():
        print("F3 skipped:", src, "missing")
        return
    res = scan_image(load_image(src))
    rect = res["rect"]["rect"]["H4"]
    norm, ink = R.binarize_rectified(rect)
    cell_ink = R.cell_crop(ink, 0, 0)
    # old rule = keep a blob unless thin line or (touches border and off centre)
    n, lab, st, ce = cv2.connectedComponentsWithStats(cell_ink, 8)
    h, w = cell_ink.shape
    old = np.zeros_like(cell_ink)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        cx, cy = ce[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        off = abs(cx - w / 2) > 0.3 * w or abs(cy - h / 2) > 0.3 * h
        thin = (bw > 0.8 * w and bh < 0.2 * h) or (bh > 0.8 * h and bw < 0.2 * w)
        if area < 4 or thin or (touches and off):
            continue
        old[lab == i] = 255
    old_in = R.prepare_for_ocr(old, R.cell_crop(norm, 0, 0))
    new_in = R.prepare_for_ocr(R.digit_components(cell_ink), R.cell_crop(norm, 0, 0))
    panel = np.hstack([big(R.cell_crop(rect, 0, 0, 0)), big(cell_ink), big(old_in), big(new_in)])
    cv2.imwrite(str(out / "F3_frame_remnant.png"),
                label(panel, f"F3: old rule read {R.ocr_digit(old_in)}, fixed rule read {R.ocr_digit(new_in)}"))


def f4(out):
    img = load_image(PERSONAL / "IMG_0842.JPG")
    res = scan_image(img)
    truth = np.array(json.loads((PERSONAL / "IMG_0842.json").read_text())["grid"])
    panels = []
    for m in ("A3", "H4"):
        rect = res["rect"]["rect"][m]
        grid, filled, _ = R.read_grid(rect)
        v, conf = verdict(grid.tolist())
        acc = np.mean(grid == truth)
        dbg = cv2.resize(debug_image(rect, grid, filled, conf, v), None, fx=0.5, fy=0.5)
        panels.append(label(dbg, f"{m}: acc {acc:.2f}"))
    cv2.imwrite(str(out / "F4_affine_oblique.png"), np.hstack(panels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/failures")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (f1, f2, f3, f4):
        f(out)
        print("done", f.__name__)


if __name__ == "__main__":
    main()
