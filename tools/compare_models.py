"""Stage 4 - affine (A3, A4) versus homography (H4) comparison.

For every annotated image (use >= 8, from near-frontal to strongly oblique):
  1. corner residual : A3 fourth-corner error, A4 RMS residual
  2. grid geometry   : spacing error and bending of the 20 grid lines
  3. recognition     : cell accuracy of the same OCR stage on A3, A4, H4
all against the obliqueness of the view (see my_stage4.my_obliqueness).
The two measures (obliqueness, grid lines) are in sudokuscan/my_stage4.py.

  python tools/compare_models.py --images data/personal --out analysis/compare
  (add --gt-corners to use the annotated corners instead of the detected
   ones; this separates the model error from the corner-detection error)

Writes compare.csv, three plots and side_by_side.png (most oblique image).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan.pipeline import scan_image, load_image
from sudokuscan.rectify import rectify_all, SIZE, CELL
from sudokuscan.recognize import read_grid
from sudokuscan.my_stage4 import my_obliqueness, my_measure_grid

MODELS = ["A3", "A4", "H4"]
COLORS = {"A3": "#e67e22", "A4": "#8e44ad", "H4": "#27ae60"}


def analyse(img, ann, use_gt):
    if use_gt:
        corners = np.asarray(ann["corners"], float)
    else:
        res = scan_image(img)
        if res["verdict"] == "not_found":
            return None
        corners = res["corners"]
    rect = rectify_all(img, corners)
    row = {"obliqueness": my_obliqueness(corners),
           "a3_corner_err_px": rect["a3_corner_err_px"],
           "a4_rms_px": rect["a4_rms_px"],
           "h4_max_err_px": rect["h4_max_err_px"]}
    truth = np.asarray(ann["grid"])
    for m in MODELS:
        g = my_measure_grid(rect["rect"][m])
        row[f"{m}_spacing_err_px"] = g["mean_spacing_err_px"]
        row[f"{m}_max_spacing_err_px"] = g["max_spacing_err_px"]
        row[f"{m}_bend_px"] = g["mean_bend_px"]
        grid, _, _ = read_grid(rect["rect"][m])
        row[f"{m}_cell_acc"] = float(np.mean(grid == truth))
    return row, rect


def scatter(rows, keys, labels, ylabel, title, path):
    ob = [r["obliqueness"] for r in rows]
    plt.figure(figsize=(6, 4))
    for key, label in zip(keys, labels):
        model = label.split()[0]
        plt.plot(ob, [r[key] for r in rows], "o", color=COLORS.get(model),
                 label=label, alpha=0.8)
    plt.xlabel("obliqueness = 1 - shorter/longer opposite side")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="analysis/compare")
    ap.add_argument("--gt-corners", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, rects = [], {}
    for ann_path in sorted(Path(args.images).glob("*.json")):
        ann = json.loads(ann_path.read_text())
        img_path = next(p for p in ann_path.parent.glob(ann_path.stem + ".*")
                        if p.suffix.lower() != ".json")
        result = analyse(load_image(img_path), ann, args.gt_corners)
        if result is None:
            print(ann_path.stem, "not found - skipped")
            continue
        row, rect = result
        row = {"image": ann_path.stem, **row}
        rows.append(row)
        rects[ann_path.stem] = rect
        print(f"{row['image']:<20} obl {row['obliqueness']:.3f}  A3 err {row['a3_corner_err_px']:6.1f}px  "
              f"A4 rms {row['a4_rms_px']:5.1f}px  spacing A3/A4/H4 "
              f"{row['A3_spacing_err_px']:.1f}/{row['A4_spacing_err_px']:.1f}/{row['H4_spacing_err_px']:.1f}px  "
              f"acc {row['A3_cell_acc']:.2f}/{row['A4_cell_acc']:.2f}/{row['H4_cell_acc']:.2f}")

    rows.sort(key=lambda r: r["obliqueness"])
    with open(out / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scatter(rows, ["a3_corner_err_px", "a4_rms_px"],
            ["A3 4th-corner error", "A4 RMS residual"], f"pixels ({SIZE} px target)",
            "Corner residual vs obliqueness", out / "residual_vs_obliqueness.png")
    scatter(rows, [f"{m}_spacing_err_px" for m in MODELS],
            [f"{m} grid-line spacing error" for m in MODELS], "mean |position - ideal| [px]",
            "Grid geometry vs obliqueness", out / "grid_vs_obliqueness.png")
    scatter(rows, [f"{m}_cell_acc" for m in MODELS],
            [f"{m} cell accuracy" for m in MODELS], "cell accuracy",
            "Recognition vs obliqueness", out / "accuracy_vs_obliqueness.png")

    # side-by-side figure for the most oblique image
    worst = rows[-1]["image"]
    panels = []
    for m in MODELS:
        p = rects[worst]["rect"][m].copy()
        for i in range(10):                          # ideal grid in green
            cv2.line(p, (i * CELL, 0), (i * CELL, SIZE), (0, 200, 0), 1)
            cv2.line(p, (0, i * CELL), (SIZE, i * CELL), (0, 200, 0), 1)
        cv2.putText(p, m, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        panels.append(cv2.copyMakeBorder(p, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    cv2.imwrite(str(out / "side_by_side.png"), np.hstack(panels))

    print("\nMeans over all images:")
    for m in MODELS:
        print(f"  {m}: spacing err {np.mean([r[f'{m}_spacing_err_px'] for r in rows]):5.2f}px  "
              f"cell acc {np.mean([r[f'{m}_cell_acc'] for r in rows]) * 100:5.1f}%")
    print("saved to", out)


if __name__ == "__main__":
    main()
