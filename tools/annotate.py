"""Annotate your own photos in the Appendix A format.

One photo:
  python tools/annotate.py data/personal/IMG_1234.jpg --conditions C2 C4
All photos listed in data/personal/photo_labels.txt, one after the other:
  python tools/annotate.py --batch data/personal

1. A window opens. Click the four OUTER corners of the puzzle grid in the
   order TL, TR, BR, BL of the PUZZLE (if the page is upside down in the
   photo, the puzzle's top-left corner is at the bottom-right of the image).
   Keys: u = undo last click, Enter = accept, Esc = skip this photo.
2. The 81 cell values:
   - if the photo's puzzle (A, B or C) is known, they are taken from
     tools/puzzles_truth.json (the printed puzzles; independent of the
     scanner);
   - otherwise the scanner's guess is shown row by row and you confirm or
     correct each row.
3. <name>.json is written next to the image.

photo_labels.txt maps file name -> {"puzzle": "A", "conditions": ["C2"]}.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan.pipeline import load_image
from sudokuscan.rectify import rectify_all
from sudokuscan.recognize import read_grid

NAMES = ["TL", "TR", "BR", "BL"]
MAX_SIDE = 1000
TRUTH = Path(__file__).resolve().parent / "puzzles_truth.json"


def click_corners(img, title):
    """Let the user click 4 corners. Returns (4,2) array or None if skipped."""
    scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        shown = view.copy()
        for i, (x, y) in enumerate(points):
            p = (int(x * scale), int(y * scale))
            cv2.circle(shown, p, 6, (0, 0, 255), 2)
            cv2.putText(shown, NAMES[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(points) == 4:
            pts = (np.array(points) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(shown, [pts], True, (0, 200, 0), 2)
            msg = "Enter = accept, u = undo"
        else:
            msg = f"click {NAMES[len(points)]} of the PUZZLE"
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 40), (255, 255, 255), -1)
        cv2.putText(shown, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(title, shown)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key == ord("u") and points:
            points.pop()
        if key in (13, 10) and len(points) == 4:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return np.array(points)


def ask_grid(guess):
    print("\nScanner guess (0 = empty). Enter = accept the row, or type 9 digits:")
    grid = []
    for r in range(9):
        row = "".join(str(d) for d in guess[r])
        while True:
            text = input(f"row {r + 1}: {row}  > ").strip()
            if text == "":
                grid.append([int(c) for c in row])
                break
            if len(text) == 9 and text.isdigit():
                grid.append([int(c) for c in text])
                break
            print("   please type exactly 9 digits")
    return grid


def annotate(path, conditions, puzzle=None):
    path = Path(path)
    img = load_image(path)
    print(f"\n=== {path.name}  (puzzle {puzzle or '?'}, {' '.join(conditions)}) ===")
    corners = click_corners(img, path.name)
    if corners is None:
        print("skipped")
        return

    if puzzle:
        grid = json.loads(TRUTH.read_text())[puzzle]
        # sanity check: does the rectified image show this puzzle?
        rect = rectify_all(img, corners)["rect"]["H4"]
        guess, _, _ = read_grid(rect)
        agree = np.mean(np.asarray(guess) == np.asarray(grid))
        print(f"cell agreement between your corners + OCR and puzzle {puzzle}: {agree:.0%}")
        if agree < 0.8:
            print("  WARNING: low agreement - check the corner order "
                  "(TL, TR, BR, BL of the puzzle) and the puzzle letter.")
    else:
        rect = rectify_all(img, corners)["rect"]["H4"]
        guess, _, _ = read_grid(rect)
        grid = ask_grid(guess)

    ann = {"image": path.name, "conditions": conditions,
           "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
           "grid": grid}
    out = path.with_suffix(".json")
    out.write_text(json.dumps(ann))
    print("saved", out.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--conditions", nargs="*", default=[], help="C1..C5")
    ap.add_argument("--puzzle", choices=["A", "B", "C"])
    ap.add_argument("--batch", help="folder with photo_labels.txt")
    ap.add_argument("--redo", action="store_true",
                    help="with --batch: also photos that already have a .json")
    args = ap.parse_args()

    if args.batch:
        folder = Path(args.batch)
        labels = json.loads((folder / "photo_labels.txt").read_text())
        for name, lab in labels.items():
            if (folder / name).with_suffix(".json").exists() and not args.redo:
                print(f"{name}: already annotated (use --redo to do it again)")
                continue
            annotate(folder / name, lab["conditions"], lab.get("puzzle"))
    elif args.image:
        annotate(args.image, args.conditions, args.puzzle)
    else:
        ap.error("give an image or --batch folder")


if __name__ == "__main__":
    main()
