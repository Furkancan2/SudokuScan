"""SudokuScan - from a photograph of a printed sudoku to a 9x9 digital grid.

Usage
  python sudoku_scan.py --input photo.jpg --outdir out/
  python sudoku_scan.py --batch images_dir/ --outdir results/
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from sudokuscan.pipeline import scan_image, load_image
from sudokuscan.visualize import harris_image, debug_image
from sudokuscan.bonus import solve, ar_overlay, many_point_homography
from sudokuscan.my_stage4 import my_measure_grid
from sudokuscan.rectify import SIZE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def save_png(path, img):
    """cv2.imwrite that also works with non-ASCII paths."""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))


def write_outputs(name, res, outdir):
    """Write all files of Section 6 for one image."""
    det = res.get("detection")
    if det is not None:
        corners_work = det["corners"]
        save_png(outdir / f"{name}_harris.png",
                 harris_image(res["work_img"], det["response"],
                              det["candidates"], det["kept"], corners_work))

    if res["verdict"] == "not_found":
        data = {"image": name, "verdict": "not_found",
                "runtime_s": round(res["runtime_s"], 3)}
        (outdir / f"{name}_result.json").write_text(json.dumps(data, indent=2))
        return

    rect = res["rect"]
    h4 = res["grids"]["H4"]
    grid = h4["grid"]

    for m in ("A3", "A4", "H4"):
        save_png(outdir / f"{name}_rect_{m}.png", rect["rect"][m])
    save_png(outdir / f"{name}_debug.png",
             debug_image(rect["rect"]["H4"], grid, h4["filled"],
                         res["conflicts"], res["verdict"]))

    lines = ["".join(str(d) for d in row) for row in grid]
    (outdir / f"{name}_grid.txt").write_text("\n".join(lines) + "\n")

    data = {
        "image": name,
        "corners": [[float(x), float(y)] for x, y in res["corners"]],
        "A3": rect["A3"].tolist(),
        "A4": rect["A4"].tolist(),
        "H4": rect["H4"].tolist(),
        "a3_br_mapped": [float(v) for v in rect["a3_br_mapped"]],
        "a3_corner_err_px": rect["a3_corner_err_px"],
        "a4_rms_px": rect["a4_rms_px"],
        "h4_max_err_px": rect["h4_max_err_px"],
        "grid": grid.tolist(),
        "verdict": res["verdict"],
        "conflicts": [list(c) for c in res["conflicts"]],
        "runtime_s": round(res["runtime_s"], 3),
    }
    (outdir / f"{name}_result.json").write_text(json.dumps(data, indent=2))


def write_bonus(name, img, res, outdir):
    """B3 (many-point homography), B1 (solver), B2 (AR overlay); --solve."""
    # B3
    cands = res["detection"]["candidates"] / res["scale"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mp = many_point_homography(cands, res["rect"]["H4"], gray)
    if mp is not None:
        H100, n_match, n_in = mp
        rect100 = cv2.warpPerspective(img, H100, (SIZE, SIZE), borderValue=(255, 255, 255))
        save_png(outdir / f"{name}_rect_H100.png", rect100)
        g4 = my_measure_grid(res["rect"]["rect"]["H4"])["mean_spacing_err_px"]
        g100 = my_measure_grid(rect100)["mean_spacing_err_px"]
        print(f"  B3: {n_match} matched intersections, {n_in} RANSAC inliers; "
              f"grid spacing error H4 {g4:.2f}px -> H100 {g100:.2f}px")

    givens = res["grids"]["H4"]["grid"].tolist()
    if res["verdict"] != "valid":
        print("  bonus: grid is invalid, not solving")
        return
    solutions, secs = solve(givens)
    if not solutions:
        print("  bonus: no solution (probably an OCR error)")
        return
    unique = len(solutions) == 1
    print(f"  bonus: solved in {secs * 1000:.1f} ms, unique = {unique}")
    sol = solutions[0]
    (outdir / f"{name}_solution.txt").write_text(
        "\n".join("".join(map(str, row)) for row in sol)
        + f"\n# solving time {secs:.4f} s, unique: {unique}\n")
    save_png(outdir / f"{name}_ar.png", ar_overlay(img, res["rect"]["H4"], givens, sol))


def process_one(path, outdir, bonus=False):
    """Process one image. Never crashes: any error -> verdict not_found."""
    name = Path(path).stem
    t0 = time.time()
    try:
        img = load_image(path)
        if img is None:
            raise ValueError("could not read image")
        res = scan_image(img)
        write_outputs(name, res, outdir)
        print(f"{name}: {res['verdict']} ({res['runtime_s']:.1f} s)")
        if bonus and res["verdict"] != "not_found":
            write_bonus(name, img, res, outdir)
    except Exception:
        traceback.print_exc()
        data = {"image": name, "verdict": "not_found",
                "runtime_s": round(time.time() - t0, 3)}
        (outdir / f"{name}_result.json").write_text(json.dumps(data, indent=2))
        print(f"{name}: not_found (error)")


def main():
    ap = argparse.ArgumentParser(description="SudokuScan")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="one image file")
    group.add_argument("--batch", help="folder with images")
    ap.add_argument("--outdir", required=True, help="output folder")
    ap.add_argument("--solve", action="store_true",
                    help="bonus: also solve the puzzle and draw an AR overlay")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.input:
        process_one(args.input, outdir, args.solve)
    else:
        files = sorted(p for p in Path(args.batch).iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
        for p in files:
            process_one(p, outdir, args.solve)


if __name__ == "__main__":
    main()
