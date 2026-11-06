"""Stage 5 OCR study (needed for R5).

  1. cell accuracy WITH and WITHOUT the cell preparation step
  2. Tesseract page-segmentation modes (--psm) on the prepared digits
  3. the most frequent confusions, with example crops saved as images

  python tools/ocr_study.py --images data/personal --out analysis/ocr

Uses the annotated corners, so that only the recognition stage is measured.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan.pipeline import load_image
from sudokuscan.rectify import rectify_all
from sudokuscan import recognize as R

PSM_MODES = [6, 7, 8, 10, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="analysis/ocr")
    ap.add_argument("--max-images", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acc = {"raw": [], "prepared": []}
    psm_ok = Counter()
    n_digits = 0
    confusions = Counter()
    examples = {}

    for ann_path in sorted(Path(args.images).glob("*.json"))[:args.max_images]:
        ann = json.loads(ann_path.read_text())
        img_path = next(p for p in ann_path.parent.glob(ann_path.stem + ".*")
                        if p.suffix.lower() != ".json")
        rect = rectify_all(load_image(img_path), ann["corners"])["rect"]["H4"]
        truth = np.asarray(ann["grid"])

        for mode, prepare in (("raw", False), ("prepared", True)):
            grid, filled, crops = R.read_grid(rect, prepare=prepare)
            acc[mode].append(np.mean(grid == truth))
            if not prepare:
                continue
            for r in range(9):
                for c in range(9):
                    t, p = truth[r, c], grid[r, c]
                    if t != p:
                        confusions[(int(t), int(p))] += 1
                        key = (int(t), int(p))
                        if crops[r][c] is not None:
                            examples.setdefault(key, []).append(crops[r][c])
                    if t and filled[r, c]:
                        n_digits += 1
                        for psm in PSM_MODES:
                            cfg = f"--psm {psm} --oem 1 -c tessedit_char_whitelist=123456789"
                            text = pytesseract.image_to_string(crops[r][c], config=cfg)
                            d = [ch for ch in text if ch in "123456789"]
                            psm_ok[psm] += bool(d) and int(d[0]) == t
        print(f"{ann_path.stem}: raw {acc['raw'][-1]:.3f}  prepared {acc['prepared'][-1]:.3f}")

    lines = ["# OCR study", "",
             f"images: {len(acc['raw'])}", "",
             "## Cell accuracy with / without preparation",
             f"- without preparation (raw 50x50 grey cell): {np.mean(acc['raw']) * 100:.1f}%",
             f"- with preparation: {np.mean(acc['prepared']) * 100:.1f}%", "",
             f"## Tesseract --psm on {n_digits} prepared digits (single pass, no fallback)"]
    for psm in PSM_MODES:
        lines.append(f"- psm {psm:2d}: {psm_ok[psm] / max(1, n_digits) * 100:.1f}%")
    lines += ["", "## Most frequent confusions (true -> predicted, 0 = empty/unread)"]
    for (t, p), n in confusions.most_common(10):
        lines.append(f"- {t} -> {p}: {n}")
        if (t, p) in examples:
            strip = np.hstack([cv2.resize(e, (64, 64)) for e in examples[(t, p)][:6]])
            cv2.imwrite(str(out / f"confusion_{t}_as_{p}.png"), strip)
    text = "\n".join(lines)
    (out / "ocr_study.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
