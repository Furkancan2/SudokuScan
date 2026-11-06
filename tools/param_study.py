"""Stage 2 parameter study (needed for R1).

Changes ONE Harris parameter at a time around the default values and
records, on every annotated image:
  - number of Harris candidates (after threshold + NMS)
  - corner error in % of the side length

  python tools/param_study.py --images data/personal --out analysis/params

Writes params.csv and params.png (one panel per parameter).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan import corners as C
from sudokuscan.pipeline import load_image
from sudokuscan.preprocess import resize_for_work, preprocess
from evaluate import corner_error_pct

# Values to try for each parameter. block_frac 0.002..0.012 is 2..12 px on
# a 1000 px wide work image.
SWEEPS = {
    "block_frac": [0.002, 0.003, 0.004, 0.006, 0.008, 0.012],
    "ksize": [3, 5, 7],
    "k": [0.02, 0.04, 0.06, 0.10, 0.15],
    "rel_thresh": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
}
DEFAULTS = {"block_frac": C.HARRIS_BLOCK_FRAC, "ksize": C.HARRIS_KSIZE,
            "k": C.HARRIS_K, "rel_thresh": C.HARRIS_REL_THRESH}
FAIL_ERR = 50.0   # corner error counted for an image where nothing is found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="analysis/params")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Preprocess every image once (Stage 1 does not depend on these params).
    data = []
    for ann_path in sorted(Path(args.images).glob("*.json")):
        ann = json.loads(ann_path.read_text())
        img_path = next(p for p in ann_path.parent.glob(ann_path.stem + ".*")
                        if p.suffix.lower() != ".json")
        work, scale = resize_for_work(load_image(img_path))
        data.append((ann_path.stem, ann, scale, preprocess(work)))
    print(f"{len(data)} images")

    rows = []
    for param, values in SWEEPS.items():
        for v in values:
            params = dict(DEFAULTS, **{param: v})
            for name, ann, scale, (gray, norm, binary) in data:
                det = C.detect_corners(gray, norm, binary, params)
                if det["corners"] is None:
                    err = FAIL_ERR
                else:
                    # Harris + the diagonal rule find WHERE the corners are;
                    # which one is the puzzle's TL is decided later by the
                    # OCR orientation check. So the error is taken over the
                    # best cyclic order (0/90/180/270 degrees).
                    pred = det["corners"] / scale
                    err = min(FAIL_ERR, min(corner_error_pct(np.roll(pred, s, axis=0), ann["corners"])
                                            for s in range(4)))
                rows.append({"param": param, "value": v, "image": name,
                             "candidates": len(det["candidates"]),
                             "in_region": len(det["kept"]),
                             "corner_err_pct": round(err, 3)})
            sel = [r for r in rows if r["param"] == param and r["value"] == v]
            print(f"{param:>10} = {v:<6} candidates {np.mean([r['candidates'] for r in sel]):7.1f}  "
                  f"corner err mean {np.mean([r['corner_err_pct'] for r in sel]):6.2f}%  "
                  f"median {np.median([r['corner_err_pct'] for r in sel]):5.2f}%")

    with open(out / "params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (param, values) in zip(axes, SWEEPS.items()):
        cand = [np.mean([r["candidates"] for r in rows if r["param"] == param and r["value"] == v]) for v in values]
        err = [np.median([r["corner_err_pct"] for r in rows if r["param"] == param and r["value"] == v]) for v in values]
        x = np.arange(len(values))
        ax.bar(x, cand, color="#9db4d6", label="candidates (mean)")
        ax.set_xticks(x, [str(v) for v in values])
        ax.set_xlabel(param + ("  (default %s)" % DEFAULTS[param]))
        ax.set_ylabel("candidates")
        ax2 = ax.twinx()
        ax2.plot(x, err, "o-", color="#c0392b", label="corner error (median %)")
        ax2.set_ylabel("corner error [% of side]")
    fig.suptitle("Harris parameter study (one parameter changed at a time)")
    fig.tight_layout()
    fig.savefig(out / "params.png", dpi=150)
    print("saved", out / "params.csv", out / "params.png")


if __name__ == "__main__":
    main()
