"""Compute the Section 7 metrics against ground-truth annotations.

  python tools/evaluate.py --images data/personal --results results/personal

The images folder must contain <name>.jpg + <name>.json (Appendix A format).
The results folder is what sudoku_scan.py --batch wrote. With --run the
script runs the scanner first.

Prints the metrics overall and per condition (C1-C5) and writes
metrics.csv (one row per image) into the results folder.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def corner_error_pct(pred, true):
    """Mean distance between predicted and true corners, as % of the mean
    side length of the true quadrilateral."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    dist = np.linalg.norm(pred - true, axis=1).mean()
    sides = [np.linalg.norm(true[i] - true[(i + 1) % 4]) for i in range(4)]
    return 100.0 * dist / np.mean(sides)


def cell_scores(pred_grid, true_grid):
    """Returns (cell accuracy, TP, FP, FN) where 'positive' = filled."""
    p, t = np.asarray(pred_grid), np.asarray(true_grid)
    acc = float(np.mean(p == t))
    pf, tf = p > 0, t > 0
    tp = int(np.sum(pf & tf))
    fp = int(np.sum(pf & ~tf))
    fn = int(np.sum(~pf & tf))
    return acc, tp, fp, fn


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def summarise(rows, label):
    n = len(rows)
    if n == 0:
        return
    found = [r for r in rows if r["found"]]
    ok = [r for r in rows if r["found"] and r["runtime_s"] <= 20]
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    ce = [r["corner_err_pct"] for r in found]
    print(f"{label:>8} n={n:3d} | corner err "
          f"{np.mean(ce) if ce else float('nan'):5.2f}% "
          f"(median {np.median(ce) if ce else float('nan'):5.2f}%) | "
          f"cell acc {np.mean([r['cell_acc'] for r in rows]) * 100:5.1f}% | "
          f"empty F1 {f1(tp, fp, fn):.3f} | "
          f"exact {np.mean([r['cell_acc'] == 1.0 for r in rows]) * 100:5.1f}% | "
          f"robust {len(ok) / n * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--run", action="store_true", help="run sudoku_scan.py first")
    args = ap.parse_args()
    img_dir, res_dir = Path(args.images), Path(args.results)

    if args.run:
        root = Path(__file__).resolve().parent.parent
        subprocess.run([sys.executable, str(root / "sudoku_scan.py"),
                        "--batch", str(img_dir), "--outdir", str(res_dir)],
                       check=True)

    rows = []
    for ann_path in sorted(img_dir.glob("*.json")):
        ann = json.loads(ann_path.read_text())
        name = ann_path.stem
        res_path = res_dir / f"{name}_result.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {"verdict": "not_found", "runtime_s": 99}
        found = res["verdict"] != "not_found"
        pred_grid = res.get("grid", [[0] * 9] * 9)
        acc, tp, fp, fn = cell_scores(pred_grid, ann["grid"])
        rows.append({
            "image": name,
            "conditions": " ".join(ann.get("conditions", [])),
            "found": found,
            "corner_err_pct": corner_error_pct(res["corners"], ann["corners"]) if found else float("nan"),
            "cell_acc": acc, "tp": tp, "fp": fp, "fn": fn,
            "runtime_s": res.get("runtime_s", 99),
        })

    if not rows:
        sys.exit(f"No annotation (.json) files found in {img_dir} - nothing to evaluate.")

    print("Per image:")
    for r in rows:
        print(f"  {r['image']:<20} [{r['conditions']:<9}] corner {r['corner_err_pct']:6.2f}%  "
              f"cells {r['cell_acc'] * 100:5.1f}%  FP {r['fp']} FN {r['fn']}  {r['runtime_s']:.1f}s")
    print("\nSummary:")
    summarise(rows, "ALL")
    by_cond = defaultdict(list)
    for r in rows:
        for c in r["conditions"].split():
            by_cond[c].append(r)
    for c in sorted(by_cond):
        summarise(by_cond[c], c)

    with open(res_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
