"""Make synthetic test photographs with exact ground truth.

These are NOT a replacement for the personal dataset (that must be real phone
photos). They are only for testing the code before real photos exist, and
for controlled experiments (e.g. the exact viewing angle is known).

  python tools/make_synthetic.py --outdir data/synthetic --n 24
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/Library/Fonts/Arial.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/times.ttf",
]
FONTS = [f for f in FONT_CANDIDATES if Path(f).exists()]


def load_font(path, size):
    """TrueType font if one was found on this computer, else Pillow's own."""
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)
WORDS = ("the city council said on monday that new bus lines will open next "
         "month weather sunny cloudy rain market prices rose sharply puzzle "
         "answers page crossword horoscope football league results").split()


def make_puzzle(rng):
    """A valid sudoku: shuffled base pattern with ~55 cells removed."""
    base = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    digits = list(range(1, 10))
    rng.shuffle(digits)
    rows = [b * 3 + r for b in rng.sample(range(3), 3) for r in rng.sample(range(3), 3)]
    cols = [b * 3 + c for b in rng.sample(range(3), 3) for c in rng.sample(range(3), 3)]
    sol = [[digits[base[r][c] - 1] for c in cols] for r in rows]
    grid = [row[:] for row in sol]
    for idx in rng.sample(range(81), rng.randint(50, 58)):
        grid[idx // 9][idx % 9] = 0
    return grid


def draw_paper(grid, rng, clutter):
    """Draw the puzzle on a white page. Returns (page, puzzle corners)."""
    W, H, S = 1400, 1800, 900
    page = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(page)
    x0, y0 = 250, 450
    font = load_font(rng.choice(FONTS) if FONTS else None, rng.randint(56, 68))
    small = load_font(FONTS[0] if FONTS else None, 30)

    # newspaper-like text around the puzzle
    if clutter:
        for y in list(range(60, 400, 45)) + list(range(1400, 1760, 45)):
            line = " ".join(rng.choice(WORDS) for _ in range(9))
            d.text((60, y), line, fill=30, font=small)
        d.rectangle([60, 470, 210, 620], outline=0, width=6)   # a second box

    # grid lines: thin, box and outer lines
    thick_out, thick_box, thick_thin = rng.choice([(8, 5, 2), (6, 4, 2), (10, 6, 3)])
    c = S / 9
    for i in range(10):
        t = thick_box if i % 3 == 0 else thick_thin
        p = round(i * c)
        d.rectangle([x0 + p - t // 2, y0, x0 + p + t // 2, y0 + S], fill=0)
        d.rectangle([x0, y0 + p - t // 2, x0 + S, y0 + p + t // 2], fill=0)
    # outer border lies exactly INSIDE [x0, x0+S]
    d.rectangle([x0, y0, x0 + S - 1, y0 + S - 1], outline=0, width=thick_out)

    for r in range(9):
        for col in range(9):
            if grid[r][col]:
                cx, cy = x0 + (col + 0.5) * c, y0 + (r + 0.5) * c
                d.text((cx, cy), str(grid[r][col]), fill=0, font=font, anchor="mm")

    corners = np.float32([[x0, y0], [x0 + S, y0], [x0 + S, y0 + S], [x0, y0 + S]])
    return np.array(page), corners


def camera_homography(page_w, page_h, img_w, img_h, tilt, roll, yaw, rng):
    """Homography that a pinhole camera produces when looking at the page.
    tilt / yaw = rotation of the page out of the image plane (degrees),
    roll = in-plane rotation (degrees)."""
    f = 0.9 * img_w                                       # phone-like lens
    a, b, g = np.radians([tilt, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    scale = rng.uniform(0.55, 0.75) * img_h / page_h       # page height in image
    Z = f / scale
    src = np.float32([[0, 0], [page_w, 0], [page_w, page_h], [0, page_h]])
    pts3 = np.c_[src - [page_w / 2, page_h / 2], np.zeros(4)] @ R.T + [0, 0, Z]
    shift = [img_w / 2 + rng.uniform(-0.08, 0.08) * img_w,
             img_h / 2 + rng.uniform(-0.05, 0.05) * img_h]
    dst = f * pts3[:, :2] / pts3[:, 2:3] + shift
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def make_background(w, h, rng):
    base = np.array([rng.randint(60, 200) for _ in range(3)], np.float32)
    noise = cv2.GaussianBlur(np.random.rand(h // 8, w // 8, 3).astype(np.float32), (0, 0), 3)
    noise = cv2.resize(noise, (w, h))
    bg = base * (0.7 + 0.6 * noise)
    return np.clip(bg, 0, 255).astype(np.uint8)


def add_shadow(img, rng, strength):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ang = rng.uniform(0, np.pi)
    d = (xx - w * rng.uniform(0.3, 0.7)) * np.cos(ang) + (yy - h * rng.uniform(0.3, 0.7)) * np.sin(ang)
    m = 1 - strength * (1 / (1 + np.exp(-d / (0.03 * w))))
    return np.clip(img * m[..., None], 0, 255).astype(np.uint8)


def make_one(i, rng, kind):
    tilt, roll, yaw, clutter, shadow, dark = 0, 0, 0, False, 0, False
    conds = []
    if kind == "C1":
        tilt, yaw, roll = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-4, 4)
    elif kind == "C2":
        tilt, yaw = rng.uniform(15, 55) * rng.choice([1, -1]), rng.uniform(-10, 10)
        roll = rng.uniform(-6, 6)
    elif kind == "C3":
        roll = rng.uniform(15, 30) * rng.choice([1, -1])
        tilt = rng.uniform(-10, 10)
    elif kind == "C4":
        tilt, roll = rng.uniform(-15, 15), rng.uniform(-8, 8)
        shadow = rng.uniform(0.4, 0.6)
        dark = rng.random() < 0.4
    elif kind == "C5":
        tilt, roll = rng.uniform(-20, 20), rng.uniform(-12, 12)
        clutter = True
    conds.append(kind)
    if abs(tilt) >= 15 and kind != "C2":
        conds.append("C2")
    if abs(roll) >= 15 and kind != "C3":
        conds.append("C3")

    grid = make_puzzle(rng)
    page, corners = draw_paper(grid, rng, clutter)
    w, h = rng.choice([(1600, 1200), (2400, 1800), (4000, 3000), (1200, 1600)])
    Hm = camera_homography(page.shape[1], page.shape[0], w, h, tilt, roll, yaw, rng)

    bg = make_background(w, h, rng)
    paper = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    paper = (paper * np.array([0.93, 0.96, 1.0])).astype(np.uint8)  # warm paper
    warped = cv2.warpPerspective(paper, Hm, (w, h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(page.shape, 255, np.uint8), Hm, (w, h)) > 0
    img = bg.copy()
    img[mask] = warped[mask]

    if shadow:
        img = add_shadow(img, rng, shadow)
    if dark:
        img = (img * 0.45).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.4) * w / 1600)
    noise = np.random.normal(0, 4 + 6 * dark, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    gt = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    name = f"synth_{i:02d}"
    ann = {"image": name + ".jpg", "conditions": conds,
           "corners": gt.round(2).tolist(), "grid": grid,
           "tilt_deg": round(tilt, 1), "roll_deg": round(roll, 1)}
    return name, img, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/synthetic")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["C1", "C2", "C2", "C3", "C4", "C5"]
    for i in range(args.n):
        name, img, ann = make_one(i, rng, kinds[i % len(kinds)])
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / f"{name}.json").write_text(json.dumps(ann))
        print(name, ann["conditions"], "tilt", ann["tilt_deg"], "roll", ann["roll_deg"])


if __name__ == "__main__":
    main()
