"""Stage 1 - Preprocessing.

All sizes are set relative to the image width, so the same code works on a
1000 px and a 4000 px photograph.
"""
import cv2
import numpy as np

# Large photos are shrunk so that their longest side is at most this many
# pixels before corner detection. Corners are scaled back at the end.
WORK_SIZE = 1000


def odd(n, minimum=3):
    """Round n to the nearest odd integer (OpenCV kernels must be odd)."""
    n = max(minimum, int(round(n)))
    return n if n % 2 == 1 else n + 1


def resize_for_work(img):
    """Downscale a large image. Returns (small_image, scale) where
    small = original * scale."""
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h, w))
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img, scale


def normalize_illumination(gray):
    """Divide the image by an estimate of its background brightness.

    A morphological closing with a big kernel removes dark, thin things
    (grid lines, digits, text) and keeps the slowly changing paper brightness,
    including shadows. Dividing by it makes the paper uniformly white.
    """
    w = gray.shape[1]
    k = odd(w * 0.04)                               # ~40 px on a 1000 px image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (k, k), 0)
    norm = gray.astype(np.float32) / (background.astype(np.float32) + 1.0)
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm


def preprocess(img_bgr):
    """Return (gray, norm, binary) for a work-size BGR image.

    gray   : plain greyscale image, lightly blurred
    norm   : greyscale with uneven lighting removed (input for Harris)
    binary : white ink on black background (used to find the puzzle region)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]

    blur_k = odd(w * 0.003)                         # ~3 px on a 1000 px image
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    norm = normalize_illumination(gray)

    block = odd(w * 0.025)                          # ~25 px neighbourhood
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 10)
    return gray, norm, binary
