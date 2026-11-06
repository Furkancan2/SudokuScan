# SudokuScan

COMP 482 Computer Vision, Project 1. From a photograph of a printed sudoku
to a 9x9 digital grid, using Harris corners, affine / projective
rectification and Tesseract OCR.

## Installation

1. Python 3.10 or newer.
2. Python packages:
   ```
   pip install -r requirements.txt
   ```
3. The Tesseract OCR engine (not a pip package):
   - Windows: installer from https://github.com/UB-Mannheim/tesseract/wiki.
     The default folder `C:\Program Files\Tesseract-OCR` is found
     automatically; otherwise set the environment variable `TESSERACT_CMD`
     to the full path of `tesseract.exe`.
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`

Check: `tesseract --version` should print version 4 or newer.

## Running

```
python sudoku_scan.py --input photo.jpg --outdir out/
python sudoku_scan.py --batch images_dir/ --outdir results/      # grader mode
python sudoku_scan.py --input photo.jpg --outdir out/ --solve    # + bonus B1, B2
```

For each image `<name>` the output folder gets:

| File | Content |
|---|---|
| `<name>_grid.txt` | 9 lines of 9 digits, 0 = empty (from H4) |
| `<name>_result.json` | corners (TL, TR, BR, BL), A3/A4/H4 matrices, A3 fourth-corner error, A4 RMS residual, grid, verdict, runtime |
| `<name>_harris.png` | Harris response map, all candidates (grey), candidates in the puzzle region (yellow), the 4 chosen corners (red) |
| `<name>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | the three 450x450 rectifications |
| `<name>_debug.png` | H4 image with cell boundaries, recognised digits, conflicting cells in red |
| `<name>_solution.txt`, `<name>_ar.png` | only with `--solve` (bonus) |

If no puzzle is found the result file contains `"verdict": "not_found"`;
the program never stops on an error.

## Code structure

| File | Stage | What it does |
|---|---|---|
| `sudokuscan/preprocess.py` | 1 | greyscale, blur, illumination normalisation, adaptive threshold; all sizes relative to image width |
| `sudokuscan/corners.py` | 2 | Harris response, threshold + NMS, cornerSubPix; calls the two files below |
| `sudokuscan/my_region.py` | 2 | puzzle region (grid-line score) and candidate filtering |
| `sudokuscan/my_corners.py` | 2 | choice of the 4 corners and quadrilateral check |
| `sudokuscan/rectify.py` | 3 | A3, A4, H4 estimation and warping, residuals |
| `sudokuscan/my_stage4.py` | 4 | obliqueness and grid-line error / bending |
| `sudokuscan/recognize.py` | 5 | cells, margin, empty-cell rule, OCR preparation, Tesseract |
| `sudokuscan/validate.py` | 6 | row / column / box check |
| `sudokuscan/visualize.py` | - | Harris and debug images |
| `sudokuscan/pipeline.py` | - | runs all stages on one image |
| `sudokuscan/bonus.py` | B1, B2, B3 | backtracking solver, AR overlay, many-point homography |
| `sudoku_scan.py` | - | command-line program |

### Corner-selection rule (Stage 2)

1. Harris response on the illumination-normalised image; keep points above
   1% of the maximum; non-maximum suppression with a radius of 1% of the
   puzzle side (so a digit next to the outer corner cannot suppress it).
2. Puzzle region (`my_region.py`): among all large four-sided contours, the
   one whose inside, warped to a square, shows about 10 evenly spaced grid
   lines in both directions (lines counted with `find_peaks`). This rejects
   the page outline, text blocks and other boxes. The contour is used only as
   a mask, never as corner coordinates.
3. Keep only Harris candidates inside the slightly enlarged region.
4. TL = min(x+y), BR = max(x+y), TR = max(x-y), BL = min(x-y). Correct for
   any in-plane rotation below 45 degrees.
5. Sanity check (convex quadrilateral covering at least 80% of the region), then
   `cv2.cornerSubPix`.
6. Orientation (bonus B4): the four cyclic corner orders are rectified and
   six digits of each are read; the order with the highest Tesseract
   confidence is kept, so pages turned by 90, 180 or 270 degrees work.

## Analysis tools

All tools read images with annotations in the Appendix A format
(`<name>.jpg` + `<name>.json` in the same folder).

```
python tools/annotate.py --batch data/personal                            # annotate all photos in photo_labels.txt
python tools/annotate.py data/personal/IMG_0001.jpg --puzzle A --conditions C2 C4   # one photo
python tools/evaluate.py --images data/personal --results results/personal --run
python tools/param_study.py --images data/personal --out analysis/params
python tools/compare_models.py --images data/personal --out analysis/compare
python tools/compare_models.py --images data/personal --out analysis/compare_gt --gt-corners
python tools/ocr_study.py --images data/personal --out analysis/ocr
python tools/failure_figures.py --out analysis/failures                  # failure-case figures
python tools/make_synthetic.py --outdir data/synthetic --n 24              # synthetic test images
```

| Tool | Report section |
|---|---|
| `evaluate.py` | Section 7 metrics, overall and per condition C1-C5 |
| `param_study.py` | Stage 2 parameter study (blockSize, ksize, k, threshold) |
| `compare_models.py` | Stage 4: residuals, grid geometry, OCR accuracy vs obliqueness; side-by-side figure |
| `ocr_study.py` | Stage 5: with/without preparation, psm modes, confusions with crops |
| `annotate.py` | personal dataset annotation |
| `failure_figures.py` | failure-case figures for the report |
| `make_synthetic.py` | synthetic photos with exact ground truth, for testing only |

## Data

- `data/personal/` - personal dataset: 13 original iPhone photos (EXIF kept)
  of three printed puzzles, one `.json` annotation per photo (Appendix A
  format), `photo_labels.txt` (puzzle letter and conditions per photo)
- `tools/puzzles_truth.json` - the givens of the three printed puzzles
- `data/dev/` - for the development set (it was not available on the LMS)
- `report/` - report source (`report.tex`, IEEEtran on Overleaf) and PDF
- `analysis/` - outputs of the analysis tools used in the report

## Results on the personal dataset (13 photos)

| Corner error | Cell accuracy | Empty-cell F1 | Exact match | Robustness | Runtime |
|---|---|---|---|---|---|
| 0.71% of side | 99.9% | 1.000 | 92.3% | 100% | 3-6 s / image |
