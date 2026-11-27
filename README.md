# Exposure Time Estimator

Estimates optimal single or multiple exposure times for laser-triangulation scans from a small set of raw calibration scans. Instead of searching for the exposure iteratively on the live image, it calibrates a per-pixel brightness model from a few full 2D scans and predicts the result for any exposure set directly.

> **Status:** prototype. Calibrated and validated on a single PCB in one setup. The sensor's merge rule and the definition of a "good" point still need to be confirmed with the sensor team, and the results need validating on more parts before production use.

---

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `Pillow`

```bash
pip install numpy scipy pillow
```

## Input

A folder with raw **16-bit** single-exposure scans of the same part. Each exposure needs three images:

| Suffix | Content |
|---|---|
| `_Z` | Height map (0 = invalid point) |
| `_I` | Intensity image |
| `_L` | Laser line width image |

The file names must contain `exposure_<time in µs>_<Z|I|L>`, as `.png`, `.tif` or `.tiff`:

```
pcb-1/scanned
  pcb_exposure_20_Z.png    pcb_exposure_20_I.png    pcb_exposure_20_L.png
  pcb_exposure_93_Z.png    pcb_exposure_93_I.png    pcb_exposure_93_L.png
  pcb_exposure_431_Z.png   pcb_exposure_431_I.png   pcb_exposure_431_L.png
  pcb_exposure_2000_Z.png  pcb_exposure_2000_I.png  pcb_exposure_2000_L.png
```

**Requirements for the calibration scans**

- At least 2 scans; **4 scans spaced evenly on a log scale are recommended:**
  `t_i = t_min · (t_max / t_min)^(i / 3)`, i = 0…3 → for [20, 2000] µs: **20 / 93 / 431 / 2000 µs**
  (`np.geomspace(t_min, t_max, 4)`)
- Same part, position and orientation in every scan.
- Same laser power, gain, ROI and scan speed. Only the exposure changes.
- Raw data, not display or screenshot exports (8-bit exports lose the intensity information).

## Usage

```bash
# Optimise 1, 2 and 3 exposures
python exposure_estimator.py <folder>

# Up to 2 exposures, save results as JSON
python exposure_estimator.py <folder> --n-max 2 --json result.json

# Only evaluate a given exposure set
python exposure_estimator.py <folder> --predict 125,1500

# Restrict to the part with a mask, or cut off a fixture by height
python exposure_estimator.py <folder> --roi-mask board_mask.png
python exposure_estimator.py <folder> --z-min 35000
```

### Options

| Option | Default | Description |
|---|---|---|
| `--n-max` | 3 | Maximum number of exposures to optimise |
| `--t-min`, `--t-max` | shortest / longest scan | Allowed exposure range [µs] |
| `--grid-points` | 61 | Number of log-spaced candidate exposures (~8 % steps for 20–2000 µs) |
| `--merge` | `shortest` | Sensor merge model: `shortest` (shortest valid exposure wins, as observed) or `best` (best point wins) |
| `--w-lo`, `--w-hi` | 0, 12 | "Good" line width range [px]: `w_lo <= L < w_hi` |
| `--w-sat` | 12 | Line width treated as saturated when fitting γ |
| `--valid-tol` | 0.5 | Accept sets within this many %-points of maximum coverage, then maximise good share |
| `--gamma`, `--threshold` | fitted | Override the fitted model parameters |
| `--roi-mask` | auto | PNG mask (non-zero = region to optimise) |
| `--z-min` | – | Automatic ROI: only Z values above this (e.g. to exclude a fixture) |
| `--predict` | – | Evaluate a given set, e.g. `125,1500` |
| `--json` | – | Write results to a JSON file |

## Output

**Example 1 – optimisation mode** (default, 4 PCB scans, automatic ROI):

```
> python exposure_estimator.py pcb-1/scanned
Scans: 20, 93, 431, 2000 us | ROI: 36,669,983 px
       20 us: valid   8.11 %   width in band   7.52 %
       93 us: valid  36.08 %   width in band  27.03 %
      431 us: valid  81.06 %   width in band  55.82 %
     2000 us: valid  94.54 %   width in band  39.06 %
Model: A=I/L grows ~ t^0.650, validity threshold A ~ 36.0
Merge model: shortest | good = valid and 0 <= L < 12 | max achievable valid: 95.02 %

 n | exposures [us]               | valid %  | good %  | sum [us]
 1 | 1588.7                       |   94.09  |  49.13  |    1589
 2 | 185.2 + 1588.7               |   94.55  |  76.25  |    1774
 3 | 54.2 + 342.3 + 1588.7        |   94.57  |  89.44  |    1985

Timing:
  load scans         11.69 s
  fit model           2.28 s
  pixel intervals     5.22 s
  build predictor    65.14 s
  optimisation        2.00 s
  total              86.33 s
```

| Block | Meaning |
|---|---|
| `Scans` / per-scan lines | Scans found, ROI size, and per scan the measured share of valid points and of points with line width in the good range |
| `Model` | Fitted exponent γ and validity threshold. If these jump between runs on similar parts, check the scans |
| `Merge model` | Merge model, definition of "good", and the maximum coverage reachable within the scanned range |
| Results table | Best set per number of exposures `n`, with predicted valid %, good % and exposure sum |
| `Timing` | Run time per processing step |

**Example 2 – `--predict` mode** (evaluate a given set, same scans):

```
Scans: 20, 93, 431, 2000 us | ROI: 36,669,983 px
       20 us: valid   8.11 %   width in band   7.52 %
       93 us: valid  36.08 %   width in band  27.03 %
      431 us: valid  81.06 %   width in band  55.82 %
     2000 us: valid  94.54 %   width in band  39.06 %
Model: A=I/L grows ~ t^0.650, validity threshold A ~ 36.0
Merge model: shortest | good = valid and 0 <= L < 12 | max achievable valid: 95.02 %
Prediction for [125.0, 1500.0] us: valid 94.41 %, good 77.07 %
```

With `--json`, the results, model parameters and timings are also saved as a file.

- **valid %:** share of ROI pixels with a valid point (Z > 0)
- **good %:** share of ROI pixels with a valid point whose line width is in the good range

---

## Processing pipeline

### 1. Calibration scans
Several single-exposure scans spaced on a log scale (recommended: 4). Each provides Z (valid or not), I (intensity) and L (line width) per pixel.

### 2. Region of interest
Only pixels on the part count: automatically the largest connected valid area in the longest scan (closing + hole filling), or a given mask. All percentages refer to this area.

### 3. Fitting the model
- **Peak brightness** per pixel: `A = I / L`
- **Growth with exposure:** `A ~ t^γ`. γ comes from the brightness ratios of unsaturated points (L < `w_sat`) between neighbouring scans (median). PCB: γ ≈ 0.65–0.67.
- **Validity threshold:** a point is valid when `A` is above a threshold. For each neighbouring scan pair, the cutoff in the longer scan that separates the points still valid in the shorter scan is converted back to a threshold with γ (median over pairs). PCB: ≈ 35.

### 4. Exposure range per pixel
- **t_need (minimum exposure):** lies between the last scan where the pixel is invalid and the first where it is valid. Within that bracket:
  `t_need = t · (threshold / A)^(1/γ)`, limited to the bracket.
- **t_lost (overexposure):** valid in one scan, invalid in the next → the point drops out between them (geometric mean of the two exposures).
- **t_w_lo / t_w_hi (line width):** exposures at which the line width enters or leaves the good range, interpolated on a log-log scale from the widths measured across the scans.

### 5. Predicting an exposure set
- **Valid:** at least one exposure in the set lies in `[t_need, t_lost]`.
- **Good:** decided by the merge model:
  - `shortest`: the point from the shortest valid exposure is used, and it must be in the good range.
  - `best`: good if any exposure in the set delivers a good point.

### 6. Optimisation
- **Candidates:** log-spaced exposures between `t_min` and `t_max` (default 61).
- **Search:** every combination of 1…`n_max` exposures is evaluated. The per-pixel ranges are compressed onto the grid first, so this stays fast.
- **Selection:** among the sets within `valid_tol` of the maximum coverage → most good points → smallest exposure sum.
- **Output:** the best set per number of exposures, with predicted valid and good shares.

```
Calibration scans (Z, I, L)
        │
        ▼
ROI ──► fit γ and threshold on A = I/L
        │
        ▼
Per pixel: t_need, t_lost, t_w_lo, t_w_hi
        │
        ▼
Grid search over 1…N exposures (merge model)
        │
        ▼
Best set per N + predicted valid / good share
```

---

## Validation (PCB, one setup)

Model fitted from the 4 calibration scans only, compared with separate measurements:

| Setting | Predicted valid / good | Measured valid / good |
|---|---|---|
| 300 µs | 57.0 / 38.9 % | 58.9 / 37.3 % |
| 20 + 500 µs | 84.5 / 65.1 % | 84.2 / 65.2 % |
| 125 + 1500 µs | 94.2 / 75.7 % | 94.4 / 71.6 % |
| 50 + 300 + 1500 µs | 94.3 / 89.3 % | 94.4 / 84.3 % |

- **Valid coverage:** accurate to about 0.1–2 points. Pixel-level agreement for the multi-exposure sets was 99.7 %.
- **Good share (L < 12):** about 4–5 points too optimistic. The misses are mostly borderline points (L = 12–13).
- **Number of scans:** leaving one scan out costs about 1 point of accuracy when the gap is ×4.6, and about 5 points when it is ×21.

## Known limitations and open points

1. **Merge rule:** how the sensor picks the point per pixel in multi-exposure mode, and whether it can be configured. This has to be confirmed with the sensor team. The PCB results suggest the shortest valid exposure wins.
2. **Quality criterion:** "L < 12" (unsaturated) is not the best proxy for Z quality. On the PCB, Z noise was lowest at a line width of about 9–20 px and highest for faint, narrow lines (L < 6). A band like `--w-lo 9 --w-hi 20` is available but not yet validated.
3. **Generalisation:** calibrated on one PCB only. Validate on more parts and surfaces (e.g. machined metal, dark plastic, shiny parts) and check repeatability between runs.
4. **Scan speed:** whether the profile rate depends on the sum of the exposures or the longest one affects which set is optimal. Currently only the exposure sum is used as a tie-breaker.
5. **Shadows/occlusion:** points invalid even at `t_max` (e.g. behind tall components) cannot be recovered by any exposure and are counted as not achievable.
6. **Runtime:** about 86 s in total for 4 full-resolution scans (~10k × 4.8k px, ROI ~37 M px), measured on a Windows laptop. The main bottleneck is "build predictor" (~65 s), which compresses the per-pixel ranges onto the grid. The search itself takes only about 2 s. This step can be sped up considerably (e.g. combine the four grid indices into one integer key before counting); worth doing before any inline use.

All figures above come from one PCB and one setup. Check them before using them in specifications or customer communication.
