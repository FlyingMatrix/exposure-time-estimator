#!/usr/bin/env python3
"""
    Exposure time estimator for laser-triangulation scans (prototype).

    Input : a folder with raw single-exposure scans of the same part, e.g.
                pcb_exposure_20_Z.png   pcb_exposure_20_I.png   pcb_exposure_20_L.png
                pcb_exposure_93_Z.png   ...
            (Z = height map, 0 = invalid; I = intensity; L = laser line width).
            Recommended: 4 scans log-spaced over the allowed range,
            e.g. np.geomspace(t_min, t_max, 4)  ->  20 / 93 / 431 / 2000 us.
    Output: the best exposure set for 1..N exposures, with predicted coverage
            and predicted share of "good" points.

    Model (per pixel):
    * Validity is decided by peak brightness A = I / L (roughly constant threshold).
    * A grows ~ t^gamma (gamma < 1). gamma and the threshold are fitted from the
        scans automatically (can be overridden).
    * t_need  : exposure at which the pixel becomes valid. It is bracketed by the
                scans (last invalid / first valid) and interpolated inside that
                bracket with the power law.
    * t_lost  : exposure above which a previously valid pixel drops out again
                (overexposure).
    * t_w_lo / t_w_hi : exposures at which the line width crosses the "good"
                band [w_lo, w_hi] (log-log interpolation of L between scans).
    A pixel is "valid" at t if t_need <= t <= t_lost, and "good" at t if it is
    valid and its line width is inside the good band.

    Merge models for multi-exposure:
    best     : the sensor picks a good point if any exposure in the set gives one.
    shortest : the sensor takes the point of the shortest exposure at which the
                pixel is valid (this matched the merged results observed on the
                PCB test).

    Caveats: prototype, calibrated and validated on one PCB only. Check the merge
    rule with the sensor team and validate on more parts before production use.
"""
import argparse
import itertools
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as nd

FNAME_RE = re.compile(r"exposure_(\d+(?:\.\d+)?)_([ZIL])\.(png|tif|tiff)$", re.IGNORECASE)


# --------------------------------------------------------------------------- I/O
def load_image(path):
    img = np.array(Image.open(path))
    if img.ndim == 3:  # e.g. accidental RGB export -> take first channel
        img = img[..., 0]
    return img


def find_scans(folder):
    """Return {exposure_us: {'Z': path, 'I': path, 'L': path}} for complete scans."""
    scans = {}
    for name in os.listdir(folder):
        m = FNAME_RE.search(name)
        if m:
            t = float(m.group(1))
            scans.setdefault(t, {})[m.group(2).upper()] = os.path.join(folder, name)
    complete = {t: p for t, p in scans.items() if {"Z", "I", "L"} <= p.keys()}
    missing = sorted(set(scans) - set(complete))
    if missing:
        print(f"[warn] incomplete scans ignored (need Z, I and L): {missing}", file=sys.stderr)
    if len(complete) < 2:
        sys.exit("Need at least 2 complete scans (Z, I, L per exposure); 4 log-spaced scans recommended.")
    return dict(sorted(complete.items()))


def build_roi(valid_longest, z_longest, roi_mask_path=None, z_min=None):
    """ROI = region whose coverage is optimised."""
    if roi_mask_path:
        roi = load_image(roi_mask_path) > 0
        if roi.shape != valid_longest.shape:
            sys.exit(f"ROI mask shape {roi.shape} does not match scans {valid_longest.shape}")
        return roi
    seed = valid_longest if z_min is None else valid_longest & (z_longest > z_min)
    roi = nd.binary_closing(seed, iterations=30)
    roi = nd.binary_fill_holes(roi)
    lab, n = nd.label(roi)
    if n == 0:
        sys.exit("Automatic ROI is empty - check the scans or pass --roi-mask / --z-min.")
    sizes = nd.sum(roi, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def load_scans(folder, roi_mask_path=None, z_min=None):
    paths = find_scans(folder)
    T = np.array(list(paths.keys()))
    z_long = load_image(paths[T[-1]]["Z"])
    roi = build_roi(z_long > 0, z_long, roi_mask_path, z_min)
    del z_long
    V, A, W = [], [], []
    for t in T:
        z = load_image(paths[t]["Z"])
        i = load_image(paths[t]["I"]).astype(np.float32)
        w = load_image(paths[t]["L"]).astype(np.float32)
        if z.shape != roi.shape:
            sys.exit(f"Scan {t} us has shape {z.shape}, expected {roi.shape}")
        v = (z > 0)[roi]
        V.append(v)
        W.append(w[roi])
        A.append((i[roi] / np.maximum(w[roi], 1.0)))
    return T, np.array(V), np.array(A), np.array(W), roi


# ---------------------------------------------------------------- calibration
def fit_gamma_threshold(T, V, A, W, w_sat):
    """gamma from A ratios of unsaturated points; threshold from validity cut-offs."""
    gammas, thrs = [], []
    for j in range(len(T) - 1):
        lo, hi = j, j + 1
        r = T[hi] / T[lo]
        m = V[lo] & V[hi] & (W[lo] < w_sat) & (W[hi] < w_sat)
        if m.sum() > 1000:
            gammas.append(np.log(np.median(A[hi][m] / np.maximum(A[lo][m], 1e-3))) / np.log(r))
    gamma = float(np.median(gammas)) if gammas else 0.67
    for j in range(len(T) - 1):
        lo, hi = j, j + 1
        m = V[hi]
        frac = V[lo][m].mean()
        if m.sum() > 1000 and 0.01 < frac < 0.99:
            cutoff = np.quantile(A[hi][m], 1 - frac)  # A at t_hi that separates valid/invalid at t_lo
            thrs.append(cutoff / (T[hi] / T[lo]) ** gamma)
    thr = float(np.median(thrs)) if thrs else 30.0
    return gamma, thr


# --------------------------------------------------------- per-pixel intervals
def crossing_time(T, V, W, level, t_need, w_floor=4.0):
    """Exposure at which line width crosses `level` (log-log interpolation).
    inf if it never reaches `level` within the scanned range."""
    n = V.shape[1]
    t_cross = np.full(n, np.inf, np.float32)
    done = np.zeros(n, bool)
    prev_t = t_need.copy()                      # start point: threshold, width ~ w_floor
    prev_w = np.full(n, w_floor, np.float32)
    for j, t in enumerate(T):
        v, w = V[j], W[j]
        hit = v & (w >= level) & ~done
        with np.errstate(all="ignore"):
            f = (np.log(level) - np.log(prev_w[hit])) / (np.log(w[hit]) - np.log(prev_w[hit]))
            tc = prev_t[hit] * (t / prev_t[hit]) ** np.clip(np.nan_to_num(f, nan=1.0), 0, 1)
        t_cross[hit] = np.minimum(tc, t)
        done |= hit
        upd = v & (w < level) & ~done
        prev_t[upd] = t
        prev_w[upd] = w[upd]
    return t_cross


def pixel_intervals(T, V, A, W, gamma, thr, w_lo, w_hi):
    n = V.shape[1]
    first = np.full(n, -1)
    for j in range(len(T)):
        first[(first < 0) & V[j]] = j
    t_need = np.full(n, np.inf, np.float32)
    for j, t in enumerate(T):
        m = first == j
        est = t * (thr / np.maximum(A[j][m], 1e-3)) ** (1.0 / gamma)
        lo = T[j - 1] * 1.0001 if j > 0 else 1e-3
        t_need[m] = np.clip(est, lo, t)
    t_lost = np.full(n, np.inf, np.float32)
    for j in range(len(T) - 1):
        m = V[j] & ~V[j + 1] & np.isinf(t_lost) & (first >= 0) & (first <= j)
        t_lost[m] = np.sqrt(T[j] * T[j + 1])
    t_w_lo = crossing_time(T, V, W, w_lo, t_need) if w_lo > 0 else t_need.copy()
    t_w_lo = np.maximum(t_w_lo, t_need)
    t_w_hi = crossing_time(T, V, W, w_hi, t_need)
    return t_need, t_lost, t_w_lo, t_w_hi


# ------------------------------------------------------------- optimisation
class Predictor:
    """Compresses per-pixel intervals onto an exposure grid for fast evaluation."""

    def __init__(self, grid, t_need, t_lost, t_w_lo, t_w_hi):
        self.grid = grid
        g = grid
        i_need = np.searchsorted(g, t_need, "left")
        i_vhi = np.searchsorted(g, t_lost, "right") - 1
        i_glo = np.searchsorted(g, t_w_lo, "left")
        i_ghi = np.searchsorted(g, np.minimum(t_w_hi, t_lost), "right") - 1
        keys = np.stack([i_need, i_vhi, i_glo, i_ghi], 1)
        u, cnt = np.unique(keys, axis=0, return_counts=True)
        self.n_total = t_need.size
        keep = u[:, 0] < len(g)
        self.u, self.cnt = u[keep], cnt[keep]
        self.max_valid = self.cnt.sum() / self.n_total * 100

    def evaluate(self, idx, merge):
        a, vh, gl, gh = self.u.T
        valid = np.zeros(len(a), bool)
        good = np.zeros(len(a), bool)
        chosen = np.full(len(a), -1)
        for s in sorted(idx):
            ok = (a <= s) & (s <= vh)
            if merge == "best":
                good |= ok & (gl <= s) & (s <= gh)
            else:  # shortest valid exposure wins
                chosen[ok & (chosen < 0)] = s
            valid |= ok
        if merge == "shortest":
            good = (chosen >= 0) & (gl <= chosen) & (chosen <= gh)
        return (self.cnt[valid].sum() / self.n_total * 100,
                self.cnt[good].sum() / self.n_total * 100)


def optimise(pred, n_max, merge, valid_tol):
    g = pred.grid
    results = {}
    for n in range(1, n_max + 1):
        rows = []
        for S in itertools.combinations(range(len(g)), n):
            v, q = pred.evaluate(S, merge)
            rows.append((v, q, float(g[list(S)].sum()), [float(g[s]) for s in S]))
        vmax = max(r[0] for r in rows)
        cand = [r for r in rows if r[0] >= vmax - valid_tol]
        cand.sort(key=lambda r: (-r[1], r[2]))       # best quality, then shortest total
        v, q, tsum, S = cand[0]
        results[n] = {"exposures_us": S, "valid_pct": round(float(v), 2),
                      "good_pct": round(float(q), 2), "exposure_sum_us": round(tsum, 1),
                      "max_valid_pct_for_n": round(float(vmax), 2)}
    return results


# ---------------------------------------------------------------------- timing
def print_timings(timings, t_start):
    total = time.perf_counter() - t_start
    print("\nTiming:")
    for name, sec in timings.items():
        print(f"  {name:<16} {sec:7.2f} s")
    print(f"  {'total':<16} {total:7.2f} s")


# ----------------------------------------------------------------------- main
def main():
    timings = {}
    t_start = time.perf_counter()
    t_last = t_start

    def lap(name):
        nonlocal t_last
        now = time.perf_counter()
        timings[name] = now - t_last
        t_last = now

    ap = argparse.ArgumentParser(description="Estimate optimal (multi-)exposure times from raw single scans.")
    ap.add_argument("folder", help="folder with *_exposure_<t>_{Z,I,L}.png scans")
    ap.add_argument("--n-max", type=int, default=3, help="max number of exposures (default 3)")
    ap.add_argument("--t-min", type=float, help="min allowed exposure [us] (default: shortest scan)")
    ap.add_argument("--t-max", type=float, help="max allowed exposure [us] (default: longest scan)")
    ap.add_argument("--grid-points", type=int, default=61, help="log-spaced candidate exposures (default 61 ~ 8%% steps for 20-2000)")
    ap.add_argument("--merge", choices=["best", "shortest"], default="shortest",
                    help="sensor merge model for multi-exposure (default: shortest, as observed)")
    ap.add_argument("--w-lo", type=float, default=0.0, help="good line width lower bound [px] (default 0 = off)")
    ap.add_argument("--w-hi", type=float, default=12.0, help="good line width upper bound [px] (default 12)")
    ap.add_argument("--w-sat", type=float, default=12.0, help="width treated as saturated for gamma fit (default 12)")
    ap.add_argument("--valid-tol", type=float, default=0.5,
                    help="accept sets within this many %%-points of max coverage, then maximise good share (default 0.5)")
    ap.add_argument("--gamma", type=float, help="override fitted brightness exponent")
    ap.add_argument("--threshold", type=float, help="override fitted validity threshold on I/L")
    ap.add_argument("--roi-mask", help="optional PNG mask (non-zero = region to optimise)")
    ap.add_argument("--z-min", type=float, help="auto ROI: only Z above this value (e.g. to exclude a fixture)")
    ap.add_argument("--predict", help="only evaluate given set, e.g. '125,1500'")
    ap.add_argument("--json", help="write results to this JSON file")
    args = ap.parse_args()

    T, V, A, W, roi = load_scans(args.folder, args.roi_mask, args.z_min)
    lap("load scans")
    print(f"Scans: {', '.join(f'{t:g}' for t in T)} us | ROI: {roi.sum():,} px")
    for j, t in enumerate(T):
        print(f"  {t:>7g} us: valid {V[j].mean()*100:6.2f} %   width in band {(V[j] & (W[j] >= args.w_lo) & (W[j] < args.w_hi)).mean()*100:6.2f} %")

    gamma, thr = fit_gamma_threshold(T, V, A, W, args.w_sat)
    lap("fit model")
    gamma = args.gamma or gamma
    thr = args.threshold or thr
    print(f"Model: A=I/L grows ~ t^{gamma:.3f}, validity threshold A ~ {thr:.1f}")

    t_need, t_lost, t_w_lo, t_w_hi = pixel_intervals(T, V, A, W, gamma, thr, args.w_lo, args.w_hi)
    lap("pixel intervals")
    del V, A, W

    t_min = args.t_min or float(T[0])
    t_max = args.t_max or float(T[-1])
    grid = np.unique(np.round(np.geomspace(t_min, t_max, args.grid_points), 1))
    if args.predict:  # evaluate the requested exposures exactly, not snapped to the grid
        S = [float(x) for x in args.predict.split(",")]
        grid = np.unique(np.concatenate([grid, S]))
    pred = Predictor(grid, t_need, t_lost, t_w_lo, t_w_hi)
    lap("build predictor")
    band = f"{args.w_lo:g} <= L < {args.w_hi:g}"
    print(f"Merge model: {args.merge} | good = valid and {band} | max achievable valid: {pred.max_valid:.2f} %")

    if args.predict:
        idx = [int(np.argmin(np.abs(grid - s))) for s in S]
        v, q = pred.evaluate(idx, args.merge)
        print(f"Prediction for {S} us: valid {v:.2f} %, good {q:.2f} %")
        lap("prediction")
        print_timings(timings, t_start)
        return

    res = optimise(pred, args.n_max, args.merge, args.valid_tol)
    lap("optimisation")
    print("\n n | exposures [us]               | valid %  | good %  | sum [us]")
    for n, r in res.items():
        ex = " + ".join(f"{e:g}" for e in r["exposures_us"])
        print(f" {n} | {ex:<28} | {r['valid_pct']:7.2f}  | {r['good_pct']:6.2f}  | {r['exposure_sum_us']:7.0f}")
    print_timings(timings, t_start)
    if args.json:
        out = {"scans_us": T.tolist(), "gamma": gamma, "threshold": thr, "merge": args.merge,
               "good_band_px": [args.w_lo, args.w_hi], "roi_px": int(roi.sum()),
               "max_valid_pct": round(pred.max_valid, 2), "results": res,
               "timings_s": {k: round(v, 3) for k, v in timings.items()}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()



"""
    To run:
        python exposure_estimator.py <folder_with_scans>
        python exposure_estimator.py <folder> --n-max 3 --json result.json
        python exposure_estimator.py <folder> --predict 125,1500                # evaluate a given set only

    Input: 
        Raw 16-bit scans named like *_exposure_<t>_Z/I/L.png, at least 2, ideally 4 log-spaced (e.g. 20 / 93 / 431 / 2000 µs).

        General rule for any range [t_min, t_max]:
            3 scans: t_min, √(t_min · t_max), t_max
            For a range above about a factor of 100, or more reliability: 4 scans, evenly spaced on a log scale. 
            For [20, 2000] that's about 20 / 90 / 430 / 2000.

        Calculation for [20, 2000] with 4 scans:
            Total ratio: 2000 / 20 = 100
            Four scans means 3 equal steps, so the factor per step is r = 100^(1/3) ≈ 4.64
            The scans:
            20 µs
            20 × 4.64 ≈ 92.8 µs
            92.8 × 4.64 ≈ 430.9 µs
            430.9 × 4.64 = 2000 µs

        General formula for N scans in [t_min, t_max]:
            t_i = t_min · (t_max / t_min)^(i / (N−1)), i = 0 … N−1
            With N = 3 this gives exactly √(t_min · t_max), the 200 µs from before.

        Same result in log₁₀:
            log₁₀ 20 = 1.301 and log₁₀ 2000 = 3.301
            Split into 3 equal steps of 0.667: 1.301 -> 1.968 -> 2.634 -> 3.301
            That gives 10^1.968 ≈ 93 and 10^2.634 ≈ 431

        In Python code:
            import numpy as np
            np.geomspace(20, 2000, 4)   # -> [20., 92.83, 430.89, 2000.]

    Output: 
        The best set of 1, 2 and 3 exposures (or up to --n-max), each with the predicted share of valid points, 
        share of good points and exposure sum. --json also saves the results as a JSON file.
"""
