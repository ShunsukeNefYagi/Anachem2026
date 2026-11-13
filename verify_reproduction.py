"""Re-run the pipeline and check that it reproduces the original analysis run.

    python verify_reproduction.py                      # ~10 s
    python verify_reproduction.py --regen-thresholds   # + rebuild threshold table (~7 min)
    python verify_reproduction.py --tif laser1traj.tif # + recompute spot intensities

These are reproducibility checks: they compare the current output with the stored
results in reference/, i.e. they confirm that the code and data give the published
numbers again.  They do not test whether the analysis itself
(linking, DC-MSS classification) is correct.

Checks
  1. linking      results/linking/tracks.csv == reference/linking_tracks.csv
                  (output of the original linking run, 2026-09-01; all tracks)
  2. DC-MSS       results/dcmss/particleA_segments.csv
                  == reference/dcmss_segments_particleA.csv (the paper's result for
                  particle A): boundaries, classes, S_MSS and every other column
  3. threshold table sha256 (and, optionally, bit-identical regeneration)
  4. optional: spot intensities recomputed from the raw image stack
A summary is written to results/verification.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(ROOT, 'reference')
RES = os.path.join(ROOT, 'results')
TOL = 1e-10
THR_SHA256 = '575093dc994831f736ec9880d3ca5976b080b8748447811659ffef092ed6e6aa'
checks = []


def record(name, ok, detail=''):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'\n       {detail}' if detail else ''),
          flush=True)


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def max_abs_diff(new, ref, cols):
    """Max |new - ref| over numeric columns (NaN positions must agree)."""
    worst = 0.0
    for c in cols:
        x, y = new[c].to_numpy(float), ref[c].to_numpy(float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return np.inf
        m = ~np.isnan(x)
        if m.any():
            worst = max(worst, float(np.max(np.abs(x[m] - y[m]))))
    return worst


def bounds(df):
    return [(int(a), int(b), str(k))
            for a, b, k in zip(df.start, df.end_exclusive, df.motion_class)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regen-thresholds', action='store_true')
    ap.add_argument('--tif', help='raw image stack, to re-check spot intensities')
    a = ap.parse_args()

    py = sys.executable
    if a.tif:
        subprocess.run([py, os.path.join(ROOT, 'src', 'spot_intensity.py'),
                        '--tif', a.tif], check=True)
        new = pd.read_csv(os.path.join(RES, 'linking', 'spot_intensity_square_roi.csv'))
        old = pd.read_csv(os.path.join(ROOT, 'data', 'spot_intensity_square_roi.csv'))
        record('spot intensities recomputed from the image stack == data/',
               new.equals(old), f'{len(new)} spots')

    sys.path.insert(0, ROOT)
    import run_pipeline
    seg = run_pipeline.main([])
    print()

    # 1. linking
    new = pd.read_csv(os.path.join(RES, 'linking', 'tracks.csv'))
    ref = pd.read_csv(os.path.join(REF, 'linking_tracks.csv'))
    same_ids = (list(new.columns) == list(ref.columns) and len(new) == len(ref)
                and all(np.array_equal(new[c], ref[c])
                        for c in ('TRACK_ID', 'SPOT_ID', 'FRAME', 'DETECTIONS', 'SPAN')))
    d = max_abs_diff(new, ref, ['X_um', 'Y_um', 'X_px', 'Y_px', 'INTENSITY',
                                'LINK_COST', 'LINK_CONFIDENCE'])
    record('linking: every spot assigned to the same track as in the original run',
           same_ids and d <= TOL,
           f'{new.TRACK_ID.nunique()} tracks, {len(new)} spots; max |diff| = {d:.1e}')

    # 2. DC-MSS on particle A
    ref = pd.read_csv(os.path.join(REF, 'dcmss_segments_particleA.csv'))
    num = [c for c in ref.columns if c != 'motion_class']
    same_cols = list(seg.columns) == list(ref.columns) and len(seg) == len(ref)
    d = max_abs_diff(seg, ref, num) if same_cols else np.inf
    record('DC-MSS, particle A: same segments, classes and values as the paper',
           same_cols and bounds(seg) == bounds(ref) and d <= TOL,
           f'{bounds(seg)}; S_MSS {np.round(seg.mss_slope.to_numpy(), 3).tolist()}; '
           f'max |diff| = {d:.1e}')

    # 3. thresholds
    shipped = os.path.join(ROOT, 'data', 'thr_sig0.14_lf3.npz')
    record('threshold table sha256', sha256(shipped) == THR_SHA256, sha256(shipped)[:16] + '...')
    if a.regen_thresholds:
        subprocess.run([py, os.path.join(ROOT, 'src', 'gen_thresholds.py')], check=True)
        n = np.load(os.path.join(RES, 'calibration', 'thr_sig0.14_lf3.npz'), allow_pickle=True)
        o = np.load(shipped, allow_pickle=True)
        record('threshold table regenerated bit-identically',
               all(np.array_equal(n[k], o[k]) for k in ('durations', 'slopes')))

    n_fail = sum(not c['ok'] for c in checks)
    import matplotlib
    import scipy
    env = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
               pandas=pd.__version__, matplotlib=matplotlib.__version__)
    inputs = {f: sha256(os.path.join(ROOT, 'data', f))
              for f in sorted(os.listdir(os.path.join(ROOT, 'data')))}
    with open(os.path.join(RES, 'verification.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(passed=n_fail == 0, checks=checks, environment=env,
                       inputs_sha256=inputs, options=vars(a)), fh, indent=2)
    print(f'\n{len(checks) - n_fail}/{len(checks)} checks passed'
          + ('' if not n_fail else f', {n_fail} FAILED'))
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
