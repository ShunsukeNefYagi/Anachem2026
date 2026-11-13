"""Re-run the pipeline and check it against the results reported in the paper.

    python verify_reproduction.py                      # ~10 s
    python verify_reproduction.py --regen-thresholds   # + rebuild threshold table (~7 min)
    python verify_reproduction.py --tif laser1traj.tif # + recompute spot intensities

Checks
  1. linking      results/linking/tracks.csv == reference/linking_tracks.csv
                  (output of the original linking run, 2026-09-01; all tracks)
  2. DC-MSS (TrackMate track of particle A, the input of the paper's figure)
                  == reference/figure3_segment_summary.csv: boundaries, classes,
                  S_MSS and the other per-segment values
  3. DC-MSS (particle A from this repository's linking)
                  boundaries and classes == the paper (0-140 free, 140-374 directed,
                  374-720 free); S_MSS differences are reported, not required to be 0
  4. threshold table sha256 (and, optionally, bit-identical regeneration)
  5. optional: spot intensities recomputed from the raw image stack
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
PAPER_SEGMENTS = [(0, 140, 'free'), (140, 374, 'directed'), (374, 720, 'free')]
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
    for c_new, c_ref in cols:
        x, y = new[c_new].to_numpy(float), ref[c_ref].to_numpy(float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return np.inf
        m = ~np.isnan(x)
        if m.any():
            worst = max(worst, float(np.max(np.abs(x[m] - y[m]))))
    return worst


def bounds(df, s='start', e='end_exclusive', c='motion_class'):
    return [(int(a), int(b), str(k)) for a, b, k in zip(df[s], df[e], df[c])]


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
    out = run_pipeline.main(['--trackmate'])
    print()

    # 1. linking
    new = pd.read_csv(os.path.join(RES, 'linking', 'tracks.csv'))
    ref = pd.read_csv(os.path.join(REF, 'linking_tracks.csv'))
    same_ids = (list(new.columns) == list(ref.columns) and len(new) == len(ref)
                and all(np.array_equal(new[c], ref[c])
                        for c in ('TRACK_ID', 'SPOT_ID', 'FRAME', 'DETECTIONS', 'SPAN')))
    d = max_abs_diff(new, ref, [(c, c) for c in ('X_um', 'Y_um', 'X_px', 'Y_px', 'INTENSITY',
                                                 'LINK_COST', 'LINK_CONFIDENCE')])
    record('linking: every spot assigned to the same track as in the original run',
           same_ids and d <= TOL,
           f'{new.TRACK_ID.nunique()} tracks, {len(new)} spots; max |diff| = {d:.1e}')

    # 2. DC-MSS on the TrackMate track (figure input)
    tm = out['trackmate']
    fig = pd.read_csv(os.path.join(REF, 'figure3_segment_summary.csv'))
    same_b = bounds(tm) == bounds(fig, 'start_min', 'end_min', 'cls')
    d = max_abs_diff(tm, fig, [('mss_slope', 'mss_slope'), ('gen_D2_px2', 'D_px2_per_min'),
                               ('conf_radius_px', 'conf_radius_px'),
                               ('straightness', 'straightness'), ('Cu_lag1', 'Cu_lag1'),
                               ('v_net_um_per_min', 'net_drift_um_per_min')])
    record('DC-MSS, TrackMate track of particle A == figure3_revision values',
           same_b and d <= TOL,
           f'{bounds(tm)}; S_MSS {np.round(tm.mss_slope.to_numpy(), 3).tolist()}; '
           f'max |diff| = {d:.1e}')

    # 3. DC-MSS on particle A from our linking
    lk = out['linked']
    ok = bounds(lk) == PAPER_SEGMENTS
    dS = np.abs(lk.mss_slope.to_numpy() - tm.mss_slope.to_numpy()) if ok else [np.nan]
    record('DC-MSS, particle A from this linking: same segments/classes as the paper',
           ok, f'{bounds(lk)}; S_MSS {np.round(lk.mss_slope.to_numpy(), 3).tolist()} '
               f'(max |S_MSS - TrackMate| = {np.max(dS):.4f})')

    # 4. thresholds
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
