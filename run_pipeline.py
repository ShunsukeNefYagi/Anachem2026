"""Detections -> linking -> particle A trajectory -> DC-MSS.

    python run_pipeline.py              # whole pipeline (~5 s)
    python run_pipeline.py --trackmate  # also run DC-MSS on the TrackMate-linked
                                        # track of particle A (data/particleA_trackmate.csv)

Outputs (results/):
    linking/spots.csv, linking/tracks.csv     all detections / all linked tracks
    particleA_track.csv                       particle A, one row per frame
    dcmss/particleA_segments.csv              DC-MSS segments (half-open frame bounds)
    dcmss/particleA_stages.json               boundaries after each DC-MSS step
    dcmss/particleA_dcmss.png                 trajectory + S_MSS per segment
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import dcmss_analysis as DA          # noqa: E402
from linking import run_linking      # noqa: E402
from tracks_io import load_tracks    # noqa: E402

RES = os.path.join(ROOT, 'results')
# Particle A ("Adot (A)" in the paper) is identified by its position at frame 0.
PARTICLE_A_START_PX = (1327.67, 255.73)


def select_particle(tracks, start_xy=PARTICLE_A_START_PX, max_dist_px=5.0):
    """Track whose frame-0 spot is closest to start_xy."""
    first = tracks[tracks.FRAME == 0]
    d = np.hypot(first.X_px - start_xy[0], first.Y_px - start_xy[1])
    if d.min() > max_dist_px:
        raise RuntimeError(f'no track starts within {max_dist_px} px of {start_xy}')
    tid = int(first.TRACK_ID.iloc[int(np.argmin(d.to_numpy()))])
    return tid, tracks[tracks.TRACK_ID == tid].reset_index(drop=True)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def run_dcmss(pos, tag, prep):
    out = os.path.join(RES, 'dcmss')
    os.makedirs(out, exist_ok=True)
    rows, res = DA.analyse(pos)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f'{tag}_segments.csv'), index=False, float_format='%.17g')
    stages = dict(config=DA.CONFIG.as_dict(), preprocessing=prep,
                  step1=dict(n_initial_peaks=res['step1']['n_initial_peaks'],
                             n_ks_merged=res['step1']['n_ks_merged'],
                             bounds=res['bounds']),
                  initial_segments=res['segs_initial'], final_segments=res['segs'])
    with open(os.path.join(out, f'{tag}_stages.json'), 'w', encoding='utf-8') as fh:
        json.dump(_clean(stages), fh, indent=2, allow_nan=False)
    DA.plot(pos, rows, os.path.join(out, f'{tag}_dcmss.png'), title=tag)
    print(f'\nDC-MSS [{tag}]')
    print(df[['segment', 'start', 'end_exclusive', 'n_positions', 'motion_class',
              'mss_slope', 'thr_free_directed', 'Cu_lag1']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trackmate', action='store_true',
                    help='also analyse data/particleA_trackmate.csv')
    a = ap.parse_args(argv)

    print('== Linking ==')
    _, tracks = run_linking(os.path.join(RES, 'linking'))
    tid, part = select_particle(tracks)
    part.to_csv(os.path.join(RES, 'particleA_track.csv'), index=False)
    print(f'particle A = linked track {tid}: {len(part)} spots, '
          f'frames {part.FRAME.min()}-{part.FRAME.max()}, '
          f'mean link confidence {part.LINK_CONFIDENCE.mean():.3f}')

    data, prep = load_tracks(os.path.join(RES, 'particleA_track.csv'), 'linked')
    results = {'linked': run_dcmss(data[tid]['pos'], 'particleA', prep[tid])}

    if a.trackmate:
        data, prep = load_tracks(os.path.join(ROOT, 'data', 'particleA_trackmate.csv'),
                                 'trackmate')
        (k,) = data
        results['trackmate'] = run_dcmss(data[k]['pos'], 'particleA_trackmate', prep[k])
    return results


if __name__ == '__main__':
    main()
