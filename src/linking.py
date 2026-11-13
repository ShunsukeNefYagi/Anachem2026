"""Link per-frame detections into trajectories with a displacement + intensity cost.

Input
    data/laser1traj.xml                 TrackMate v8.1.5 detection-only export
                                        (LoG detector; coordinates in pixels)
    data/spot_intensity_square_roi.csv  background-corrected integrated intensity of
                                        each spot (made by spot_intensity.py from the
                                        raw image stack, which is not distributed)

Cost of linking the last spot of a track to a candidate spot in the next frame
    C = w_d * d / d_max + w_i * (r - 1) / (R_max - 1)
    d = displacement [um], r = max(I1, I2) / min(I1, I2) (intensity ratio)
Candidates with d > d_max * (1 + gamma * gap) or r > R_max are not allowed, nor are
links involving a non-positive intensity (r undefined; none occur in the shipped data).
Tracks are served in order of their current number of detections (longer tracks
first), each takes its lowest-cost candidate (greedy).  A track may skip up to
G_max frames; unmatched spots start new tracks.  Tracks with fewer than
MIN_DETECTIONS spots are discarded.
Per link, a confidence score 1 - C_best / C_second is recorded (1 if there was only
one candidate).

Output (results/linking/)
    spots.csv     parsed detections with intensity
    tracks.csv    one row per linked spot
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from units import PIXEL_SIZE   # um / px

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, 'data', 'laser1traj.xml')
INTENSITY_CSV = os.path.join(ROOT, 'data', 'spot_intensity_square_roi.csv')
OUT = os.path.join(ROOT, 'results', 'linking')

# linking parameters
MAX_DIST_UM = 3.0             # d_max
MAX_INTENSITY_RATIO = 2.0     # R_max
DIST_WEIGHT = 0.7             # w_d
INT_WEIGHT = 0.3              # w_i
MAX_GAP = 3                   # G_max (frames)
GAP_RELAX = 0.3               # gamma
MIN_DETECTIONS = 3

SPOT_FIELDS = ['POSITION_X', 'POSITION_Y', 'QUALITY', 'RADIUS',
               'MEAN_INTENSITY_CH1', 'MEDIAN_INTENSITY_CH1', 'MAX_INTENSITY_CH1',
               'MIN_INTENSITY_CH1', 'TOTAL_INTENSITY_CH1', 'STD_INTENSITY_CH1',
               'CONTRAST_CH1', 'SNR_CH1']


# ---------------------------------------------------------------------------
def parse_trackmate_xml(path=XML):
    """TrackMate XML -> DataFrame of spots (X_px, Y_px in pixels).

    The export has spatialunits=" " and pixelwidth="1.0", i.e. POSITION_X/Y are
    pixels.
    """
    rows = []
    for _, el in ET.iterparse(path, events=('end',)):
        if el.tag == 'Spot':
            r = {'ID': int(el.get('ID')), 'FRAME': int(el.get('FRAME'))}
            for k in SPOT_FIELDS:
                v = el.get(k)
                r[k] = float(v) if v is not None else float('nan')
            rows.append(r)
            el.clear()
    df = pd.DataFrame(rows).rename(columns={'POSITION_X': 'X_px', 'POSITION_Y': 'Y_px'})
    return df.sort_values(['FRAME', 'ID']).reset_index(drop=True)


def add_intensity(spots, path=INTENSITY_CSV):
    it = pd.read_csv(path)
    df = spots.merge(it[['spot_id', 'bg_corrected_intensity', 'n_pixels']],
                     left_on='ID', right_on='spot_id', how='left').drop(columns=['spot_id'])
    df = df.rename(columns={'bg_corrected_intensity': 'INTENSITY'})
    df['POSITION_X'] = df.X_px * PIXEL_SIZE      # um
    df['POSITION_Y'] = df.Y_px * PIXEL_SIZE
    return df.sort_values(['FRAME', 'ID']).reset_index(drop=True)


# ---------------------------------------------------------------------------
def track(df, max_dist=MAX_DIST_UM, max_gap=MAX_GAP,
          dist_w=DIST_WEIGHT, int_w=INT_WEIGHT,
          max_int_ratio=MAX_INTENSITY_RATIO):
    """Priority-greedy linking with gap closing.

    df needs 'ID', 'FRAME', 'POSITION_X', 'POSITION_Y' [um], 'INTENSITY'.
    Returns (tracks, link_info):
        tracks    {track_id: {'spots': [(frame, spot_id), ...], ...}}
        link_info {(frame, spot_id): (best_cost, second_cost)}
    """
    frames = sorted(df['FRAME'].unique())
    sbf = {f: g for f, g in df.groupby('FRAME')}

    next_tid = 0
    active = {}
    finished = {}
    link_info = {}

    def new_track(fr, sid, x, y, intensity):
        nonlocal next_tid
        active[next_tid] = dict(spots=[(fr, sid)], last_frame=fr,
                                last_x=x, last_y=y, last_int=intensity, gap=0)
        next_tid += 1

    for fi, frame in enumerate(frames):
        curr = sbf[frame]
        c_ids = curr['ID'].values
        c_xy = curr[['POSITION_X', 'POSITION_Y']].values
        c_int = curr['INTENSITY'].values

        if fi == 0:
            for j in range(len(c_ids)):
                new_track(frame, c_ids[j], c_xy[j, 0], c_xy[j, 1], c_int[j])
            continue

        # longer tracks choose first, so that short tracks cannot steal spots
        cand_ids = sorted(active.keys(),
                          key=lambda t: len(active[t]['spots']), reverse=True)
        assigned = set()

        for tid in cand_ids:
            t = active[tid]
            sr = max_dist * (1.0 + t['gap'] * GAP_RELAX)
            costs = []
            for j in range(len(c_ids)):
                if j in assigned:
                    continue
                d = np.hypot(c_xy[j, 0] - t['last_x'], c_xy[j, 1] - t['last_y'])
                if d > sr:
                    continue
                a, b = c_int[j], t['last_int']
                if min(a, b) <= 0:     # ratio undefined for non-positive intensity
                    continue
                ratio = max(a, b) / min(a, b)
                if ratio > max_int_ratio:
                    continue
                int_term = 0.0 if np.isinf(max_int_ratio) else \
                    int_w * (ratio - 1.0) / (max_int_ratio - 1.0)
                costs.append((dist_w * d / max_dist + int_term, j))

            if not costs:
                t['gap'] += 1
                if t['gap'] > max_gap:
                    finished[tid] = active.pop(tid)
                continue

            costs.sort()
            best_cost, best_j = costs[0]
            second_cost = costs[1][0] if len(costs) > 1 else float('inf')

            active[tid]['spots'].append((frame, c_ids[best_j]))
            active[tid].update(last_frame=frame,
                               last_x=c_xy[best_j, 0], last_y=c_xy[best_j, 1],
                               last_int=c_int[best_j], gap=0)
            assigned.add(best_j)
            link_info[(frame, int(c_ids[best_j]))] = (best_cost, second_cost)

        for j in range(len(c_ids)):
            if j not in assigned:
                new_track(frame, c_ids[j], c_xy[j, 0], c_xy[j, 1], c_int[j])

    for tid in list(active):
        finished[tid] = active.pop(tid)
    return finished, link_info


def tracks_table(finished, link_info, df, min_det=MIN_DETECTIONS):
    """Linked spots as a DataFrame (one row per spot of tracks with >= min_det)."""
    lk = df.set_index('ID')
    rows = []
    for tid, t in finished.items():
        if len(t['spots']) < min_det:
            continue
        ss = sorted(t['spots'], key=lambda x: x[0])
        span = ss[-1][0] - ss[0][0] + 1
        for frame, sid in ss:
            sid = int(sid)
            r = lk.loc[sid]
            key = (frame, sid)
            if key in link_info:
                bc, sc = link_info[key]
                conf = max(0.0, min(1.0, 1.0 - bc / sc)) if sc < float('inf') else 1.0
            else:                      # first spot of a track
                bc, conf = np.nan, np.nan
            rows.append(dict(TRACK_ID=tid, DETECTIONS=len(ss), SPAN=span,
                             SPOT_ID=sid, FRAME=frame,
                             X_um=r.POSITION_X, Y_um=r.POSITION_Y,
                             X_px=r.X_px, Y_px=r.Y_px, INTENSITY=r.INTENSITY,
                             LINK_COST=bc, LINK_CONFIDENCE=conf))
    return pd.DataFrame(rows).sort_values(['TRACK_ID', 'FRAME']).reset_index(drop=True)


def run_linking(out_dir=OUT, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    spots = add_intensity(parse_trackmate_xml())
    finished, link_info = track(spots)
    tracks = tracks_table(finished, link_info, spots)
    spots.to_csv(os.path.join(out_dir, 'spots.csv'), index=False)
    tracks.to_csv(os.path.join(out_dir, 'tracks.csv'), index=False)
    if verbose:
        n_full = tracks.groupby('TRACK_ID').SPAN.first().eq(spots.FRAME.nunique()).sum()
        print(f'spots {len(spots)} in {spots.FRAME.nunique()} frames; '
              f'tracks (>= {MIN_DETECTIONS} spots) {tracks.TRACK_ID.nunique()}, '
              f'spanning all frames {n_full}')
        print(f'-> {out_dir}')
    return spots, tracks


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run_linking()
