"""Load linked trajectories into the uniform-time-step form that DC-MSS needs.

Input is a track table written by ``src/linking.py`` (``results/linking/tracks.csv``,
or the single-track subset ``results/particleA_track.csv`` written by
``run_pipeline.py``); only the columns TRACK_ID, FRAME, X_px and Y_px are used.

Frames skipped by gap closing are filled by linear interpolation, because DC-MSS
assumes a uniform time series.  Exactly what was changed is returned in ``report``.

Units: length = pixel, time = frame (1 frame = 1 min).
"""
import numpy as np
import pandas as pd

from units import DT_MIN


def _regularise(g):
    """One track -> uniformly sampled positions (linear interpolation over gaps)."""
    g = g.sort_values('FRAME')
    if g['FRAME'].duplicated().any():
        raise ValueError('a track has two spots in the same frame')
    f = g['FRAME'].to_numpy(int)
    full = np.arange(f.min(), f.max() + 1)
    x = np.interp(full, f, g['X_px'].to_numpy())
    y = np.interp(full, f, g['Y_px'].to_numpy())
    interpolated = ~np.isin(full, f)
    track = dict(frame=full, t_min=full * DT_MIN, pos=np.column_stack([x, y]),
                 interpolated=interpolated)
    report = dict(n_raw=len(g), n_frames=len(full),
                  n_interpolated=int(interpolated.sum()),
                  frame_min=int(f.min()), frame_max=int(f.max()))
    return track, report


def load_tracks(path):
    """Track table from src/linking.py -> ({track_id: track}, {track_id: report})."""
    df = pd.read_csv(path)
    tracks, report = {}, {}
    for tid, g in df.groupby('TRACK_ID'):
        tracks[int(tid)], report[int(tid)] = _regularise(g)
    return tracks, report
