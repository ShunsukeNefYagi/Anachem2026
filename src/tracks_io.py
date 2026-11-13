"""Load trajectories into the uniform-time-step form that DC-MSS needs.

Two input formats are supported:

* ``trackmate`` - a TrackMate "Spots in tracks" CSV export (e.g. ``data/two.csv``).
  Row 1 is the header, rows 2-4 are TrackMate's label / short-name / unit rows.
* ``linked``    - the output of ``analysis/linking/s4_track.py``
  (``results/linking/tracks_<variant>.csv``; columns TRACK_ID, FRAME, X_px, Y_px).

In both cases a frame that holds two spots keeps the higher-QUALITY spot (only the
TrackMate export has QUALITY; the linker never assigns two spots to one frame), and
missing frames are filled by linear interpolation, because DC-MSS assumes a uniform
time series.  Exactly what was changed is returned in ``report``.

Units: length = pixel, time = frame (1 frame = 1 min).
"""
import numpy as np
import pandas as pd

DT_MIN = 1.0            # 1 frame = 1 min


def _regularise(g, xcol, ycol, extra=()):
    """One track -> uniformly sampled positions (dedup + linear interpolation)."""
    n_raw = len(g)
    if 'QUALITY' in g:
        # duplicated frames: keep the spot with the highest QUALITY
        g = g.sort_values(['FRAME', 'QUALITY'])
        g = g.drop_duplicates('FRAME', keep='last').sort_values('FRAME')
    else:
        g = g.sort_values('FRAME')
        if g['FRAME'].duplicated().any():
            raise ValueError('duplicated frames in a track without QUALITY column')
    n_dedup = len(g)

    f = g['FRAME'].to_numpy(int)
    full = np.arange(f.min(), f.max() + 1)
    x = np.interp(full, f, g[xcol].to_numpy())
    y = np.interp(full, f, g[ycol].to_numpy())
    interpolated = ~np.isin(full, f)
    track = dict(frame=full, t_min=full * DT_MIN, pos=np.column_stack([x, y]),
                 interpolated=interpolated)
    for name, col in extra:
        track[name] = np.interp(full, f, g[col].to_numpy())
    report = dict(n_raw=n_raw, n_duplicates_removed=n_raw - n_dedup,
                  n_frames=len(full), n_interpolated=int(interpolated.sum()),
                  frame_min=int(f.min()), frame_max=int(f.max()))
    return track, report


def load_trackmate_csv(path):
    """TrackMate spots CSV -> ({track_id: track}, {track_id: report})."""
    df = pd.read_csv(path, skiprows=[1, 2, 3])
    tracks, report = {}, {}
    extra = [('quality', 'QUALITY'), ('intensity', 'MEAN_INTENSITY_CH1'),
             ('snr', 'SNR_CH1')]
    for tid, g in df.groupby('TRACK_ID'):
        tracks[int(tid)], report[int(tid)] = _regularise(
            g, 'POSITION_X', 'POSITION_Y', extra)
    return tracks, report


def load_linked_csv(path):
    """Output of s4_track.py -> ({track_id: track}, {track_id: report})."""
    df = pd.read_csv(path)
    tracks, report = {}, {}
    for tid, g in df.groupby('TRACK_ID'):
        tracks[int(tid)], report[int(tid)] = _regularise(g, 'X_px', 'Y_px')
    return tracks, report


def load_tracks(path, fmt='auto'):
    """Dispatch on ``fmt`` ('trackmate' | 'linked' | 'auto')."""
    if fmt == 'auto':
        with open(path, encoding='utf-8') as fh:
            header = fh.readline()
        fmt = 'trackmate' if 'POSITION_X' in header else 'linked'
    if fmt == 'trackmate':
        return load_trackmate_csv(path)
    if fmt == 'linked':
        return load_linked_csv(path)
    raise ValueError(f'unknown format: {fmt}')
