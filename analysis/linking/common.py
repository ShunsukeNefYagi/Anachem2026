"""
common.py — 共通定数とリンキングアルゴリズム

priority-greedy + ギャップ拡大リンキング。座標だけを差し替えられるように関数化してあり、
s4_track.py が raw / vib / full の 3 座標系に対して同じアルゴリズムを適用する。
アルゴリズムとパラメータは本研究の初期解析（TrackMate 検出 + 自作リンカ）と同一。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))     # repository root
RAW  = os.path.join(ROOT, 'data', 'linking')      # inputs (XML, per-spot intensity)
OUT  = os.path.join(ROOT, 'results', 'linking')   # all generated files
os.makedirs(OUT, exist_ok=True)

# ── スケール ────────────────────────────────────────────────────────────────
PIXEL_SIZE = 10.0 / 222.0     # µm/px  (222 px = 10 µm)
DT_MIN     = 1.0              # min/frame
N_FRAMES   = 720
IMG_SIZE   = 1431

# ── リンキングの名目パラメータ（過去解析と同一）──────────────────────────────
MAX_DIST_UM         = 3.0
MAX_INTENSITY_RATIO = 2.0
DIST_WEIGHT         = 0.7
INT_WEIGHT          = 0.3
MAX_GAP             = 3
GAP_RELAX           = 0.3
MIN_DETECTIONS      = 3


def load_spots(coord_cols=('X_px', 'Y_px')):
    """spots_raw.csv + spot_intensity_square_roi.csv を結合して返す。

    coord_cols で使う座標列を選ぶ（生: X_px/Y_px, 補正後: X_px_corr/Y_px_corr）。
    """
    df = pd.read_csv(os.path.join(OUT, 'spots_raw.csv'))
    df['FRAME'] = df['FRAME'].astype(int)
    # s2_intensity.py needs the 2.9 GB image stack, which is not distributed;
    # fall back to the precomputed copy shipped in data/linking/.
    ipath = os.path.join(OUT, 'spot_intensity_square_roi.csv')
    if not os.path.exists(ipath):
        ipath = os.path.join(RAW, 'spot_intensity_square_roi.csv')
    it = pd.read_csv(ipath)
    df = df.merge(it[['spot_id', 'bg_corrected_intensity', 'n_pixels']],
                  left_on='ID', right_on='spot_id', how='left').drop(columns=['spot_id'])
    df = df.rename(columns={'bg_corrected_intensity': 'INTENSITY'})
    return df.sort_values(['FRAME', 'ID']).reset_index(drop=True)


def track(df, max_dist=MAX_DIST_UM, max_gap=MAX_GAP,
          dist_w=DIST_WEIGHT, int_w=INT_WEIGHT,
          max_int_ratio=MAX_INTENSITY_RATIO,
          record_confidence=False):
    """priority-greedy + ギャップ拡大リンキング。

    df は 'ID', 'FRAME', 'POSITION_X', 'POSITION_Y'[µm], 'INTENSITY' を持つこと。
    返り値: {track_id: {'spots': [(frame, spot_id), ...], ...}}
    """
    frames = sorted(df['FRAME'].unique())
    sbf = {f: g for f, g in df.groupby('FRAME')}

    next_tid  = 0
    active    = {}
    finished  = {}
    link_info = {}

    def new_track(fr, sid, x, y, intensity):
        nonlocal next_tid
        active[next_tid] = dict(spots=[(fr, sid)], last_frame=fr,
                                last_x=x, last_y=y, last_int=intensity, gap=0)
        next_tid += 1

    for fi, frame in enumerate(frames):
        curr  = sbf[frame]
        c_ids = curr['ID'].values
        c_xy  = curr[['POSITION_X', 'POSITION_Y']].values
        c_int = curr['INTENSITY'].values

        if fi == 0:
            for j in range(len(c_ids)):
                new_track(frame, c_ids[j], c_xy[j, 0], c_xy[j, 1], c_int[j])
            continue

        cand_ids = sorted(active.keys(),
                          key=lambda t: len(active[t]['spots']), reverse=True)
        assigned = set()

        for tid in cand_ids:
            t  = active[tid]
            sr = max_dist * (1.0 + t['gap'] * GAP_RELAX)
            costs = []
            for j in range(len(c_ids)):
                if j in assigned:
                    continue
                d = np.hypot(c_xy[j, 0] - t['last_x'], c_xy[j, 1] - t['last_y'])
                if d > sr:
                    continue
                a, b  = c_int[j], t['last_int']
                ratio = max(a, b) / (min(a, b) + 1e-9)
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
            if record_confidence:
                link_info[(frame, int(c_ids[best_j]))] = (best_cost, second_cost)

        for j in range(len(c_ids)):
            if j not in assigned:
                new_track(frame, c_ids[j], c_xy[j, 0], c_xy[j, 1], c_int[j])

    for tid in list(active):
        finished[tid] = active.pop(tid)

    return (finished, link_info) if record_confidence else finished


def good_tracks(finished, min_det=MIN_DETECTIONS):
    return {tid: t for tid, t in finished.items() if len(t['spots']) >= min_det}


def tracks_to_frame(finished, df, min_det=MIN_DETECTIONS):
    """finished dict → スポット単位の DataFrame（TRACK_ID, FRAME, 座標, 輝度）"""
    lk = df.set_index('ID')
    rows = []
    for tid, t in good_tracks(finished, min_det).items():
        ss = sorted(t['spots'], key=lambda x: x[0])
        span = ss[-1][0] - ss[0][0] + 1
        for fr, sid in ss:
            r = lk.loc[int(sid)]
            rows.append(dict(TRACK_ID=tid, DETECTIONS=len(ss), SPAN=span,
                             SPOT_ID=int(sid), FRAME=fr,
                             X_um=r['POSITION_X'], Y_um=r['POSITION_Y'],
                             INTENSITY=r['INTENSITY']))
    return pd.DataFrame(rows).sort_values(['TRACK_ID', 'FRAME']).reset_index(drop=True)


def slide_style():
    import matplotlib
    matplotlib.rcParams.update({
        'figure.facecolor': 'white', 'savefig.dpi': 150,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 9, 'axes.titlesize': 10,
    })
