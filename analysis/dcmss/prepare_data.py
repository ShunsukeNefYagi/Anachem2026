"""two.csv を読み込み、重複除去・欠損内挿を行って解析用の等間隔軌跡を作る。"""
import numpy as np, pandas as pd, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))     # repository root
DATA = os.path.join(ROOT, 'data')
DT_MIN = 1.0            # 1 frame = 1 min


def load_tracks(csv=os.path.join(DATA, 'two.csv')):
    df = pd.read_csv(csv, skiprows=[1, 2, 3])
    tracks, report = {}, {}
    for tid, g in df.groupby('TRACK_ID'):
        g = g.sort_values(['FRAME', 'QUALITY'])
        n_raw = len(g)
        # 重複フレームは QUALITY 最大のスポットを採用
        g = g.drop_duplicates('FRAME', keep='last').sort_values('FRAME')
        n_dedup = len(g)

        f = g['FRAME'].to_numpy(int)
        full = np.arange(f.min(), f.max() + 1)
        x = np.interp(full, f, g['POSITION_X'].to_numpy())
        y = np.interp(full, f, g['POSITION_Y'].to_numpy())
        interpolated = ~np.isin(full, f)

        tracks[int(tid)] = dict(
            frame=full, t_min=full * DT_MIN,
            pos=np.column_stack([x, y]),
            interpolated=interpolated,
            quality=np.interp(full, f, g['QUALITY'].to_numpy()),
            intensity=np.interp(full, f, g['MEAN_INTENSITY_CH1'].to_numpy()),
            snr=np.interp(full, f, g['SNR_CH1'].to_numpy()),
        )
        report[int(tid)] = dict(n_raw=n_raw, n_duplicates_removed=n_raw - n_dedup,
                                n_frames=len(full),
                                n_interpolated=int(interpolated.sum()),
                                frame_min=int(f.min()), frame_max=int(f.max()))
    return tracks, report


if __name__ == '__main__':
    tr, rep = load_tracks()
    print(json.dumps(rep, indent=2))
    for k, v in tr.items():
        p = v['pos']
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        print(f'TRACK {k}: N={len(p)}  step median={np.median(step):.3f} '
              f'mean={step.mean():.3f} max={step.max():.3f} px/frame; '
              f'net displacement={np.linalg.norm(p[-1]-p[0]):.1f} px; '
              f'path length={step.sum():.1f} px')
