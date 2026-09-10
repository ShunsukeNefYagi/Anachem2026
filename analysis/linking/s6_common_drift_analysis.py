"""
s6_common_drift_analysis.py — 全粒子共通変位を全帯域で除去した追加解析。

`s3e_decompose.py` の full 補正を common として明示的に評価する。補正量を
推定した raw リンク対を固定したまま、補正前後のフレーム間中央値を比較するため、
再 linking による対応粒子の変化を「改善」と誤認しない。

入力:
  out/drift_components.csv
  out/tracks_raw.csv
  out/tracks_common.csv
  out/track_summary_raw.csv
  out/track_summary_common.csv
  out/validate.json

出力:
  out/common_drift_per_frame.csv
  out/common_drift_summary.json
  out/common_drift_long_tracks.csv
  out/fig_common_drift_analysis.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
C.slide_style()

MIN_DET_FOR_DRIFT = 10
T = C.N_FRAMES

dcomp = pd.read_csv(os.path.join(C.OUT, 'drift_components.csv'))
raw = pd.read_csv(os.path.join(C.OUT, 'tracks_raw.csv'))
common = pd.read_csv(os.path.join(C.OUT, 'tracks_common.csv'))
sm_raw = pd.read_csv(os.path.join(C.OUT, 'track_summary_raw.csv'))
sm_common = pd.read_csv(os.path.join(C.OUT, 'track_summary_common.csv'))
with open(os.path.join(C.OUT, 'validate.json'), encoding='utf-8') as fh:
    validation = json.load(fh)

# full/common 補正は frame 0 を原点に取り直す。絶対座標の任意定数は解析に
# 影響しないが、CSV と図で「frame 0 の補正量 = 0」を保証すると解釈しやすい。
D = dcomp[['Dx_full_px', 'Dy_full_px']].to_numpy(float)
D = D - D[0]
dD = np.vstack([np.zeros(2), np.diff(D, axis=0)])

# 全9,663検出の補正座標も、追跡結果とは別に保存する。
spots = pd.read_csv(os.path.join(C.OUT, 'spots_variants.csv'))
sf = spots.FRAME.to_numpy(int)
spots_common = spots[['ID', 'FRAME', 'X_px', 'Y_px', 'INTENSITY',
                      'QUALITY', 'SNR_CH1', 'n_pixels']].copy()
spots_common['X_px_common'] = spots.X_px.to_numpy() - D[sf, 0]
spots_common['Y_px_common'] = spots.Y_px.to_numpy() - D[sf, 1]
spots_common['X_um_common'] = spots_common.X_px_common * C.PIXEL_SIZE
spots_common['Y_um_common'] = spots_common.Y_px_common * C.PIXEL_SIZE
spots_common.to_csv(os.path.join(C.OUT, 'spots_common_drift.csv'), index=False)

# drift 推定に使った条件と同じく、検出数 >= 10 の raw トラックから連続
# フレームの変位を集める。補正後も同じリンク対を使う。
use = raw[raw.DETECTIONS >= MIN_DET_FOR_DRIFT]
steps = {t: [] for t in range(T)}
for _, g in use.groupby('TRACK_ID'):
    g = g.sort_values('FRAME')
    fr = g.FRAME.to_numpy(int)
    xy = g[['X_px', 'Y_px']].to_numpy(float)
    for k in np.flatnonzero(np.diff(fr) == 1):
        steps[int(fr[k + 1])].append(xy[k + 1] - xy[k])

rows = []
for t in range(T):
    arr = np.asarray(steps[t], dtype=float)
    if len(arr):
        med_raw = np.median(arr, axis=0)
        med_corr = np.median(arr - dD[t], axis=0)
        mad_raw = np.median(np.abs(arr - med_raw), axis=0)
    else:
        med_raw = med_corr = mad_raw = np.array([np.nan, np.nan])
    rows.append(dict(
        frame=t,
        n_particles=len(arr),
        dx_median_raw_px=med_raw[0], dy_median_raw_px=med_raw[1],
        dx_common_removed_px=dD[t, 0], dy_common_removed_px=dD[t, 1],
        dx_median_residual_px=med_corr[0], dy_median_residual_px=med_corr[1],
        mad_x_raw_px=mad_raw[0], mad_y_raw_px=mad_raw[1],
        Dx_common_px=D[t, 0], Dy_common_px=D[t, 1],
        Dx_common_um=D[t, 0] * C.PIXEL_SIZE,
        Dy_common_um=D[t, 1] * C.PIXEL_SIZE,
    ))
per = pd.DataFrame(rows)
per.to_csv(os.path.join(C.OUT, 'common_drift_per_frame.csv'), index=False)

valid = per.n_particles >= 4
vr = per.loc[valid, ['dx_median_raw_px', 'dy_median_raw_px']].to_numpy()
vc = per.loc[valid, ['dx_median_residual_px', 'dy_median_residual_px']].to_numpy()

def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))

def pooled_step_stats(df):
    vals = []
    for _, g in df.groupby('TRACK_ID'):
        g = g.sort_values('FRAME')
        consecutive = np.diff(g.FRAME.to_numpy()) == 1
        dx = np.diff(g.X_px.to_numpy())[consecutive]
        dy = np.diff(g.Y_px.to_numpy())[consecutive]
        vals.append(np.column_stack([dx, dy]))
    a = np.vstack(vals)
    r = np.hypot(a[:, 0], a[:, 1])
    return dict(n_steps=int(len(a)), step_median_um=float(np.median(r) * C.PIXEL_SIZE),
                step_mean_um=float(np.mean(r) * C.PIXEL_SIZE),
                D_px2_per_min=float(np.mean(np.sum(a * a, axis=1)) / 4))

raw_stats = pooled_step_stats(raw)
common_stats = pooled_step_stats(common)

# 長寿命トラックは ID が共通するものだけを同じ ID で比較する。
long_ids = sorted(set(sm_raw.loc[sm_raw.DETECTIONS >= T // 2, 'TRACK_ID'].astype(int)) &
                  set(sm_common.TRACK_ID.astype(int)))
long_rows = []
for tid in long_ids:
    rec = {'TRACK_ID': int(tid)}
    for label, data in [('raw', raw), ('common', common)]:
        g = data[data.TRACK_ID == tid].sort_values('FRAME')
        if g.empty:
            continue
        cons = np.diff(g.FRAME.to_numpy()) == 1
        dx = np.diff(g.X_px.to_numpy())[cons]
        dy = np.diff(g.Y_px.to_numpy())[cons]
        rec[f'n_{label}'] = int(len(g))
        rec[f'D_{label}_px2_per_min'] = float(np.mean(dx * dx + dy * dy) / 4)
        rec[f'net_{label}_um'] = float(np.hypot(
            g.X_px.iloc[-1] - g.X_px.iloc[0],
            g.Y_px.iloc[-1] - g.Y_px.iloc[0]) * C.PIXEL_SIZE)
    rec['D_common/raw'] = rec['D_common_px2_per_min'] / rec['D_raw_px2_per_min']
    long_rows.append(rec)
long = pd.DataFrame(long_rows)
long.to_csv(os.path.join(C.OUT, 'common_drift_long_tracks.csv'), index=False)

cv = validation['cross_validation']
summary = dict(
    method=('consecutive-frame displacement median across tracks, split-half null, '
            'frequency-domain Wiener shrinkage; all significant frequency bands removed'),
    interpretation=('ensemble-relative coordinates; common sample motion is removed together '
                    'with instrumental drift'),
    n_frames=int(T),
    n_tracks_used_for_estimation=int(use.TRACK_ID.nunique()),
    particles_per_frame=dict(min=int(per.loc[valid, 'n_particles'].min()),
                             mean=float(per.loc[valid, 'n_particles'].mean()),
                             max=int(per.loc[valid, 'n_particles'].max())),
    cumulative_common_displacement=dict(
        x_px=float(D[-1, 0]), y_px=float(D[-1, 1]),
        x_um=float(D[-1, 0] * C.PIXEL_SIZE),
        y_um=float(D[-1, 1] * C.PIXEL_SIZE),
        magnitude_um=float(np.hypot(D[-1, 0], D[-1, 1]) * C.PIXEL_SIZE)),
    same_link_pair_residual_check=dict(
        raw_median_step_rms_px=dict(x=rms(vr[:, 0]), y=rms(vr[:, 1])),
        residual_median_step_rms_px=dict(x=rms(vc[:, 0]), y=rms(vc[:, 1])),
        rms_ratio=dict(x=rms(vc[:, 0]) / rms(vr[:, 0]),
                       y=rms(vc[:, 1]) / rms(vr[:, 1])),
        residual_cumulative_px=dict(x=float(np.nansum(vc[:, 0])),
                                    y=float(np.nansum(vc[:, 1])))),
    tracking=dict(raw=raw_stats, common=common_stats,
                  D_ratio=float(common_stats['D_px2_per_min'] / raw_stats['D_px2_per_min'])),
    independent_split_half_cross_validation={
        lag: dict(msd_ratio=float(cv[lag]['ratio_mean']),
                  sd=float(cv[lag]['ratio_sd'])) for lag in cv},
)
with open(os.path.join(C.OUT, 'common_drift_summary.json'), 'w', encoding='utf-8') as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

# 図: 累積共通変位、同一リンク対での残差、全体ステップ、CV。
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
t = per.frame.to_numpy()
a = ax[0, 0]
a.plot(t, per.Dx_common_um, label='x', lw=1.1)
a.plot(t, per.Dy_common_um, label='y', lw=1.1)
a.set_xlabel('frame (min)'); a.set_ylabel('cumulative common displacement (µm)')
a.set_title('A. Removed common displacement'); a.legend()

a = ax[0, 1]
a.plot(t[valid], per.loc[valid, 'dx_median_raw_px'], lw=0.6, alpha=0.65,
       color='0.55', label='raw x median')
a.plot(t[valid], per.loc[valid, 'dx_median_residual_px'], lw=0.7,
       color='tab:blue', label='residual x median')
a.axhline(0, color='k', lw=0.6)
a.set_xlabel('frame (min)'); a.set_ylabel('x displacement (px/frame)')
a.set_title('B. Same-link-pair residual check'); a.legend(fontsize=8)

a = ax[1, 0]
labels = ['common RMS x', 'common RMS y', 'step median', 'D']
ratios = [rms(vc[:, 0]) / rms(vr[:, 0]), rms(vc[:, 1]) / rms(vr[:, 1]),
          common_stats['step_median_um'] / raw_stats['step_median_um'],
          common_stats['D_px2_per_min'] / raw_stats['D_px2_per_min']]
x = np.arange(len(labels))
a.bar(x, ratios, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
a.axhline(1, color='k', ls='--', lw=0.8, label='raw = 1')
a.set_xticks(x); a.set_xticklabels(labels, rotation=12)
a.set_ylim(0, 1.12); a.set_ylabel('common-corrected / raw')
a.set_title('C. Reduction after correction'); a.legend(fontsize=8)

a = ax[1, 1]
lags = np.array([int(k) for k in cv])
means = np.array([cv[str(k)]['ratio_mean'] for k in lags])
sds = np.array([cv[str(k)]['ratio_sd'] for k in lags])
a.errorbar(lags, means, yerr=sds, marker='o', capsize=3, color='tab:blue')
a.axhline(1, color='k', ls='--', lw=0.8)
a.set_xscale('log'); a.set_xlabel('lag (min)'); a.set_ylabel('MSD corrected / raw')
a.set_title('D. Split-half cross-validation')

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_common_drift_analysis.png'), dpi=180)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print('\nSaved out/spots_common_drift.csv, out/common_drift_per_frame.csv,')
print('      out/common_drift_summary.json, out/common_drift_long_tracks.csv,')
print('      out/fig_common_drift_analysis.png')
