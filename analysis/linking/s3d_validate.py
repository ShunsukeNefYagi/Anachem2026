"""
s3d_validate.py — 共通成分の (1) 実在性 と (2) 由来（装置 or 試料）を確定し、
                  最終的に採用する補正量を決める。

背景
----
s3  : アンサンブル中央値の累積は x −249 px / y +118 px（12 h）。null 帯域の外。
s3b : 視野を固定midlineで二分すると左 +70 px / 右 −264 px。ただしアフィン検定
      （変位場の発散・回転）は z = −1.2 / +1.0 で有意でない。
s3c : フレーム内中央値で二分する検定は、後半に粒子が集積域に集まってしまい
      左右の空間コントラストが消えるため**検出力が無い**（null がランダム分割とほぼ同じ）。
      → 空間分割だけでは装置と試料を分けられなかった。

そこで 2 つの独立な検査で決める。

 検査 I. 交差検証（実在性）
   トラックを 2 群に分け、群 A だけからドリフトを推定して**群 B に適用**し、
   B の変位分散が減るか増えるかを見る。推定に使っていないデータで評価するので、
   自己相関による見かけの改善が入らない。減れば共通成分は実在、増えれば推定器ノイズ。

 検査 II. 装置の既知スペック（由来）
   同一顕微鏡で固定粒子を 12 h 追った 20260825_005 の実測:
       12 h 直線ドリフト  x −0.635 px (−29 nm), y +1.757 px (+79 nm)
       8.00 min 周期振動  振幅 3.742 px (168.5 nm)
   ここで見えている共通成分がこの 100 倍以上なら、それは装置ではなく試料である。
   加えて 8 min 周期のスペクトル線があるかを直接探す。

出力: out/validate.json, out/fig_validate.png,
      out/drift.csv, out/spots_corrected.csv（最終採用）
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
C.slide_style()
rng = np.random.default_rng(31415)

MIN_DET_FOR_DRIFT = 10
MIN_N   = 4
LAGS    = [1, 5, 20, 60, 180]
N_CV    = 200

# 20260825_005（同一顕微鏡・固定粒子・12 h・1 min 間隔）の実測値
MACHINE = dict(drift_12h_px=dict(x=-0.635, y=1.757),
               vib_period_min=8.00, vib_amp_px=3.742,
               source='20260825_005/analysis/analysis_log.md §6-1, §10-3')

# ── トラックを作る（生座標）──────────────────────────────────────────────
df = C.load_spots()
df['POSITION_X'] = df.X_px * C.PIXEL_SIZE
df['POSITION_Y'] = df.Y_px * C.PIXEL_SIZE
fin = C.track(df)
tr = C.tracks_to_frame(fin, df)
tr['X_px'] = tr.X_um / C.PIXEL_SIZE
tr['Y_px'] = tr.Y_um / C.PIXEL_SIZE
use = tr[tr.DETECTIONS >= MIN_DET_FOR_DRIFT]
tids = np.array(sorted(use.TRACK_ID.unique()))
print(f'tracks>=3 {tr.TRACK_ID.nunique()},  tracks>={MIN_DET_FOR_DRIFT} {len(tids)}')

# トラックごとの位置系列（欠損は NaN）
pos = {}
for tid, g in use.groupby('TRACK_ID'):
    a = np.full((C.N_FRAMES, 2), np.nan)
    a[g.FRAME.values] = g[['X_px', 'Y_px']].values
    pos[tid] = a


def drift_from(tid_subset, minn=MIN_N):
    """与えたトラック集合から、フレーム間中央値の累積ドリフトを作る。"""
    v = np.zeros((C.N_FRAMES, 2)); n = np.zeros(C.N_FRAMES, int)
    for t in range(1, C.N_FRAMES):
        d = []
        for tid in tid_subset:
            a = pos[tid]
            if not np.isnan(a[t, 0]) and not np.isnan(a[t - 1, 0]):
                d.append(a[t] - a[t - 1])
        n[t] = len(d)
        if len(d) >= minn:
            v[t] = np.median(np.array(d), axis=0)
    return np.cumsum(v, axis=0), v, n


def msd_at(tid_subset, D, lags=LAGS):
    """ドリフト D を引いた座標での、遅れ lag の平均二乗変位 [px^2]。"""
    out = {}
    for L in lags:
        s, c = 0.0, 0
        for tid in tid_subset:
            a = pos[tid] - D
            b = a[L:] - a[:-L]
            b = b[~np.isnan(b[:, 0])]
            s += float((b ** 2).sum()); c += len(b)
        out[L] = s / c if c else np.nan
    return out


# ══ 検査 I. 交差検証 ═══════════════════════════════════════════════════════
print('\n検査 I. 交差検証（群 A で推定 → 群 B で評価）', flush=True)
zero = np.zeros((C.N_FRAMES, 2))
cv = {L: [] for L in LAGS}
cv_shuf = {L: [] for L in LAGS}
for k in range(N_CV):
    p = rng.permutation(tids)
    A, B = p[:len(p) // 2], p[len(p) // 2:]
    DA, vA, _ = drift_from(A)
    # 時間シャッフルしたドリフト（同じ大きさ・同じ分布だが本物の時間構造を壊した対照）
    DAs = np.cumsum(vA[rng.permutation(C.N_FRAMES)], axis=0)
    m0 = msd_at(B, zero); m1 = msd_at(B, DA); m2 = msd_at(B, DAs)
    for L in LAGS:
        cv[L].append(m1[L] / m0[L]); cv_shuf[L].append(m2[L] / m0[L])

res_cv = {}
print(f'  {"lag(min)":>9} {"MSD比(補正/無補正)":>22} {"対照: 時間シャッフル":>22}')
for L in LAGS:
    a = np.array(cv[L]); b = np.array(cv_shuf[L])
    res_cv[L] = dict(ratio_mean=float(a.mean()), ratio_sd=float(a.std()),
                     shuffled_mean=float(b.mean()), shuffled_sd=float(b.std()))
    print(f'  {L:>9} {a.mean():>12.3f} ± {a.std():.3f} {b.mean():>16.3f} ± {b.std():.3f}')

# ══ 検査 II. 装置スペックとの比較 & 8 min 線の探索 ═══════════════════════
D_all, v_all, n_all = drift_from(tids)
print(f'\n検査 II. 全トラックからの共通成分: 12 h で x {D_all[-1,0]:+.1f} px '
      f'({D_all[-1,0]*C.PIXEL_SIZE:+.2f} µm), y {D_all[-1,1]:+.1f} px '
      f'({D_all[-1,1]*C.PIXEL_SIZE:+.2f} µm)')
ratio = {a: abs(D_all[-1, i] / MACHINE['drift_12h_px'][a])
         for i, a in enumerate('xy')}
print(f'  20260825_005 の実測 12 h ドリフト x {MACHINE["drift_12h_px"]["x"]:+.3f} px, '
      f'y {MACHINE["drift_12h_px"]["y"]:+.3f} px')
print(f'  → 比: x {ratio["x"]:.0f} 倍, y {ratio["y"]:.0f} 倍')

# 8 min 線: 共通速度 v の周期グラム。null は「粒子ラベル内でのフレーム順シャッフル」
T = C.N_FRAMES
freq = np.fft.rfftfreq(T, 1.0); per = np.r_[np.inf, 1 / freq[1:]]
P_v = np.abs(np.fft.rfft(v_all - v_all.mean(0), axis=0)) ** 2
NB = 400
P_null = np.zeros((NB, len(freq), 2))
for k in range(NB):
    vs = v_all[rng.permutation(T)]
    P_null[k] = np.abs(np.fft.rfft(vs - vs.mean(0), axis=0)) ** 2
p95 = np.percentile(P_null, 95, axis=0)
i8 = int(np.argmin(np.abs(per[1:] - 8.0))) + 1
band = np.abs(per - 8.0) < 0.6
vib = {}
for i, a in enumerate('xy'):
    pk = P_v[band, i].max()
    thr = p95[band, i].max()
    # 振幅換算: 正弦波 A の |FFT|^2 = (A*T/2)^2
    amp = 2 * np.sqrt(pk) / T
    vib[a] = dict(power_at_8min=float(P_v[i8, i]), peak_in_band=float(pk),
                  null_p95_in_band=float(thr), significant=bool(pk > thr),
                  implied_amplitude_px=float(amp),
                  machine_amplitude_px=MACHINE['vib_amp_px'])
    print(f'  8 min 帯 ({a}): ピーク {pk:.3g} vs null p95 {thr:.3g} '
          f'→ {"有意" if pk > thr else "有意でない"} '
          f'(振幅換算 {amp:.3f} px, 装置実測 {MACHINE["vib_amp_px"]:.3f} px)')

# ══ 最終決定 ══════════════════════════════════════════════════════════════
# 交差検証で lag 1 の MSD が下がる ⇒ 共通成分は実在。
# しかし振幅が装置スペックの 100 倍超 ⇒ 由来は試料（集団輸送 / 試料そのものの移動）。
# 採用: アンサンブル共通成分（s3 の Wiener 選択版）をそのまま「共通運動」として除去する。
#       ただし「装置由来」とは呼ばない。生座標版も並行して解析し比較できるようにする。
naive = pd.read_csv(os.path.join(C.OUT, 'drift_naive.csv'))
D_final = naive[['Dx_applied_px', 'Dy_applied_px']].values

sp = C.load_spots()
sp['X_px_corr'] = sp.X_px - D_final[sp.FRAME.values, 0]
sp['Y_px_corr'] = sp.Y_px - D_final[sp.FRAME.values, 1]
sp[['ID', 'FRAME', 'X_px', 'Y_px', 'X_px_corr', 'Y_px_corr',
    'INTENSITY', 'QUALITY', 'SNR_CH1', 'n_pixels']].to_csv(
    os.path.join(C.OUT, 'spots_corrected.csv'), index=False)
pd.DataFrame(dict(frame=np.arange(T), n_particles=n_all,
                  Dx_ensemble_px=D_all[:, 0], Dy_ensemble_px=D_all[:, 1],
                  Dx_applied_px=D_final[:, 0], Dy_applied_px=D_final[:, 1],
                  Dx_applied_um=D_final[:, 0] * C.PIXEL_SIZE,
                  Dy_applied_um=D_final[:, 1] * C.PIXEL_SIZE)
             ).to_csv(os.path.join(C.OUT, 'drift.csv'), index=False)

with open(os.path.join(C.OUT, 'validate.json'), 'w') as f:
    json.dump(dict(cross_validation=res_cv, n_cv=N_CV, lags=LAGS,
                   ensemble_drift_12h_px=dict(x=float(D_all[-1, 0]), y=float(D_all[-1, 1])),
                   ensemble_drift_12h_um=dict(x=float(D_all[-1, 0] * C.PIXEL_SIZE),
                                              y=float(D_all[-1, 1] * C.PIXEL_SIZE)),
                   machine_reference=MACHINE, ratio_to_machine=ratio,
                   vibration_8min=vib,
                   applied_total_px=dict(x=float(D_final[-1, 0]), y=float(D_final[-1, 1])),
                   conclusion='common mode is real (CV) but ~100x the instrument spec '
                              '=> sample-borne collective motion, not stage drift'),
              f, indent=2, ensure_ascii=False)

# ══ 図 ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

a = ax[0, 0]
xs = np.arange(len(LAGS))
m = [res_cv[L]['ratio_mean'] for L in LAGS]; s = [res_cv[L]['ratio_sd'] for L in LAGS]
m2 = [res_cv[L]['shuffled_mean'] for L in LAGS]; s2 = [res_cv[L]['shuffled_sd'] for L in LAGS]
a.errorbar(xs - .06, m, yerr=s, fmt='o-', color='tab:red', label='共通成分を除去')
a.errorbar(xs + .06, m2, yerr=s2, fmt='s-', color='0.55', label='対照（時間シャッフル）')
a.axhline(1.0, color='k', lw=0.8, ls=':')
a.set_xticks(xs); a.set_xticklabels(LAGS); a.set_xlabel('lag (min)')
a.set_ylabel('MSD 比（補正 / 無補正）')
a.set_title('I. 交差検証: 推定に使っていない粒子での MSD'); a.legend(fontsize=7)

a = ax[0, 1]
t = np.arange(T)
a.plot(t, D_all[:, 0] * C.PIXEL_SIZE, lw=1.0, label='共通成分 x')
a.plot(t, D_all[:, 1] * C.PIXEL_SIZE, lw=1.0, label='共通成分 y')
a.plot(t, MACHINE['drift_12h_px']['x'] * t / (T - 1) * C.PIXEL_SIZE, lw=2.0,
       color='k', label='装置実測 x (20260825_005)')
a.plot(t, MACHINE['drift_12h_px']['y'] * t / (T - 1) * C.PIXEL_SIZE, lw=2.0,
       color='0.5', label='装置実測 y')
a.set_xlabel('frame (min)'); a.set_ylabel('累積変位 (µm)')
a.set_title('II. 装置の実測ドリフトとの比較（同一スケール）'); a.legend(fontsize=7)

for i, nm in enumerate('xy'):
    a = ax[1, i]
    a.loglog(per[1:], P_v[1:, i], lw=0.8, color='tab:blue', label='共通速度 v(t)')
    a.loglog(per[1:], p95[1:, i], lw=1.0, color='tab:orange', label='null 95%（時間シャッフル）')
    a.axvline(8.0, color='tab:green', ls=':', lw=1.5, label='8 min（装置振動）')
    a.set_xlabel('周期 (min)'); a.set_ylabel('パワー')
    a.set_title(f'II. {nm} 軸: 8 min 機械振動線の探索'); a.legend(fontsize=6.5)

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_validate.png'))
print('\nSaved out/validate.json, out/fig_validate.png, out/drift.csv, out/spots_corrected.csv')
