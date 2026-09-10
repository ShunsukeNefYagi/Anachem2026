"""
s3e_decompose.py — 共通成分を「装置由来（高周波＝機械振動）」と
                   「試料由来（低周波＝集団輸送）」に分解し、除去版の座標を作る。

ここまでで分かったこと
----------------------
* 共通成分の 1 フレームあたり分散分解（s3 の A/B 分割、バイアス補正後）
      Var(共通) = Cov(med_A, med_B) − Var(推定器ノイズ)
    x: 3.077² − 1.878² = 5.94 px²  → RMS 2.44 px   ← 実在
    y: 1.612² − 1.717² < 0         → 検出されず
  ※ s3 が「共通 RMS 3.08 px」と出したのは、同じフレームを繰り返し分割したため
    Cov に推定器ノイズ Var(m) が乗った上方バイアス。ここで差し引いた。

* この x の速い共通ゆらぎは 20260825_005（同一顕微鏡・固定粒子）の
    「x は秒スケールの振動があり 60 s 間隔では白色ノイズに見える、
      生データ x 標準偏差 133.8 nm (2.97 px) / y 39.1 nm (0.87 px)」
  と大きさも x≫y の非対称性も一致する。→ **装置の機械振動**である。

* 一方 12 h 累積 x −249 px (−11.2 µm) の低周波トレンドは、同じ顕微鏡の
  実測 12 h ドリフト（x −0.635 px, y +1.757 px）の 400 倍 / 67 倍であり、
  装置では説明できない。s3b の集積解析（集積中心までの中央値距離 530→98 px）
  と整合するので、**試料側の集団運動**である。

したがって周期 30 min を境に 2 つに分ける（20260825_005 §11 の帯域分解と同じ切り方）。

  1. Wiener 収縮   H(f) = max(0, 1 − P_noise(f)/P_v(f))   ← 推定器ノイズを通さない
  2. 帯域分割      周期 < 30 min → 装置（振動）  /  周期 ≥ 30 min → 試料（集団輸送）

出力する座標 3 種:
  raw  : 無補正
  vib  : 機械振動のみ除去  ← 推奨（装置由来だけを引く）
  full : 共通成分すべて除去（＝アンサンブル相対座標）

出力: out/drift_components.csv, out/spots_variants.csv,
      out/decompose.json, out/fig_decompose.png
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
rng = np.random.default_rng(2718)

MIN_DET_FOR_DRIFT = 10
MIN_N      = 4
K_SPLIT    = 400
CUT_PERIOD = 30.0        # min。これより短い周期を「装置由来」とする

T = C.N_FRAMES

# ── 変位を集める ─────────────────────────────────────────────────────────
df = C.load_spots()
df['POSITION_X'] = df.X_px * C.PIXEL_SIZE
df['POSITION_Y'] = df.Y_px * C.PIXEL_SIZE
tr = C.tracks_to_frame(C.track(df), df)
tr['X_px'] = tr.X_um / C.PIXEL_SIZE
tr['Y_px'] = tr.Y_um / C.PIXEL_SIZE
use = tr[tr.DETECTIONS >= MIN_DET_FOR_DRIFT]
print(f'tracks>=3 {tr.TRACK_ID.nunique()}, tracks>={MIN_DET_FOR_DRIFT} {use.TRACK_ID.nunique()}')

disp = {t: [] for t in range(T)}
for tid, g in use.groupby('TRACK_ID'):
    g = g.sort_values('FRAME')
    f, x, y = g.FRAME.values, g.X_px.values, g.Y_px.values
    for k in np.nonzero(np.diff(f) == 1)[0]:
        disp[int(f[k + 1])].append((x[k + 1] - x[k], y[k + 1] - y[k]))

v = np.zeros((T, 2)); nobs = np.zeros(T, int)
w = np.zeros((K_SPLIT, T, 2))
cov_pairs = []
for t in range(1, T):
    d = np.array(disp[t]) if disp[t] else np.zeros((0, 2))
    nobs[t] = len(d)
    if len(d) < MIN_N:
        continue
    v[t] = np.median(d, axis=0)
    if len(d) >= 6:
        for k in range(K_SPLIT):
            idx = rng.permutation(len(d)); h = len(d) // 2
            a = np.median(d[idx[:h]], axis=0); b = np.median(d[idx[h:2 * h]], axis=0)
            w[k, t] = (a - b) / 2
            if k < 40:
                cov_pairs.append((a, b))
cov_pairs = np.array(cov_pairs)

# ── 分散分解（バイアス補正あり）──────────────────────────────────────────
# Cov(med_A, med_B) = Var(c) + Var(m)   （m = 全体中央値の推定器ノイズ）
# Var(m) は w の分散で測れる（half-sample 分散推定量）
dec = {}
print('\n1 フレームあたりの分散分解 [px²]:')
for i, nm in enumerate('xy'):
    a, b = cov_pairs[:, 0, i], cov_pairs[:, 1, i]
    cov = float(np.mean((a - a.mean()) * (b - b.mean())))
    var_noise = float(np.var(w[:, 1:, i]))
    var_c = cov - var_noise
    # 粒子固有の運動の分散（参考）
    allstep = np.concatenate([np.array(disp[t])[:, i] for t in range(1, T) if disp[t]])
    dec[nm] = dict(cov_AB=cov, var_estimator_noise=var_noise,
                   var_common=var_c, rms_common_px=float(np.sqrt(max(var_c, 0))),
                   rms_estimator_noise_px=float(np.sqrt(var_noise)),
                   var_total_step=float(allstep.var()),
                   shrinkage_lambda=float(max(var_c, 0) / (max(var_c, 0) + var_noise)))
    print(f'  {nm}: Cov(A,B) {cov:7.3f}  − 推定器ノイズ {var_noise:6.3f} '
          f'= 共通 {var_c:7.3f}  (RMS {np.sqrt(max(var_c,0)):.3f} px)   '
          f'粒子ステップ全体 {allstep.var():.1f} px²   λ={dec[nm]["shrinkage_lambda"]:.3f}')

# ── Wiener 収縮 + 帯域分割 ───────────────────────────────────────────────
freq = np.fft.rfftfreq(T, 1.0)
per  = np.r_[np.inf, 1.0 / freq[1:]]
fast = per < CUT_PERIOD          # 装置帯域
slow = ~fast

SMOOTH = 15   # ピリオドグラムの移動平均幅 [bin]


def smooth(p, m=SMOOTH):
    """ピリオドグラムの平滑化。1 実現のピリオドグラムは自由度 2 で
    ばらつきが 100 % あるため、平滑化しないと Wiener H がノイズを通してしまう。"""
    k = np.ones(m) / m
    return np.convolve(np.r_[p[m::-1], p, p[-2:-m - 2:-1]], k, mode='same')[m + 1:m + 1 + len(p)]


comp = {}
D = {'vib': np.zeros((T, 2)), 'slow': np.zeros((T, 2)), 'full': np.zeros((T, 2))}
print(f'\nWiener 収縮 + 帯域分割（境界 周期 {CUT_PERIOD:.0f} min, 平滑化 {SMOOTH} bin）:')
for i, nm in enumerate('xy'):
    V  = np.fft.rfft(v[:, i] - v[:, i].mean())
    P  = np.abs(V) ** 2
    Pn = np.mean(np.abs(np.fft.rfft(w[:, :, i] - w[:, :, i].mean(axis=1, keepdims=True),
                                    axis=1)) ** 2, axis=0)
    # 帰無の閾値は 400 本の w 実現の平滑化ピリオドグラムの実測 99 パーセンタイル
    Pw_all = np.abs(np.fft.rfft(w[:, :, i] - w[:, :, i].mean(axis=1, keepdims=True),
                                axis=1)) ** 2
    Pw_s   = np.array([smooth(p) for p in Pw_all])
    Ps, Pns = smooth(P), smooth(Pn)
    P99 = np.percentile(Pw_s, 99, axis=0)
    H  = np.maximum(0.0, 1.0 - Pns / np.maximum(Ps, 1e-30))
    H[Ps < P99] = 0.0            # 帰無 99% を超えないビンは通さない

    v_fast = np.fft.irfft(V * H * fast, n=T)
    v_slow = np.fft.irfft(V * H * slow, n=T) + v[:, i].mean()
    D['vib'][:, i]  = np.cumsum(v_fast)
    D['slow'][:, i] = np.cumsum(v_slow)
    D['full'][:, i] = D['vib'][:, i] + D['slow'][:, i]

    comp[nm] = dict(
        rms_v_fast_px=float(v_fast.std()), rms_v_slow_px=float(v_slow.std()),
        vib_excursion_px=float(np.ptp(D['vib'][:, i])),
        slow_total_px=float(D['slow'][-1, i]),
        full_total_px=float(D['full'][-1, i]),
        mean_H_fast=float(H[fast].mean()), mean_H_slow=float(H[slow].mean()))
    print(f'  {nm}: 振動成分 RMS {v_fast.std():.3f} px/frame '
          f'(累積振れ幅 {np.ptp(D["vib"][:,i]):.1f} px) | '
          f'低周波 12 h 累積 {D["slow"][-1,i]:+.1f} px '
          f'({D["slow"][-1,i]*C.PIXEL_SIZE:+.2f} µm)')
    comp[nm]['_H'] = H; comp[nm]['_P'] = Ps; comp[nm]['_Pn'] = Pns

# ── 座標 3 種を書き出す ─────────────────────────────────────────────────
sp = C.load_spots()
f = sp.FRAME.values
out = sp[['ID', 'FRAME', 'X_px', 'Y_px', 'INTENSITY', 'QUALITY', 'SNR_CH1', 'n_pixels']].copy()
out['X_px_vib']  = sp.X_px - D['vib'][f, 0]
out['Y_px_vib']  = sp.Y_px - D['vib'][f, 1]
out['X_px_full'] = sp.X_px - D['full'][f, 0]
out['Y_px_full'] = sp.Y_px - D['full'][f, 1]
out.to_csv(os.path.join(C.OUT, 'spots_variants.csv'), index=False)

pd.DataFrame(dict(frame=np.arange(T), n_particles=nobs,
                  vx_median_px=v[:, 0], vy_median_px=v[:, 1],
                  Dx_vib_px=D['vib'][:, 0],  Dy_vib_px=D['vib'][:, 1],
                  Dx_slow_px=D['slow'][:, 0], Dy_slow_px=D['slow'][:, 1],
                  Dx_full_px=D['full'][:, 0], Dy_full_px=D['full'][:, 1],
                  Dx_vib_nm=D['vib'][:, 0] * C.PIXEL_SIZE * 1000,
                  Dy_vib_nm=D['vib'][:, 1] * C.PIXEL_SIZE * 1000,
                  Dx_full_um=D['full'][:, 0] * C.PIXEL_SIZE,
                  Dy_full_um=D['full'][:, 1] * C.PIXEL_SIZE)
             ).to_csv(os.path.join(C.OUT, 'drift_components.csv'), index=False)

with open(os.path.join(C.OUT, 'decompose.json'), 'w') as fjs:
    json.dump(dict(cut_period_min=CUT_PERIOD, variance_decomposition=dec,
                   components={k: {kk: vv for kk, vv in comp[k].items()
                                   if not kk.startswith('_')} for k in 'xy'},
                   machine_reference_20260825_005=dict(
                       x_position_sd_px=2.97, y_position_sd_px=0.87,
                       drift_12h_px=dict(x=-0.635, y=1.757),
                       note='x は秒スケール振動が 60 s 標本化で白色化して見える')),
              fjs, indent=2, ensure_ascii=False)

# ── 図 ──────────────────────────────────────────────────────────────────
t = np.arange(T)
fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
for i, nm in enumerate('xy'):
    a = ax[i, 0]
    a.plot(t, D['full'][:, i] * C.PIXEL_SIZE, lw=1.0, color='tab:blue', label='full（共通成分すべて）')
    a.plot(t, D['slow'][:, i] * C.PIXEL_SIZE, lw=1.4, color='tab:orange',
           label=f'低周波（周期≥{CUT_PERIOD:.0f} min）= 試料由来')
    a.plot(t, D['vib'][:, i] * C.PIXEL_SIZE, lw=1.0, color='tab:red',
           label=f'高周波（<{CUT_PERIOD:.0f} min）= 機械振動')
    a.set_xlabel('frame (min)'); a.set_ylabel(f'{nm} 累積 (µm)')
    a.set_title(f'{nm} 軸の共通成分の分解'); a.legend(fontsize=6.5)

    a = ax[i, 1]
    a.plot(t, D['vib'][:, i] * C.PIXEL_SIZE * 1000, lw=0.8, color='tab:red')
    a.set_xlabel('frame (min)'); a.set_ylabel('nm')
    a.set_title(f'{nm} 除去する機械振動成分（拡大）'
                f'  RMS {D["vib"][:,i].std()*C.PIXEL_SIZE*1000:.0f} nm')

    a = ax[i, 2]
    a.loglog(per[1:], comp[nm]['_P'][1:], lw=0.7, color='tab:blue', label='共通速度 v(t)')
    a.loglog(per[1:], comp[nm]['_Pn'][1:], lw=1.1, color='0.5', label='推定器ノイズ')
    a.axvline(CUT_PERIOD, color='k', ls='--', lw=1.0, label=f'{CUT_PERIOD:.0f} min 境界')
    a.axvline(8.0, color='tab:green', ls=':', lw=1.2, label='8 min（20260825_005 の振動）')
    a.set_xlabel('周期 (min)'); a.set_ylabel('パワー')
    a.set_title(f'{nm} 軸 共通速度のスペクトル'); a.legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_decompose.png'))
print('\nSaved out/spots_variants.csv, out/drift_components.csv, '
      'out/decompose.json, out/fig_decompose.png')
