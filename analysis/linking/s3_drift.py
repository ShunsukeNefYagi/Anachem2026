"""
s3_drift.py — 全粒子に共通する運動（ステージドリフト・機械振動）の推定と除去

考え方
------
視野内の粒子はそれぞれ独立にブラウン運動しているが、ステージドリフトや機械振動が
あると「全粒子が同じベクトルだけ同時に動く」成分が重畳する。フレーム t における
粒子 i の変位は

    Δr_i(t) = c(t) + u_i(t)          c: 共通成分, u_i: 粒子固有の運動 + 位置決め誤差

と書ける。c(t) は粒子について平均（中央値）を取れば取り出せるが、**粒子数が
たかだか 13 個しかないので、u_i の平均は 0 にならない**。素朴に median を積算すると

    σ(median) ≈ 1.25 σ_u / √n   →   累積は √T 倍のランダムウォーク

として「存在しないドリフト」を数十 px も作ってしまう。したがって

  (1) 共通成分が本当に有るのか（有るならどれだけか）を**推定量そのものと分離して**測る
  (2) 測れた分だけを引く（ノイズは引かない）

の 2 段構えにする。

(1) A/B 分割法（20260825_005 §10-2 のフレーム 2 分割法と同じ発想）
    各フレームで寄与粒子をランダムに 2 群 A/B に分け、それぞれの median を取る。
      med_A = c + n_A,  med_B = c + n_B   (n_A ⟂ n_B)
      Cov(med_A, med_B) = Var(c)                    ← 共通成分の分散（不偏）
      w = (med_A − med_B)/2                          ← 共通成分が完全に消えた「ノイズだけ」の系列
    しかも Var(w) = Var(全体 median のノイズ) なので、w を積算した cumsum(w) は
    「共通成分がゼロだったときにこの推定器が吐くドリフト曲線」そのものになる。
    → 帰無分布を仮定なしで作れる。

(2) 直線成分は null の傾き分布と比べて有意なら採用、
    残差（振動・ゆらぎ）は null のパワースペクトルを使った Wiener フィルタ
      H(f) = max(0, 1 − P_null(f) / P_obs(f))
    で、ノイズを超えて出ている帯域だけを通す。周期的な機械振動があればここでピークとして残る。

入力 : out/spots_raw.csv, out/spot_intensity_square_roi.csv
出力 : out/drift_naive.csv, out/spots_corrected_naive.csv, out/drift_stats_naive.json,
       out/fig_drift_naive.png, out/fig_drift_spectrum_naive.png
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

rng = np.random.default_rng(20260901)

MIN_DET_FOR_DRIFT = 10     # ドリフト推定に使うトラックの最小検出数
MIN_N_PER_FRAME   = 4      # このフレームで寄与粒子がこれ未満なら c(t)=0 とする
K_SPLIT           = 400    # A/B ランダム分割の反復数
N_ITER            = 2      # リンク→ドリフト推定→補正 の反復回数

C.slide_style()
os.makedirs(C.OUT, exist_ok=True)


def link_and_displacements(df, xcol, ycol):
    """与えた px 座標でリンクし、連続フレーム間の変位リストを返す。"""
    d = df.copy()
    d['POSITION_X'] = d[xcol] * C.PIXEL_SIZE
    d['POSITION_Y'] = d[ycol] * C.PIXEL_SIZE
    fin = C.track(d)
    tr  = C.tracks_to_frame(fin, d)
    # px に戻す
    tr['X_px'] = tr['X_um'] / C.PIXEL_SIZE
    tr['Y_px'] = tr['Y_um'] / C.PIXEL_SIZE

    long_tr = tr[tr['DETECTIONS'] >= MIN_DET_FOR_DRIFT]
    disp = {t: [] for t in range(1, C.N_FRAMES)}
    for tid, g in long_tr.groupby('TRACK_ID'):
        g = g.sort_values('FRAME')
        f = g['FRAME'].values
        x = g['X_px'].values
        y = g['Y_px'].values
        step = np.diff(f) == 1                      # 連続フレームのみ
        for k in np.nonzero(step)[0]:
            disp[int(f[k + 1])].append((tid, x[k + 1] - x[k], y[k + 1] - y[k]))
    return fin, tr, disp


def estimate_common(disp):
    """median 推定量と、A/B 分割によるノイズ系列 w_k（K_SPLIT 本）を返す。"""
    T = C.N_FRAMES
    v      = np.zeros((T, 2))          # v[t] = c_hat(t)  (t=0 は 0)
    nobs   = np.zeros(T, dtype=int)
    w      = np.zeros((K_SPLIT, T, 2))  # ノイズだけの系列
    cov_ab = []                         # (med_A, med_B) のペア（分散分解用）

    for t in range(1, T):
        arr = disp.get(t, [])
        n = len(arr)
        nobs[t] = n
        if n < MIN_N_PER_FRAME:
            continue
        d = np.array([[a[1], a[2]] for a in arr])
        v[t] = np.median(d, axis=0)

        if n >= 2 * 3:                  # A/B とも 3 個以上
            for k in range(K_SPLIT):
                idx = rng.permutation(n)
                h = n // 2
                mA = np.median(d[idx[:h]], axis=0)
                mB = np.median(d[idx[h:2 * h]], axis=0)
                w[k, t] = (mA - mB) / 2.0
                if k < 40:
                    cov_ab.append((mA, mB))
    return v, nobs, w, np.array(cov_ab)


def wiener_correct(v, w):
    """cumsum(v) から、null (cumsum(w)) を超える成分だけを取り出す。"""
    T = len(v)
    tt = np.arange(T)
    D_obs  = np.cumsum(v, axis=0)                 # (T,2)
    D_null = np.cumsum(w, axis=1)                 # (K,T,2)

    out  = np.zeros_like(D_obs)
    info = {}
    for ax, name in enumerate('xy'):
        d = D_obs[:, ax]
        dn = D_null[:, :, ax]

        # --- 直線成分 ---
        A = np.vstack([tt, np.ones(T)]).T
        slope_obs = np.linalg.lstsq(A, d, rcond=None)[0][0]
        slope_null = np.array([np.linalg.lstsq(A, x, rcond=None)[0][0] for x in dn])
        s_sd = slope_null.std()
        z = slope_obs / s_sd if s_sd > 0 else 0.0
        keep_lin = abs(z) > 2.0
        lin_obs  = slope_obs * tt if keep_lin else np.zeros(T)

        # --- 直線を抜いた残差を周波数領域で Wiener ---
        r_obs  = d - np.linalg.lstsq(A, d, rcond=None)[0][0] * tt \
                   - np.linalg.lstsq(A, d, rcond=None)[0][1]
        r_null = np.array([x - np.linalg.lstsq(A, x, rcond=None)[0][0] * tt
                             - np.linalg.lstsq(A, x, rcond=None)[0][1] for x in dn])

        F_obs  = np.fft.rfft(r_obs)
        P_obs  = np.abs(F_obs) ** 2
        P_null = np.mean(np.abs(np.fft.rfft(r_null, axis=1)) ** 2, axis=0)

        H = np.maximum(0.0, 1.0 - P_null / np.maximum(P_obs, 1e-30))
        r_filt = np.fft.irfft(F_obs * H, n=T)

        out[:, ax] = lin_obs + r_filt
        info[name] = dict(
            slope_obs_px_per_frame=float(slope_obs),
            slope_null_sd=float(s_sd), slope_z=float(z), keep_linear=bool(keep_lin),
            total_drift_linear_px=float(slope_obs * (T - 1)) if keep_lin else 0.0,
            resid_rms_obs_px=float(np.std(r_obs)),
            resid_rms_null_px=float(np.mean([np.std(x) for x in r_null])),
            resid_rms_kept_px=float(np.std(r_filt)),
            frac_bands_passed=float(np.mean(H > 0.5)),
        )
        info[name]['_P_obs']  = P_obs
        info[name]['_P_null'] = P_null
        info[name]['_H']      = H
        info[name]['_r_obs']  = r_obs
        info[name]['_r_filt'] = r_filt
    return out, D_obs, D_null, info


# ══════════════════════════════════════════════════════════════════════════
df = C.load_spots()
print(f'spots {len(df)}  frames {df.FRAME.min()}-{df.FRAME.max()}', flush=True)
print(f'intensity: mean={df.INTENSITY.mean():.1f} median={df.INTENSITY.median():.1f}\n',
      flush=True)

df['X_px_corr'] = df['X_px']
df['Y_px_corr'] = df['Y_px']

D_total = np.zeros((C.N_FRAMES, 2))
iter_log = []

for it in range(N_ITER):
    xc = 'X_px' if it == 0 else 'X_px_corr'
    yc = 'Y_px' if it == 0 else 'Y_px_corr'
    print(f'── iteration {it}: linking on {xc}/{yc} …', flush=True)
    fin, tr, disp = link_and_displacements(df, xc, yc)
    ngood = len(C.good_tracks(fin))
    nlong = tr[tr.DETECTIONS >= MIN_DET_FOR_DRIFT].TRACK_ID.nunique()
    print(f'   tracks>=3: {ngood}, tracks>={MIN_DET_FOR_DRIFT}: {nlong}, '
          f'spots used: {sum(len(v) for v in disp.values())}', flush=True)

    v, nobs, w, cov_ab = estimate_common(disp)

    # 分散分解
    dec = {}
    for ax, name in enumerate('xy'):
        a = cov_ab[:, 0, ax]; b = cov_ab[:, 1, ax]
        var_common = float(np.mean((a - a.mean()) * (b - b.mean())))
        var_half   = float(np.mean([a.var(), b.var()]))
        dec[name] = dict(var_common_px2=var_common,
                         var_half_px2=var_half,
                         var_noise_half_px2=var_half - var_common,
                         common_fraction=var_common / var_half if var_half > 0 else 0.0,
                         rms_common_px=float(np.sqrt(max(var_common, 0))),
                         rms_noise_of_estimator_px=float(np.std(w[:, 1:, ax])))
    print(f'   共通成分 RMS: x {dec["x"]["rms_common_px"]:.4f} px, '
          f'y {dec["y"]["rms_common_px"]:.4f} px   |   '
          f'推定器ノイズ RMS: x {dec["x"]["rms_noise_of_estimator_px"]:.4f}, '
          f'y {dec["y"]["rms_noise_of_estimator_px"]:.4f} px', flush=True)

    D_keep, D_obs, D_null, info = wiener_correct(v, w)
    for name in 'xy':
        i = info[name]
        print(f'   {name}: 直線 {i["slope_obs_px_per_frame"]*1e3:+.3f}e-3 px/frame '
              f'(z={i["slope_z"]:+.2f}, keep={i["keep_linear"]}) '
              f'→ 12h で {i["total_drift_linear_px"]:+.2f} px ; '
              f'残差 RMS obs {i["resid_rms_obs_px"]:.2f} / null {i["resid_rms_null_px"]:.2f} '
              f'→ 採用 {i["resid_rms_kept_px"]:.2f} px', flush=True)

    D_total = D_total + D_keep
    df['X_px_corr'] = df['X_px'] - D_total[df['FRAME'].values, 0]
    df['Y_px_corr'] = df['Y_px'] - D_total[df['FRAME'].values, 1]

    iter_log.append(dict(iteration=it, n_tracks_ge3=int(ngood),
                         n_tracks_used=int(nlong),
                         variance_decomposition=dec,
                         wiener={k: {kk: vv for kk, vv in info[k].items()
                                     if not kk.startswith('_')} for k in 'xy'},
                         applied_range_px=dict(
                             x=[float(D_keep[:, 0].min()), float(D_keep[:, 0].max())],
                             y=[float(D_keep[:, 1].min()), float(D_keep[:, 1].max())])))
    if it == 0:
        first = dict(v=v.copy(), nobs=nobs.copy(), D_obs=D_obs.copy(),
                     D_null=D_null.copy(), info=info, dec=dec)

# ── 保存 ──────────────────────────────────────────────────────────────────
drift = pd.DataFrame(dict(
    frame=np.arange(C.N_FRAMES),
    n_particles=first['nobs'],
    vx_raw_px=first['v'][:, 0], vy_raw_px=first['v'][:, 1],
    Dx_raw_px=first['D_obs'][:, 0], Dy_raw_px=first['D_obs'][:, 1],
    Dx_applied_px=D_total[:, 0], Dy_applied_px=D_total[:, 1],
    Dx_applied_um=D_total[:, 0] * C.PIXEL_SIZE,
    Dy_applied_um=D_total[:, 1] * C.PIXEL_SIZE))
drift.to_csv(os.path.join(C.OUT, 'drift_naive.csv'), index=False)

df[['ID', 'FRAME', 'X_px', 'Y_px', 'X_px_corr', 'Y_px_corr', 'INTENSITY',
    'QUALITY', 'SNR_CH1', 'n_pixels']].to_csv(
    os.path.join(C.OUT, 'spots_corrected_naive.csv'), index=False)

stats = dict(
    method='ensemble median displacement + A/B split null + Wiener band selection',
    min_det_for_drift=MIN_DET_FOR_DRIFT, min_n_per_frame=MIN_N_PER_FRAME,
    k_split=K_SPLIT, n_iter=N_ITER,
    frames_with_enough_particles=int((first['nobs'] >= MIN_N_PER_FRAME).sum()),
    n_particles_per_frame=dict(mean=float(first['nobs'][1:].mean()),
                               min=int(first['nobs'][1:].min()),
                               max=int(first['nobs'][1:].max())),
    raw_cumsum_total_px=dict(x=float(first['D_obs'][-1, 0]),
                             y=float(first['D_obs'][-1, 1])),
    null_cumsum_endpoint_sd_px=dict(x=float(first['D_null'][:, -1, 0].std()),
                                    y=float(first['D_null'][:, -1, 1].std())),
    applied_total_px=dict(x=float(D_total[-1, 0]), y=float(D_total[-1, 1])),
    applied_max_excursion_px=dict(x=float(np.abs(D_total[:, 0]).max()),
                                  y=float(np.abs(D_total[:, 1]).max())),
    iterations=iter_log)
with open(os.path.join(C.OUT, 'drift_stats_naive.json'), 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

# ── 図 1: ドリフト曲線 ────────────────────────────────────────────────────
t = np.arange(C.N_FRAMES)
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

for ax_i, name in enumerate('xy'):
    a = ax[ax_i, 0]
    lo = np.percentile(first['D_null'][:, :, ax_i], 2.5, axis=0)
    hi = np.percentile(first['D_null'][:, :, ax_i], 97.5, axis=0)
    a.fill_between(t, lo, hi, color='0.8',
                   label='null 95% (共通成分ゼロのときの推定器の出力)')
    a.plot(t, first['D_obs'][:, ax_i], lw=1.0, color='tab:blue',
           label='cumsum(median) 生')
    a.plot(t, D_total[:, ax_i], lw=1.8, color='tab:red', label='採用した補正量')
    a.set_xlabel('frame (min)'); a.set_ylabel(f'{name} 累積変位 (px)')
    a.set_title(f'{name} 軸: 共通成分の累積'); a.legend(fontsize=7)

a = ax[0, 1]
a.plot(t[1:], first['nobs'][1:], lw=0.6, color='0.3')
a.axhline(MIN_N_PER_FRAME, color='r', ls=':', lw=1)
a.set_xlabel('frame'); a.set_ylabel('寄与粒子数')
a.set_title(f'ドリフト推定に寄与した粒子数 (DETECTIONS≥{MIN_DET_FOR_DRIFT})')

a = ax[1, 1]
a.plot(D_total[:, 0] * C.PIXEL_SIZE * 1000, D_total[:, 1] * C.PIXEL_SIZE * 1000,
       '-', lw=1.2, color='tab:red')
a.scatter([0], [0], c='k', s=25, zorder=5, label='frame 0')
a.set_xlabel('Δx (nm)'); a.set_ylabel('Δy (nm)')
a.set_title('採用した共通ドリフトの軌跡'); a.legend(fontsize=7); a.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_drift_naive.png'))
print('\nSaved out/fig_drift_naive.png')

# ── 図 2: スペクトル ──────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
freq = np.fft.rfftfreq(C.N_FRAMES, d=1.0)     # cycles / frame(min)
per = np.empty_like(freq); per[0] = np.inf; per[1:] = 1.0 / freq[1:]
for ax_i, name in enumerate('xy'):
    a = ax[ax_i]
    i = first['info'][name]
    a.loglog(per[1:], i['_P_obs'][1:], lw=0.8, color='tab:blue', label='観測 (cumsum median)')
    a.loglog(per[1:], i['_P_null'][1:], lw=1.2, color='0.5', label='null (A/B ノイズ)')
    keep = i['_H'][1:] > 0.5
    a.scatter(per[1:][keep], i['_P_obs'][1:][keep], s=10, color='tab:red',
              zorder=5, label='採用帯域 (H>0.5)')
    a.axvline(8.0, color='tab:green', ls=':', lw=1, label='8 min (20260825_005 の振動)')
    a.set_xlabel('周期 (min)'); a.set_ylabel('パワー')
    a.set_title(f'{name} 軸 累積変位のパワースペクトル（直線成分除去後）')
    a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_drift_spectrum_naive.png'))
print('Saved out/fig_drift_spectrum_naive.png')

print(f'\nSaved out/drift_naive.csv, out/spots_corrected_naive.csv, out/drift_stats_naive.json')
print(f'最終的に適用した補正量: '
      f'x {D_total[:,0].min():+.2f}…{D_total[:,0].max():+.2f} px, '
      f'y {D_total[:,1].min():+.2f}…{D_total[:,1].max():+.2f} px')
