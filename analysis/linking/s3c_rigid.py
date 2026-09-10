"""
s3c_rigid.py — 「剛体並進（＝装置由来）」だけを取り出して除去する（本番の補正）

s3 の素朴なアンサンブル中央値は x に −250 px という巨大な共通成分を検出したが、
s3b の診断で

  * 視野左半分 +70 px / 右半分 −264 px と**空間的に一様でない**
  * 生座標のまま 81 分間ほとんど動かないトラック (T531, net 3.1 px) が存在する
  * 全粒子の集積中心までの距離が 530 → 98 px と収束している

ことが分かった。つまりその共通成分の大半は**細胞内の集団輸送（集積）という生物学**で
あって、ステージドリフトではない。これを引いてしまうと本題の所見が消える。

そこで「共通」の定義を **空間的に一様な成分だけ** に絞る。

  各フレームで粒子をその**フレーム内の中央値**で左右（および上下）に二分し、
      med_L = c_rigid + f_L + n_L
      med_R = c_rigid + f_R + n_R
  剛体並進 c_rigid は両群に同じだけ入るが、位置に依存する集団流 f は入り方が違う。
      v̂ = (med_L + med_R)/2      ← 推定量（全体中央値とほぼ同じ）
      w  = (med_L − med_R)/2      ← 剛体成分が完全に消えた系列
                                    ＝ 粒子ノイズ ＋ 空間不均一な集団流
  この w を帰無に使って s3 と同じ Wiener 選択をすれば、
  **空間的に一様な帯域だけ**が通る。左右・上下の 2 通りの分割で厳しい方を採用する。

出力: out/drift.csv, out/spots_corrected.csv, out/drift_stats.json,
      out/fig_rigid.png, out/fig_rigid_spectrum.png
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

rng = np.random.default_rng(20260901)
MIN_DET_FOR_DRIFT = 10
MIN_N_SPLIT       = 6      # 分割するには 1 フレームに 6 個以上（各群 3 個以上）
K_RANDOM          = 400
N_ITER            = 2


def displacements(df, xcol, ycol):
    d = df.copy()
    d['POSITION_X'] = d[xcol] * C.PIXEL_SIZE
    d['POSITION_Y'] = d[ycol] * C.PIXEL_SIZE
    fin = C.track(d)
    tr = C.tracks_to_frame(fin, d)
    tr['X_px'] = tr['X_um'] / C.PIXEL_SIZE
    tr['Y_px'] = tr['Y_um'] / C.PIXEL_SIZE
    rows = []
    for tid, g in tr[tr.DETECTIONS >= MIN_DET_FOR_DRIFT].groupby('TRACK_ID'):
        g = g.sort_values('FRAME')
        f, x, y = g.FRAME.values, g.X_px.values, g.Y_px.values
        for k in np.nonzero(np.diff(f) == 1)[0]:
            rows.append((int(f[k + 1]), tid, x[k], y[k], x[k + 1] - x[k], y[k + 1] - y[k]))
    return fin, tr, pd.DataFrame(rows, columns=['frame', 'tid', 'x0', 'y0', 'dx', 'dy'])


def split_series(D, mode):
    """mode: 'x' / 'y' = 位置で二分, 'rand' = ランダム二分（K_RANDOM 本）。
    返り値 v (T,2), w (K,T,2), nobs (T,)"""
    T = C.N_FRAMES
    K = K_RANDOM if mode == 'rand' else 1
    v = np.zeros((T, 2)); w = np.zeros((K, T, 2)); nobs = np.zeros(T, int)
    pairs = []
    for t, g in D.groupby('frame'):
        n = len(g); nobs[t] = n
        d = g[['dx', 'dy']].values
        v[t] = np.median(d, axis=0)
        if n < MIN_N_SPLIT:
            continue
        if mode == 'rand':
            for k in range(K):
                idx = rng.permutation(n); h = n // 2
                a = np.median(d[idx[:h]], axis=0); b = np.median(d[idx[h:2 * h]], axis=0)
                w[k, t] = (a - b) / 2
                if k < 40:
                    pairs.append((a, b))
        else:
            key = g[('x0' if mode == 'x' else 'y0')].values
            m = np.median(key)
            A, B = d[key < m], d[key >= m]
            if len(A) < 3 or len(B) < 3:      # 同値で偏ったら順位で強制二分
                o = np.argsort(key); h = n // 2
                A, B = d[o[:h]], d[o[h:]]
            a, b = np.median(A, axis=0), np.median(B, axis=0)
            v[t] = (a + b) / 2
            w[0, t] = (a - b) / 2
            pairs.append((a, b))
    return v, w, nobs, np.array(pairs)


def band_select(D_obs, w, n_null=400):
    """w（1 本または K 本）から帰無アンサンブルを作り、Wiener で帯域選択。"""
    T = len(D_obs); tt = np.arange(T)
    if w.shape[0] == 1:
        # 1 本しかない場合はブロック位相ランダム化で帰無アンサンブルを合成
        base = w[0]
        null_v = np.empty((n_null, T, 2))
        for k in range(n_null):
            for ax in range(2):
                sh = rng.permutation(T)          # 白色化（w は本質的に白色ノイズ）
                null_v[k, :, ax] = base[sh, ax]
    else:
        null_v = w
    D_null = np.cumsum(null_v, axis=1)

    out = np.zeros_like(D_obs); info = {}
    A = np.vstack([tt, np.ones(T)]).T
    for ax, name in enumerate('xy'):
        d = D_obs[:, ax]; dn = D_null[:, :, ax]
        co = np.linalg.lstsq(A, d, rcond=None)[0]
        sl_null = np.array([np.linalg.lstsq(A, x, rcond=None)[0][0] for x in dn])
        sd = sl_null.std(); z = co[0] / sd if sd > 0 else 0.0
        keep = abs(z) > 2.0
        lin = co[0] * tt if keep else np.zeros(T)

        r = d - co[0] * tt - co[1]
        rn = np.array([x - np.linalg.lstsq(A, x, rcond=None)[0][0] * tt
                         - np.linalg.lstsq(A, x, rcond=None)[0][1] for x in dn])
        F = np.fft.rfft(r)
        P, Pn = np.abs(F) ** 2, np.mean(np.abs(np.fft.rfft(rn, axis=1)) ** 2, axis=0)
        H = np.maximum(0.0, 1.0 - Pn / np.maximum(P, 1e-30))
        rf = np.fft.irfft(F * H, n=T)
        out[:, ax] = lin + rf
        info[name] = dict(slope_px_per_frame=float(co[0]), slope_null_sd=float(sd),
                          slope_z=float(z), keep_linear=bool(keep),
                          linear_total_px=float(co[0] * (T - 1)) if keep else 0.0,
                          resid_rms_obs_px=float(r.std()),
                          resid_rms_null_px=float(np.mean(rn.std(axis=1))),
                          resid_rms_kept_px=float(rf.std()),
                          _P=P, _Pn=Pn, _H=H)
    return out, info


# ══════════════════════════════════════════════════════════════════════════
df = C.load_spots()
df['X_px_corr'] = df['X_px']; df['Y_px_corr'] = df['Y_px']
D_total = np.zeros((C.N_FRAMES, 2))
log = []

for it in range(N_ITER):
    xc, yc = ('X_px', 'Y_px') if it == 0 else ('X_px_corr', 'Y_px_corr')
    print(f'── iteration {it} (link on {xc}) …', flush=True)
    fin, tr, Dsp = displacements(df, xc, yc)
    print(f'   tracks>=3 {len(C.good_tracks(fin))},  '
          f'used tracks {Dsp.tid.nunique()},  displacements {len(Dsp)}', flush=True)

    v_all, w_rand, nobs, pairs_rand = split_series(Dsp, 'rand')
    _,     w_lr,   _,    pairs_lr   = split_series(Dsp, 'x')
    _,     w_tb,   _,    pairs_tb   = split_series(Dsp, 'y')

    # 分散分解: ランダム分割 = 共通成分すべて / 位置分割 = 剛体成分のみ
    dec = {}
    for lbl, P in [('random', pairs_rand), ('LR', pairs_lr), ('TB', pairs_tb)]:
        dec[lbl] = {}
        for ax, nm in enumerate('xy'):
            a, b = P[:, 0, ax], P[:, 1, ax]
            cov = float(np.mean((a - a.mean()) * (b - b.mean())))
            dec[lbl][nm] = dict(cov_px2=cov, rms_px=float(np.sqrt(max(cov, 0.0))))
    print('   1 フレームあたり共通成分 RMS [px]  '
          f'(random = 剛体+集団流, LR/TB = 剛体のみ)')
    for lbl in ('random', 'LR', 'TB'):
        print(f'     {lbl:>6}: x {dec[lbl]["x"]["rms_px"]:.3f}  y {dec[lbl]["y"]["rms_px"]:.3f}')

    D_obs = np.cumsum(v_all, axis=0)
    # LR / TB それぞれで帯域選択し、絶対値の小さい（＝保守的な）方を採る
    out_lr, info_lr = band_select(D_obs, w_lr)
    out_tb, info_tb = band_select(D_obs, w_tb)
    D_keep = np.where(np.abs(out_lr) < np.abs(out_tb), out_lr, out_tb)

    for nm, i1, i2 in [('x', info_lr['x'], info_tb['x']), ('y', info_lr['y'], info_tb['y'])]:
        print(f'   {nm}: 直線 z = {i1["slope_z"]:+.2f} (LR) / {i2["slope_z"]:+.2f} (TB) '
              f'→ keep {i1["keep_linear"]}/{i2["keep_linear"]} ; '
              f'残差 obs {i1["resid_rms_obs_px"]:.2f} vs null LR {i1["resid_rms_null_px"]:.2f} '
              f'/ TB {i2["resid_rms_null_px"]:.2f}', flush=True)

    D_total = D_total + D_keep
    df['X_px_corr'] = df['X_px'] - D_total[df.FRAME.values, 0]
    df['Y_px_corr'] = df['Y_px'] - D_total[df.FRAME.values, 1]
    print(f'   適用後の累積補正: x {D_total[:,0].min():+.2f}…{D_total[:,0].max():+.2f} px, '
          f'y {D_total[:,1].min():+.2f}…{D_total[:,1].max():+.2f} px', flush=True)

    log.append(dict(iteration=it, n_tracks_ge3=len(C.good_tracks(fin)),
                    n_tracks_used=int(Dsp.tid.nunique()),
                    per_frame_common_rms=dec,
                    band_select_LR={k: {kk: vv for kk, vv in info_lr[k].items()
                                        if not kk.startswith('_')} for k in 'xy'},
                    band_select_TB={k: {kk: vv for kk, vv in info_tb[k].items()
                                        if not kk.startswith('_')} for k in 'xy'},
                    applied_px=dict(x=[float(D_keep[:, 0].min()), float(D_keep[:, 0].max())],
                                    y=[float(D_keep[:, 1].min()), float(D_keep[:, 1].max())])))
    if it == 0:
        first = dict(D_obs=D_obs.copy(), w_lr=w_lr.copy(), w_tb=w_tb.copy(),
                     nobs=nobs.copy(), info_lr=info_lr, info_tb=info_tb,
                     tr=tr.copy(), D_keep=D_keep.copy())

# ── 静止トラックによる独立検証 ────────────────────────────────────────────
naive = pd.read_csv(os.path.join(C.OUT, 'drift_naive.csv'))
Dn = naive[['Dx_applied_px', 'Dy_applied_px']].values
check = []
for tid, g in first['tr'].groupby('TRACK_ID'):
    g = g.sort_values('FRAME')
    if len(g) < 50:
        continue
    f0, f1 = int(g.FRAME.iloc[0]), int(g.FRAME.iloc[-1])
    raw = np.hypot(g.X_px.iloc[-1] - g.X_px.iloc[0], g.Y_px.iloc[-1] - g.Y_px.iloc[0])
    dn_ = np.hypot(*(Dn[f1] - Dn[f0]))
    dr_ = np.hypot(*(D_total[f1] - D_total[f0]))
    check.append(dict(TRACK_ID=int(tid), n=len(g), f0=f0, f1=f1,
                      raw_net_px=float(raw), naive_drift_px=float(dn_),
                      rigid_drift_px=float(dr_)))
chk = pd.DataFrame(check).sort_values('raw_net_px')
print('\n静止トラックによる検証（生の正味移動量 vs 各補正量, px）:')
print(chk.head(8).to_string(index=False, float_format=lambda v: f'{v:.1f}'))
chk.to_csv(os.path.join(C.OUT, 'quiet_track_check.csv'), index=False)

# ── 保存 ──────────────────────────────────────────────────────────────────
pd.DataFrame(dict(frame=np.arange(C.N_FRAMES), n_particles=first['nobs'],
                  Dx_ensemble_px=first['D_obs'][:, 0], Dy_ensemble_px=first['D_obs'][:, 1],
                  Dx_applied_px=D_total[:, 0], Dy_applied_px=D_total[:, 1],
                  Dx_applied_um=D_total[:, 0] * C.PIXEL_SIZE,
                  Dy_applied_um=D_total[:, 1] * C.PIXEL_SIZE)
             ).to_csv(os.path.join(C.OUT, 'drift.csv'), index=False)

df[['ID', 'FRAME', 'X_px', 'Y_px', 'X_px_corr', 'Y_px_corr',
    'INTENSITY', 'QUALITY', 'SNR_CH1', 'n_pixels']].to_csv(
    os.path.join(C.OUT, 'spots_corrected.csv'), index=False)

with open(os.path.join(C.OUT, 'drift_stats.json'), 'w') as f:
    json.dump(dict(method='rigid-only: position-split null + Wiener band selection',
                   min_det_for_drift=MIN_DET_FOR_DRIFT, min_n_split=MIN_N_SPLIT,
                   n_iter=N_ITER,
                   applied_total_px=dict(x=float(D_total[-1, 0]), y=float(D_total[-1, 1])),
                   applied_max_excursion_px=dict(x=float(np.abs(D_total[:, 0]).max()),
                                                 y=float(np.abs(D_total[:, 1]).max())),
                   applied_max_excursion_nm=dict(
                       x=float(np.abs(D_total[:, 0]).max() * C.PIXEL_SIZE * 1000),
                       y=float(np.abs(D_total[:, 1]).max() * C.PIXEL_SIZE * 1000)),
                   iterations=log), f, indent=2, ensure_ascii=False)

# ── 図 ──────────────────────────────────────────────────────────────────
t = np.arange(C.N_FRAMES)
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for i, nm in enumerate('xy'):
    a = ax[i, 0]
    a.plot(t, first['D_obs'][:, i], lw=0.9, color='tab:blue',
           label='アンサンブル中央値の累積（剛体＋集団流）')
    a.plot(t, naive[['Dx_applied_px', 'Dy_applied_px']].values[:, i], lw=1.2,
           color='0.55', label='s3 素朴補正（ランダム分割 null）')
    a.plot(t, D_total[:, i], lw=2.0, color='tab:red', label='採用: 剛体成分のみ')
    a.set_xlabel('frame (min)'); a.set_ylabel(f'{nm} 累積変位 (px)')
    a.set_title(f'{nm} 軸'); a.legend(fontsize=7)

a = ax[0, 1]
a.plot(D_total[:, 0] * C.PIXEL_SIZE * 1000, D_total[:, 1] * C.PIXEL_SIZE * 1000,
       lw=1.2, color='tab:red')
a.scatter([0], [0], c='k', s=25, zorder=5, label='frame 0')
a.set_xlabel('Δx (nm)'); a.set_ylabel('Δy (nm)'); a.set_aspect('equal')
a.set_title('採用した剛体ドリフトの軌跡'); a.legend(fontsize=7)

a = ax[1, 1]
a.scatter(chk.raw_net_px, chk.naive_drift_px, s=18, color='0.55', label='s3 素朴補正')
a.scatter(chk.raw_net_px, chk.rigid_drift_px, s=18, color='tab:red', label='剛体のみ')
lim = max(chk.raw_net_px.max(), chk.naive_drift_px.max()) * 1.05
a.plot([0, lim], [0, lim], 'k:', lw=0.8, label='y = x（補正量が実測移動量と同じ）')
a.set_xlabel('生座標での正味移動量 (px)'); a.set_ylabel('その期間の補正量 (px)')
a.set_title('C. 静止トラックとの整合性（n≥50 のトラック）'); a.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_rigid.png'))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
freq = np.fft.rfftfreq(C.N_FRAMES, d=1.0); per = np.r_[np.inf, 1 / freq[1:]]
for i, nm in enumerate('xy'):
    a = ax[i]
    a.loglog(per[1:], first['info_lr'][nm]['_P'][1:], lw=0.8, color='tab:blue',
             label='観測（アンサンブル）')
    a.loglog(per[1:], first['info_lr'][nm]['_Pn'][1:], lw=1.2, color='tab:orange',
             label='null: 左右分割差（＝非剛体）')
    a.loglog(per[1:], first['info_tb'][nm]['_Pn'][1:], lw=1.2, color='tab:green',
             label='null: 上下分割差')
    keep = (first['info_lr'][nm]['_H'][1:] > 0.5) & (first['info_tb'][nm]['_H'][1:] > 0.5)
    a.scatter(per[1:][keep], first['info_lr'][nm]['_P'][1:][keep], s=12, color='tab:red',
              zorder=5, label='剛体として採用した帯域')
    a.axvline(8.0, color='0.4', ls=':', lw=1, label='8 min (20260825_005 の振動)')
    a.set_xlabel('周期 (min)'); a.set_ylabel('パワー')
    a.set_title(f'{nm} 軸 累積変位スペクトル（直線成分除去後）'); a.legend(fontsize=6.5)
plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_rigid_spectrum.png'))

print('\nSaved out/drift.csv, out/spots_corrected.csv, out/drift_stats.json, '
      'out/fig_rigid.png, out/fig_rigid_spectrum.png')
print(f'最終補正量: x {D_total[:,0].min():+.2f}…{D_total[:,0].max():+.2f} px '
      f'({D_total[:,0].min()*C.PIXEL_SIZE*1000:+.0f}…{D_total[:,0].max()*C.PIXEL_SIZE*1000:+.0f} nm), '
      f'y {D_total[:,1].min():+.2f}…{D_total[:,1].max():+.2f} px')
