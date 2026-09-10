"""
s3b_diagnose.py — 検出された「共通成分」がステージドリフトなのか、
                  それとも細胞内の集団輸送（＝消してはいけない生物学）なのかを判定する。

s3 が見つけた x 方向 −250 px (−11 µm) / 12 h という共通成分は大きい。過去ログ
には「粒子が特定のドメインに集積する」という所見があるので、
それを機械ドリフトと誤認して引いてしまう危険が現実にある。

区別する検査:
  A. 剛体並進性 — ステージドリフトなら視野のどこでも同じベクトル。
     視野を左右／上下に分けて別々にドリフトを推定し、一致するかを見る。
  B. アフィン分解 — Δr = c + B·(r − r̄) を各フレームで最小二乗フィットし、
     B のトレース（発散 = 一様な拡大縮小 → 集積なら負）と反対称成分（回転）を積算。
     null は粒子ラベルのブートストラップで作る。
  C. 静止粒子の有無 — 生座標のまま長期間ほとんど動かないトラックがあれば、
     その期間に大きなステージドリフトは存在し得ない（決定的な反証）。
  D. 集積中心への収束 — 各粒子の「集積中心までの距離」の時間変化。

出力: out/fig_diagnose.png, out/diagnose.json
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro',
                                      'Yu Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
C.slide_style()
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']

rng = np.random.default_rng(7)
MIN_DET = 10

df = C.load_spots()
df['POSITION_X'] = df['X_px'] * C.PIXEL_SIZE
df['POSITION_Y'] = df['Y_px'] * C.PIXEL_SIZE
fin = C.track(df)
tr  = C.tracks_to_frame(fin, df)
tr['X_px'] = tr['X_um'] / C.PIXEL_SIZE
tr['Y_px'] = tr['Y_um'] / C.PIXEL_SIZE
print(f'tracks>=3: {tr.TRACK_ID.nunique()}   spots: {len(tr)}')

long_tr = tr[tr['DETECTIONS'] >= MIN_DET]

# フレームごとの (x0, y0, dx, dy)
rec = []
for tid, g in long_tr.groupby('TRACK_ID'):
    g = g.sort_values('FRAME')
    f, x, y = g.FRAME.values, g.X_px.values, g.Y_px.values
    for k in np.nonzero(np.diff(f) == 1)[0]:
        rec.append((int(f[k + 1]), tid, x[k], y[k], x[k + 1] - x[k], y[k + 1] - y[k]))
D = pd.DataFrame(rec, columns=['frame', 'tid', 'x0', 'y0', 'dx', 'dy'])
print(f'frame-to-frame displacements: {len(D)}')

res = {}

# ── A. 視野を分けて別々に推定 ────────────────────────────────────────────
XMID = YMID = C.IMG_SIZE / 2.0


def drift_of(sub, minn=3):
    v = np.zeros((C.N_FRAMES, 2)); n = np.zeros(C.N_FRAMES, int)
    for t, g in sub.groupby('frame'):
        n[t] = len(g)
        if len(g) >= minn:
            v[t] = [np.median(g.dx), np.median(g.dy)]
    return np.cumsum(v, axis=0), n


halves = {
    'x<half (left)':  D[D.x0 < XMID],  'x>half (right)': D[D.x0 >= XMID],
    'y<half (top)':   D[D.y0 < YMID],  'y>half (bottom)': D[D.y0 >= YMID],
}
Dh = {k: drift_of(v) for k, v in halves.items()}
res['half_field'] = {
    k: dict(n_disp=int(len(halves[k])),
            end_x_px=float(d[0][-1, 0]), end_y_px=float(d[0][-1, 1]),
            frames_with_data=int((d[1] >= 3).sum()))
    for k, d in Dh.items()}
print('\nA. 視野半分ごとの累積ドリフト (12 h 終端, px):')
for k, d in Dh.items():
    print(f'   {k:>16}: x {d[0][-1,0]:+8.1f}  y {d[0][-1,1]:+8.1f}   '
          f'(n_disp={len(halves[k])})')

# ── B. アフィン分解 ─────────────────────────────────────────────────────
def affine_series(Dsub, minn=6):
    """各フレームで Δr = c + B(r-r̄) をフィット。c, B を返す。"""
    cs = np.zeros((C.N_FRAMES, 2)); Bs = np.zeros((C.N_FRAMES, 2, 2))
    ok = np.zeros(C.N_FRAMES, bool)
    for t, g in Dsub.groupby('frame'):
        if len(g) < minn:
            continue
        r = np.c_[g.x0.values, g.y0.values]
        r = r - r.mean(0)
        d = np.c_[g.dx.values, g.dy.values]
        A = np.c_[np.ones(len(g)), r]
        sol, *_ = np.linalg.lstsq(A, d, rcond=None)   # (3,2)
        cs[t] = sol[0]
        Bs[t] = sol[1:].T                             # d = c + B r
        ok[t] = True
    return cs, Bs, ok


cs, Bs, ok = affine_series(D)
div  = np.where(ok, Bs[:, 0, 0] + Bs[:, 1, 1], 0.0)          # 発散 (1/frame)
rot  = np.where(ok, Bs[:, 1, 0] - Bs[:, 0, 1], 0.0) / 2.0    # 回転 (rad/frame)
cum_div, cum_rot = np.cumsum(div), np.cumsum(rot)

# null: 各フレームで位置ラベルをシャッフル（位置と変位の対応を壊す）
NB = 300
null_div_end, null_rot_end = [], []
for b in range(NB):
    Ds = D.copy()
    Ds[['x0', 'y0']] = (D.groupby('frame')[['x0', 'y0']]
                          .transform(lambda s: s.sample(frac=1, random_state=rng.integers(1 << 30)).values))
    _, Bb, okb = affine_series(Ds)
    null_div_end.append(np.cumsum(np.where(okb, Bb[:, 0, 0] + Bb[:, 1, 1], 0)).sum() * 0 +
                        np.cumsum(np.where(okb, Bb[:, 0, 0] + Bb[:, 1, 1], 0))[-1])
    null_rot_end.append(np.cumsum(np.where(okb, (Bb[:, 1, 0] - Bb[:, 0, 1]) / 2, 0))[-1])
null_div_end = np.array(null_div_end); null_rot_end = np.array(null_rot_end)

z_div = cum_div[-1] / null_div_end.std()
z_rot = cum_rot[-1] / null_rot_end.std()
res['affine'] = dict(cum_divergence=float(cum_div[-1]), null_sd=float(null_div_end.std()),
                     z_divergence=float(z_div),
                     cum_rotation_rad=float(cum_rot[-1]),
                     z_rotation=float(z_rot),
                     interpretation='cum_divergence<0 は視野全体の収縮＝集積を意味する')
print(f'\nB. アフィン: 累積発散 {cum_div[-1]:+.4f} (null sd {null_div_end.std():.4f}, z={z_div:+.2f})')
print(f'   累積回転   {np.degrees(cum_rot[-1]):+.3f} deg (z={z_rot:+.2f})')

# ── C. 生座標での「ほぼ静止」トラックの探索 ──────────────────────────────
rows = []
for tid, g in tr.groupby('TRACK_ID'):
    g = g.sort_values('FRAME')
    if len(g) < 50:
        continue
    x, y = g.X_px.values, g.Y_px.values
    rows.append(dict(TRACK_ID=tid, n=len(g), f0=int(g.FRAME.iloc[0]), f1=int(g.FRAME.iloc[-1]),
                     net_px=float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
                     rms_px=float(np.hypot(x - x.mean(), y - y.mean()).mean()),
                     dx_px=float(x[-1] - x[0]), dy_px=float(y[-1] - y[0])))
st = pd.DataFrame(rows).sort_values('rms_px')
res['quiet_tracks'] = st.head(10).to_dict('records')
print('\nC. 生座標で最も動かないトラック（n>=50）:')
print(st.head(8).to_string(index=False, float_format=lambda v: f'{v:.1f}'))

# ── D. 集積中心への収束 ─────────────────────────────────────────────────
late = tr[tr.FRAME >= 500]
cx, cy = float(late.X_px.median()), float(late.Y_px.median())
tr['dist_c'] = np.hypot(tr.X_px - cx, tr.Y_px - cy)
prof = tr.groupby('FRAME')['dist_c'].median()
res['accumulation_center_px'] = [cx, cy]
res['median_distance_to_center_px'] = dict(
    frame0_100=float(prof.loc[0:100].mean()), frame300_400=float(prof.loc[300:400].mean()),
    frame620_719=float(prof.loc[620:719].mean()))
print(f'\nD. 集積中心 (frame>=500 の中央値) = ({cx:.0f}, {cy:.0f}) px')
print(f'   中心までの中央値距離: f0-100 {prof.loc[0:100].mean():.0f} → '
      f'f300-400 {prof.loc[300:400].mean():.0f} → f620-719 {prof.loc[620:719].mean():.0f} px')

# ── 図 ──────────────────────────────────────────────────────────────────
t = np.arange(C.N_FRAMES)
fig, ax = plt.subplots(2, 3, figsize=(17, 9))

for j, (lbl, keys) in enumerate([('x 軸', ['x<half (left)', 'x>half (right)']),
                                 ('y 軸', ['y<half (top)', 'y>half (bottom)'])]):
    a = ax[j, 0]
    for k in keys:
        a.plot(t, Dh[k][0][:, 0], lw=1.0, label=f'{k}  Δx')
        a.plot(t, Dh[k][0][:, 1], lw=1.0, ls='--', label=f'{k}  Δy')
    a.set_xlabel('frame (min)'); a.set_ylabel('累積変位 (px)')
    a.set_title(f'A. 視野を{lbl}で二分した独立推定'); a.legend(fontsize=6)

a = ax[0, 1]
a.plot(t, cum_div, lw=1.2, color='tab:purple')
a.axhline(0, color='k', lw=0.6)
a.fill_between(t, -2 * null_div_end.std(), 2 * null_div_end.std(), color='0.85',
               label='null ±2σ (終端)')
a.set_xlabel('frame'); a.set_ylabel('累積発散 (無次元)')
a.set_title(f'B. 変位場の発散（負＝収縮/集積）  z={z_div:+.1f}'); a.legend(fontsize=7)

a = ax[1, 1]
a.plot(t, np.degrees(cum_rot), lw=1.2, color='tab:olive')
a.axhline(0, color='k', lw=0.6)
a.set_xlabel('frame'); a.set_ylabel('累積回転 (deg)')
a.set_title(f'B. 変位場の回転  z={z_rot:+.1f}')

a = ax[0, 2]
for _, r in st.head(6).iterrows():
    g = tr[tr.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    a.plot(g.FRAME, g.X_px - g.X_px.iloc[0], lw=1.0,
           label=f'T{int(r.TRACK_ID)} n={int(r.n)}')
a.set_xlabel('frame'); a.set_ylabel('x − x(0)  (px, 生座標)')
a.set_title('C. 生座標で最も静止したトラックの x'); a.legend(fontsize=6)

a = ax[1, 2]
a.plot(prof.index, prof.values, lw=1.0, color='tab:brown')
a.set_xlabel('frame'); a.set_ylabel('集積中心までの距離 中央値 (px)')
a.set_title(f'D. 集積中心 ({cx:.0f},{cy:.0f}) px への収束')

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_diagnose.png'))
with open(os.path.join(C.OUT, 'diagnose.json'), 'w') as f:
    json.dump(res, f, indent=2, ensure_ascii=False, default=float)
print('\nSaved out/fig_diagnose.png, out/diagnose.json')
