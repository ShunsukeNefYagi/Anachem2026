"""
s5_compare.py — 3 つの座標系（raw / vib / full）と過去解析の比較

過去解析: data/linking/legacy/track_summary_traj.csv（2026-08-30, 補正なし, 旧 XML 9,196 spots）
出力: out/comparison.csv, out/comparison.json, out/fig_compare.png
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

VAR = ['raw', 'vib', 'full']
tk = {v: pd.read_csv(os.path.join(C.OUT, f'tracks_{v}.csv')) for v in VAR}
sm = {v: pd.read_csv(os.path.join(C.OUT, f'track_summary_{v}.csv')) for v in VAR}

# ── 全体像 ──────────────────────────────────────────────────────────────
print(f'{"variant":<8}{"tracks>=3":>10}{"assigned":>10}{"SPAN=720":>10}'
      f'{"step med (µm)":>15}{"D (px²/min)":>13}')
overall = {}
for v in VAR:
    t = tk[v]
    steps, d2 = [], []
    for tid, g in t.groupby('TRACK_ID'):
        g = g.sort_values('FRAME')
        f = g.FRAME.values
        dx = np.diff(g.X_px.values); dy = np.diff(g.Y_px.values)
        k = np.diff(f) == 1
        steps.append(np.hypot(dx[k], dy[k]) * C.PIXEL_SIZE)
        d2.append((dx[k] ** 2 + dy[k] ** 2))
    steps = np.concatenate(steps); d2 = np.concatenate(d2)
    overall[v] = dict(n_tracks=int(sm[v].TRACK_ID.nunique()), n_spots=int(len(t)),
                      n_full_span=int((sm[v].SPAN == C.N_FRAMES).sum()),
                      step_median_um=float(np.median(steps)),
                      step_mean_um=float(steps.mean()),
                      D_px2_per_min=float(d2.mean() / 4))
    o = overall[v]
    print(f'{v:<8}{o["n_tracks"]:>10}{o["n_spots"]:>10}{o["n_full_span"]:>10}'
          f'{o["step_median_um"]:>15.4f}{o["D_px2_per_min"]:>13.2f}')

for v in ('vib', 'full'):
    print(f'  → {v}: D は raw の {overall[v]["D_px2_per_min"]/overall["raw"]["D_px2_per_min"]:.3f} 倍 '
          f'({100*(overall[v]["D_px2_per_min"]/overall["raw"]["D_px2_per_min"]-1):+.1f} %)')

# ── 長寿命トラック ───────────────────────────────────────────────────────
long_ids = sorted(sm['raw'][sm['raw'].DETECTIONS >= 360].TRACK_ID)
rows = []
for tid in long_ids:
    r = dict(TRACK_ID=tid)
    for v in VAR:
        g = tk[v][tk[v].TRACK_ID == tid].sort_values('FRAME')
        if not len(g):
            continue
        dx, dy = np.diff(g.X_px.values), np.diff(g.Y_px.values)
        k = np.diff(g.FRAME.values) == 1
        r[f'n_{v}'] = len(g)
        r[f'step_{v}_um'] = float(np.hypot(dx[k], dy[k]).mean() * C.PIXEL_SIZE)
        r[f'D_{v}'] = float((dx[k] ** 2 + dy[k] ** 2).mean() / 4)
        r[f'net_{v}_um'] = float(np.hypot(g.X_px.iloc[-1] - g.X_px.iloc[0],
                                          g.Y_px.iloc[-1] - g.Y_px.iloc[0]) * C.PIXEL_SIZE)
    rows.append(r)
cmp_long = pd.DataFrame(rows)
cmp_long['D_vib/raw']  = cmp_long.D_vib / cmp_long.D_raw
cmp_long['D_full/raw'] = cmp_long.D_full / cmp_long.D_raw
cmp_long.to_csv(os.path.join(C.OUT, 'comparison.csv'), index=False)
print('\n長寿命トラック（検出数 ≥ 360）:')
print(cmp_long.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# ── 過去解析との照合 ─────────────────────────────────────────────────────
old_p = os.path.join(C.RAW, 'legacy', 'track_summary_traj.csv')
old_cmp = None
if os.path.exists(old_p):
    old = pd.read_csv(old_p)
    ol = old[old.DETECTIONS >= 360][['TRACK_ID', 'DETECTIONS', 'STEP_MEAN_UM', 'INTENSITY_CV']]
    print('\n過去解析（2026-08-30, 旧 XML, 補正なし）の長寿命トラック:')
    print(ol.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    old_cmp = ol.to_dict('records')

with open(os.path.join(C.OUT, 'comparison.json'), 'w') as f:
    json.dump(dict(overall=overall, long_tracks=cmp_long.to_dict('records'),
                   previous_analysis=old_cmp), f, indent=2, ensure_ascii=False)

# ── 図 ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
cols = dict(raw='0.45', vib='tab:red', full='tab:blue')

a = ax[0, 0]
for v in VAR:
    t = tk[v]
    s = []
    for tid, g in t.groupby('TRACK_ID'):
        g = g.sort_values('FRAME')
        k = np.diff(g.FRAME.values) == 1
        s.append(np.hypot(np.diff(g.X_px.values)[k], np.diff(g.Y_px.values)[k]) * C.PIXEL_SIZE)
    s = np.concatenate(s)
    a.hist(s, bins=np.linspace(0, 3, 80), histtype='step', lw=1.4, color=cols[v],
           label=f'{v}  中央値 {np.median(s):.3f} µm')
a.set_xlabel('1 分あたりのステップ (µm)'); a.set_ylabel('件数')
a.set_title('ステップ長分布（全トラック）'); a.legend(fontsize=8)

a = ax[0, 1]
w = 0.26; xs = np.arange(len(cmp_long))
for i, v in enumerate(VAR):
    a.bar(xs + (i - 1) * w, cmp_long[f'D_{v}'], w, color=cols[v], label=v)
a.set_xticks(xs); a.set_xticklabels([f'T{int(t)}' for t in cmp_long.TRACK_ID])
a.set_ylabel('D = ⟨|Δr(1)|²⟩/4  (px²/min)')
a.set_title('長寿命トラックの見かけの拡散係数'); a.legend(fontsize=8)

a = ax[0, 2]
d = pd.read_csv(os.path.join(C.OUT, 'drift_components.csv'))
a.plot(d.frame, d.Dx_full_um, lw=1.0, color='tab:blue', label='共通成分 x（全体）')
a.plot(d.frame, d.Dy_full_um, lw=1.0, color='tab:cyan', label='共通成分 y（全体）')
a.plot(d.frame, d.Dx_vib_px * C.PIXEL_SIZE, lw=1.0, color='tab:red', label='除去した振動 x')
a.plot(d.frame, d.Dy_vib_px * C.PIXEL_SIZE, lw=1.0, color='tab:orange', label='除去した振動 y')
a.set_xlabel('frame (min)'); a.set_ylabel('累積 (µm)')
a.set_title('共通成分と、実際に除去した量'); a.legend(fontsize=7)

for j, tid in enumerate(long_ids[:3]):
    a = ax[1, j]
    for v in VAR:
        g = tk[v][tk[v].TRACK_ID == tid].sort_values('FRAME')
        a.plot(g.X_px, g.Y_px, lw=1.0, color=cols[v], alpha=0.85, label=v)
    a.set_aspect('equal'); a.invert_yaxis()
    a.set_xlabel('X (px)'); a.set_ylabel('Y (px)')
    a.set_title(f'Track {int(tid)} の軌跡（3 座標系）'); a.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, 'fig_compare.png'))
print('\nSaved out/comparison.csv, out/comparison.json, out/fig_compare.png')
