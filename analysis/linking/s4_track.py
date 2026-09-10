"""
s4_track.py — 粒子追跡（linking）

アルゴリズム・パラメータ（実装は common.track()）:
  δmax = 3.0 µm, (w_d, w_i) = (0.7, 0.3), Rmax = 2.0, Gmax = 3, γ = 0.3, 最小検出数 3
  コスト C = w_d·d/δmax + w_i·(r−1)/(Rmax−1)、検出数の多いトラック優先の貪欲マッチング

使い方:  python3 s4_track.py <variant> [--bootstrap]
    variant = raw | vib | full
      raw  : 無補正座標
      vib  : 機械振動（高周波共通成分）のみ除去   ← 推奨
  full : 共通成分すべて除去（アンサンブル相対座標）
  common: full と同じ補正を、今回の追加解析用の明示的な名前で出力

出力（すべて results/linking/ 以下）:
      tracks_<variant>.csv, track_summary_<variant>.csv,
      uncertainty_long_<variant>.csv,
      tracks_overview_<variant>.png, trajectories_<variant>.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import common as C

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

VARIANT   = sys.argv[1] if len(sys.argv) > 1 else 'vib'
BOOTSTRAP = '--bootstrap' in sys.argv
assert VARIANT in ('raw', 'vib', 'full', 'common')

BOOTSTRAP_DISTS = [round(C.MAX_DIST_UM * f, 2)
                   for f in [0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45]]
WEIGHT_VARIANTS = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]
RATIO_VARIANTS  = [1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 5.0, np.inf]

# ── 座標の読み込み ───────────────────────────────────────────────────────
sv = pd.read_csv(os.path.join(C.OUT, 'spots_variants.csv'))
xc, yc = ({'raw': ('X_px', 'Y_px'),
           'vib': ('X_px_vib', 'Y_px_vib'),
           'full': ('X_px_full', 'Y_px_full'),
           'common': ('X_px_full', 'Y_px_full')})[VARIANT]
df = sv[['ID', 'FRAME', 'INTENSITY']].copy()
df['X_px'] = sv[xc]; df['Y_px'] = sv[yc]
if VARIANT == 'common':
    # full はFFT再構成由来の任意定数を持つ。common 出力では frame 0 の
    # 補正量を0にして、生座標と同じ原点から開始させる。
    dc = pd.read_csv(os.path.join(C.OUT, 'drift_components.csv'))
    df['X_px'] += float(dc.Dx_full_px.iloc[0])
    df['Y_px'] += float(dc.Dy_full_px.iloc[0])
df['POSITION_X'] = df.X_px * C.PIXEL_SIZE
df['POSITION_Y'] = df.Y_px * C.PIXEL_SIZE
df = df.sort_values(['FRAME', 'ID']).reset_index(drop=True)
spot_lk = df.set_index('ID')
N = C.N_FRAMES

print(f'=== variant "{VARIANT}"  ({xc}, {yc}) ===')
print(f'spots {len(df)}  frames {df.FRAME.min()}-{df.FRAME.max()}')
print(f'intensity mean {df.INTENSITY.mean():.1f}  median {df.INTENSITY.median():.1f}\n',
      flush=True)


def spot_set(fin):
    s = set()
    for t in fin.values():
        if len(t['spots']) >= C.MIN_DETECTIONS:
            for fr, sid in t['spots']:
                s.add((fr, int(sid)))
    return s


# ── 名目値での追跡 ───────────────────────────────────────────────────────
print('nominal tracking …', flush=True)
finished, link_info = C.track(df, record_confidence=True)
good  = C.good_tracks(finished)
spans = {tid: (sorted(t['spots'])[-1][0] - sorted(t['spots'])[0][0] + 1)
         for tid, t in good.items()}
n_full = sum(1 for s in spans.values() if s == N)
print(f'  tracks >= {C.MIN_DETECTIONS}: {len(good)}   SPAN=720: {n_full}   '
      f'assigned spots: {sum(len(t["spots"]) for t in good.values())}', flush=True)

# ── Bootstrap A/B/C ─────────────────────────────────────────────────────
if BOOTSTRAP:
    print(f'Bootstrap A: {len(BOOTSTRAP_DISTS)} max_dist …', flush=True)
    bs_dist = [spot_set(C.track(df, max_dist=b)) for b in BOOTSTRAP_DISTS]
    print(f'Bootstrap B: {len(WEIGHT_VARIANTS)} weights …', flush=True)
    bs_weight = [spot_set(C.track(df, dist_w=a, int_w=b)) for a, b in WEIGHT_VARIANTS]
    print(f'Bootstrap C: {len(RATIO_VARIANTS)} Rmax …', flush=True)
    ratio_fin = [C.track(df, max_int_ratio=r) for r in RATIO_VARIANTS]
    bs_ratio  = [spot_set(f) for f in ratio_fin]
    counts = [len(C.good_tracks(f)) for f in ratio_fin]
    print('\n  Rmax      tracks>=3')
    for r, n in zip(RATIO_VARIANTS, counts):
        print(f'  {("inf" if np.isinf(r) else f"{r:.1f}"):<9} {n}')
else:
    bs_dist = bs_weight = bs_ratio = []
    counts = None

BA, BW, BR = len(bs_dist), len(bs_weight), len(bs_ratio)
all_bs = bs_dist + bs_weight + bs_ratio

# ── スポット単位の出力 ───────────────────────────────────────────────────
print('\nper-spot metrics …', flush=True)
rows = []
for tid, t in good.items():
    ss = sorted(t['spots'], key=lambda x: x[0])
    for frame, sid in ss:
        sid = int(sid); r = spot_lk.loc[sid]; key = (frame, sid)
        if key in link_info:
            bc, sc = link_info[key]
            conf = max(0.0, min(1.0, 1.0 - bc / sc)) if sc < float('inf') else 1.0
        else:
            bc, conf = np.nan, np.nan
        rec = dict(TRACK_ID=tid, DETECTIONS=len(ss), SPAN=spans[tid],
                   SPOT_ID=sid, FRAME=frame,
                   X_um=r.POSITION_X, Y_um=r.POSITION_Y,
                   X_px=r.X_px, Y_px=r.Y_px, INTENSITY=r.INTENSITY,
                   LINK_COST=bc, LINK_CONFIDENCE=conf)
        if all_bs:
            rec.update(DIST_STABILITY=sum(1 for b in bs_dist if key in b) / BA,
                       WEIGHT_STABILITY=sum(1 for b in bs_weight if key in b) / BW,
                       RATIO_STABILITY=sum(1 for b in bs_ratio if key in b) / BR,
                       COMBINED_STABILITY=sum(1 for b in all_bs if key in b) / len(all_bs))
        rows.append(rec)
out = pd.DataFrame(rows).sort_values(['TRACK_ID', 'FRAME']).reset_index(drop=True)

# ── トラック単位の統計 ───────────────────────────────────────────────────
rows = []
for tid, g in out.groupby('TRACK_ID'):
    g = g.sort_values('FRAME')
    xs, ys, ints = g.X_um.values, g.Y_um.values, g.INTENSITY.values
    fr = g.FRAME.values
    st = np.hypot(np.diff(xs), np.diff(ys))
    ir = np.where(ints[:-1] > 0,
                  np.maximum(ints[1:], ints[:-1]) / (np.minimum(ints[1:], ints[:-1]) + 1e-9),
                  np.nan)
    d = dict(TRACK_ID=tid, DETECTIONS=len(g), SPAN=int(g.SPAN.iloc[0]),
             FRAME_START=int(fr[0]), FRAME_END=int(fr[-1]),
             STEP_MEAN_UM=st.mean() if len(st) else np.nan,
             STEP_STD_UM=st.std() if len(st) else np.nan,
             TOTAL_DIST_UM=float(st.sum()) if len(st) else np.nan,
             NET_DISP_UM=float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])),
             INTENSITY_MEAN=float(ints.mean()),
             INTENSITY_CV=ints.std() / (ints.mean() + 1e-9),
             INT_RATIO_MEAN=np.nanmean(ir) if len(ir) else np.nan,
             INT_RATIO_MAX=np.nanmax(ir) if len(ir) else np.nan,
             GAP_RATE=1.0 - len(g) / g.SPAN.iloc[0],
             MEAN_LINK_CONFIDENCE=g.LINK_CONFIDENCE.mean(),
             MIN_LINK_CONFIDENCE=g.LINK_CONFIDENCE.min())
    for k in ('DIST', 'WEIGHT', 'RATIO', 'COMBINED'):
        if f'{k}_STABILITY' in g:
            d[f'MEAN_{k}_STABILITY'] = g[f'{k}_STABILITY'].mean()
            d[f'MIN_{k}_STABILITY']  = g[f'{k}_STABILITY'].min()
    rows.append(d)
summary = pd.DataFrame(rows)

fp = {}
for r in out.itertuples():
    fp.setdefault(int(r.FRAME), {})[int(r.TRACK_ID)] = (r.X_um, r.Y_um)
summary['CONFUSION_RISK'] = summary.TRACK_ID.map({
    tid: sum(1 for r in g.itertuples()
             if any(np.hypot(ox - r.X_um, oy - r.Y_um) < 2 * C.MAX_DIST_UM
                    for o, (ox, oy) in fp.get(int(r.FRAME), {}).items() if o != tid)) / len(g)
    for tid, g in out.groupby('TRACK_ID')})
summary = summary.sort_values(['DETECTIONS', 'SPAN'], ascending=False).reset_index(drop=True)

out.to_csv(os.path.join(C.OUT, f'tracks_{VARIANT}.csv'), index=False)
summary.to_csv(os.path.join(C.OUT, f'track_summary_{VARIANT}.csv'), index=False)
print(f'Saved out/tracks_{VARIANT}.csv ({len(out)} rows), '
      f'out/track_summary_{VARIANT}.csv ({len(summary)} tracks)')

LONG_MIN = N // 2
long_tr = summary[summary.DETECTIONS >= LONG_MIN].copy()
cols = ['TRACK_ID', 'DETECTIONS', 'SPAN', 'FRAME_START', 'FRAME_END', 'STEP_MEAN_UM',
        'INTENSITY_CV', 'GAP_RATE', 'MEAN_LINK_CONFIDENCE', 'MIN_LINK_CONFIDENCE']
if 'MEAN_COMBINED_STABILITY' in summary:
    cols += ['MEAN_COMBINED_STABILITY', 'MIN_COMBINED_STABILITY']
cols += ['CONFUSION_RISK']
print(f'\nLong tracks (DETECTIONS >= {LONG_MIN}): {len(long_tr)}')
print((long_tr if len(long_tr) else summary.head(10))[cols].to_string(
    index=False, float_format=lambda v: f'{v:.3f}'))

top = (long_tr if len(long_tr) else summary.head(5)).sort_values('TRACK_ID')
top.to_csv(os.path.join(C.OUT, f'uncertainty_long_{VARIANT}.csv'), index=False)

# ── 図 ──────────────────────────────────────────────────────────────────
colors = [matplotlib.colormaps['tab10'](i % 10) for i in range(len(top))]
np_ = 3 if all_bs else 2
fig, axes = plt.subplots(np_, 3, figsize=(18, 5 * np_), squeeze=False)

ax = axes[0, 0]
others = summary[~summary.TRACK_ID.isin(top.TRACK_ID)]
pal = cm.Greys(np.linspace(0.2, 0.6, max(len(others), 1)))
for i, (_, r) in enumerate(others.iterrows()):
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    ax.plot(s.X_um, s.Y_um, '-', lw=0.4, alpha=0.4, color=pal[i])
for ci, (_, r) in enumerate(top.iterrows()):
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    ax.plot(s.X_um, s.Y_um, '-', lw=1.5, color=colors[ci],
            label=f'T{int(r.TRACK_ID)} (n={int(r.DETECTIONS)})')
    ax.plot(s.X_um.iloc[0], s.Y_um.iloc[0], 'o', ms=5, color=colors[ci])
ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)'); ax.set_aspect('equal'); ax.invert_yaxis()
ax.set_title(f'Trajectories [{VARIANT}]'); ax.legend(fontsize=7)

ax = axes[0, 1]
for ci, (_, r) in enumerate(top.iterrows()):
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    ax.plot(s.FRAME, s.INTENSITY, lw=0.8, color=colors[ci], label=f'T{int(r.TRACK_ID)}')
ax.set_xlabel('Frame'); ax.set_ylabel('bg-corrected intensity')
ax.set_title('Intensity over time'); ax.legend(fontsize=7)

ax = axes[0, 2]
for ci, (_, r) in enumerate(top.iterrows()):
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    ax.plot(s.FRAME, s.LINK_CONFIDENCE, lw=0.7, alpha=0.85, color=colors[ci])
ax.axhline(0.5, color='k', ls=':', lw=0.8)
ax.set_ylim(-0.05, 1.05); ax.set_xlabel('Frame'); ax.set_ylabel('Link confidence')
ax.set_title('Link confidence')

if all_bs:
    for (rr, cc), key, B, ttl in [((1, 0), 'DIST_STABILITY', BA, f'Bootstrap A: δmax (n={BA})'),
                                  ((1, 1), 'WEIGHT_STABILITY', BW, f'Bootstrap B: weights (n={BW})'),
                                  ((1, 2), 'RATIO_STABILITY', BR, f'Bootstrap C: Rmax (n={BR})'),
                                  ((2, 0), 'COMBINED_STABILITY', len(all_bs),
                                   f'Combined (n={len(all_bs)})')]:
        ax = axes[rr, cc]
        for ci, (_, r) in enumerate(top.iterrows()):
            s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
            ax.plot(s.FRAME, s[key], lw=0.7, alpha=0.85, color=colors[ci])
        ax.axhline(0.8, color='k', ls=':', lw=0.8); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Frame'); ax.set_ylabel(f'Stability (B={B})'); ax.set_title(ttl)
    ax = axes[2, 1]
    lab = ['inf' if np.isinf(r) else f'{r:.1f}' for r in RATIO_VARIANTS]
    ax.plot(range(len(lab)), counts, 'o-', color='steelblue')
    ax.axhline(len(good), color='red', ls='--', lw=1.0, label=f'nominal ({len(good)})')
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
    ax.set_xlabel('Rmax'); ax.set_ylabel('tracks >= 3'); ax.legend(fontsize=8)
    ax.set_title('Track count vs Rmax')
    ax = axes[2, 2]
else:
    ax = axes[1, 0]
ax.hist(summary.DETECTIONS, bins=40, color='slategray'); ax.set_yscale('log')
ax.set_xlabel('Detections per track'); ax.set_ylabel('Tracks (log)')
ax.set_title(f'Track length distribution (n={len(summary)})')
if not all_bs:
    axes[1, 1].axis('off'); axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(C.OUT, f'tracks_overview_{VARIANT}.png'), dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
for _, r in summary.iterrows():
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    ax.plot(s.X_px, s.Y_px, '-', lw=0.5, alpha=0.5, color='0.5')
for ci, (_, r) in enumerate(top.iterrows()):
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    ax.plot(s.X_px, s.Y_px, '-', lw=1.6, color=matplotlib.colormaps['tab10'](ci % 10),
            label=f'T{int(r.TRACK_ID)} (n={int(r.DETECTIONS)})')
    ax.plot(s.X_px.iloc[0], s.Y_px.iloc[0], 'o', ms=6,
            color=matplotlib.colormaps['tab10'](ci % 10))
ax.set_xlim(0, C.IMG_SIZE); ax.set_ylim(C.IMG_SIZE, 0); ax.set_aspect('equal')
ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
ax.set_title(f'All tracks (grey, n={len(summary)}) + long tracks [{VARIANT}]')
ax.legend(fontsize=7, ncol=2)

ax = axs[1]
sc = None
for ci, (_, r) in enumerate(top.iterrows()):
    s = out[out.TRACK_ID == r.TRACK_ID].sort_values('FRAME')
    sc = ax.scatter(s.X_px, s.Y_px, c=s.FRAME, cmap='viridis', s=6, vmin=0, vmax=N - 1)
    ax.plot(s.X_px, s.Y_px, '-', lw=0.6, color='0.7', zorder=0)
if sc is not None:
    plt.colorbar(sc, ax=ax, label='Frame')
ax.set_xlim(0, C.IMG_SIZE); ax.set_ylim(C.IMG_SIZE, 0); ax.set_aspect('equal')
ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)'); ax.set_title('Long tracks colored by frame')
plt.tight_layout()
plt.savefig(os.path.join(C.OUT, f'trajectories_{VARIANT}.png'), dpi=150)
print(f'Saved out/tracks_overview_{VARIANT}.png, out/trajectories_{VARIANT}.png')
