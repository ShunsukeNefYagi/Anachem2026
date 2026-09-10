"""two.csv の 2 粒子に DC-MSS (Vega et al. 2018) を適用し、
図と数値結果を出力する。"""
import numpy as np, pandas as pd, os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import load_tracks, DT_MIN
import mss_core as mc
from mss_core import (MSSThresholds, dc_mss, mss_analysis, msd_curve,
                      fit_D_sigma, confinement_radius, CLASS_NAMES, CLASS_COLORS,
                      MIN_SEG, MAX_LAG_CAP)

ROOT = os.path.dirname(os.path.dirname(HERE))     # repository root
FIG = os.path.join(ROOT, 'results', 'dcmss', 'figs')
OUT = os.path.join(ROOT, 'results', 'dcmss')
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': ['Hiragino Sans', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 130, 'savefig.dpi': 200,
    'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 9, 'savefig.bbox': 'tight',
})
TCOL = {5: '#d1495b', 8: '#1b6ca8'}


def cve_localization_error(pos, dt=1.0, R=0.0):
    """Vestergaard et al. (2014) covariance-based estimator。
    sigma^2 = R <dx_n^2> + (2R-1) <dx_n dx_{n+1}>  (R=0: 露光時間無視)
    負値は「局在誤差が運動に比べ検出限界以下」を意味する。"""
    d = np.diff(pos, axis=0)
    out = {}
    v2 = np.mean(d ** 2, axis=0)
    cov1 = np.mean(d[:-1] * d[1:], axis=0)
    D = v2 / (2 * dt) + cov1 / dt
    s2 = R * v2 + (2 * R - 1) * cov1
    out['D_cve'] = float(D.mean())
    out['sigma2'] = float(s2.mean())
    out['sigma_p'] = float(np.sqrt(s2.mean())) if s2.mean() > 0 else np.nan
    return out


# ==========================================================================
def main():
    tracks, prep_report = load_tracks()
    thr = MSSThresholds(os.path.join(OUT, 'mss_thresholds.npz'))
    results, seg_rows = {}, []

    for tid, tk in tracks.items():
        pos = tk['pos']
        res = dc_mss(pos, thr, verbose=True)
        whole = mss_analysis(pos)
        cve = cve_localization_error(pos)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)

        res.update(dict(
            whole_mss=float(whole['mss_slope'][0]),
            whole_alpha2=float(whole['alphas'][0, 2]),
            whole_alphas=whole['alphas'][0],
            whole_D2=float(whole['gen_D'][0, 2]),
            whole_class=thr.classify(float(whole['mss_slope'][0]), len(pos)),
            path_length=float(step.sum()),
            net_disp=float(np.linalg.norm(pos[-1] - pos[0])),
            step_median=float(np.median(step)), step_mean=float(step.mean()),
            cve=cve, conf_radius_whole=confinement_radius(pos),
        ))
        results[tid] = res

        for k, s in enumerate(res['segs']):
            sub = pos[s['start']:s['end']]
            sl = np.linalg.norm(np.diff(sub, axis=0), axis=1).sum()
            net = np.linalg.norm(sub[-1] - sub[0])
            D = s['D']
            row = dict(track=tid, seg=k + 1, start_min=s['start'], end_min=s['end'],
                       dur_min=s['len'], cls=CLASS_NAMES[s['cls']],
                       mss_slope=s['mss_slope'], D=D,
                       conf_radius=s['conf_radius'],
                       straightness=net / sl if sl > 0 else np.nan,
                       v_net=s['v_net'],
                       CE=(np.sqrt(4 * D * DT_MIN) / s['conf_radius']
                           if (D == D and D > 0 and s['conf_radius'] > 0)
                           else np.nan),
                       t_explore=(s['conf_radius'] ** 2 / (4 * D)
                                  if (D == D and D > 0) else np.nan),
                       DE=(s['v_net'] * DT_MIN / np.sqrt(4 * D * DT_MIN)
                           if (D == D and D > 0) else np.nan),
                       p_mis=(thr.p_misclass(s['mss_slope'], s['len'], s['cls'])
                              if s['cls'] >= 0 else np.nan))
            seg_rows.append(row)

    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(os.path.join(OUT, 'segments.csv'), index=False)

    state = {tid: motion_state_stats(results[tid]['segs']) for tid in tracks}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(dict(
            prep=prep_report,
            tracks={str(t): {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                             for k, v in results[t].items()
                             if k not in ('segs', 'segs_initial', 'step1', 'bounds')}
                    for t in results},
            state={str(t): state[t] for t in state}), f, indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # ---- 図 ----
    fig_trajectories(tracks)
    fig_thresholds(thr)
    fig_msd_mss(tracks, results)
    fig_step1(tracks, results)
    fig_classified(tracks, results)
    fig_segment_params(seg_df)
    fig_state_diagram(state)
    fig_qc(tracks, results, seg_df)

    print('\n=== SEGMENTS ===')
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(seg_df.round(4).to_string(index=False))
    print('\n=== STATE STATS ===')
    print(json.dumps(state, indent=2, default=float))
    return results, seg_df, state, thr


# ==========================================================================
def motion_state_stats(segs):
    """Note S5: motion class probability と遷移速度。"""
    T = {c: 0.0 for c in range(4)}
    N = {c: 0 for c in range(4)}
    trans = {}
    for i, s in enumerate(segs):
        c = s['cls']
        if c < 0:
            continue
        T[c] += s['len']; N[c] += 1
        if i + 1 < len(segs) and segs[i + 1]['cls'] >= 0 and segs[i + 1]['cls'] != c:
            trans[(c, segs[i + 1]['cls'])] = trans.get((c, segs[i + 1]['cls']), 0) + 1
    tot = sum(T.values())
    prob = {CLASS_NAMES[c]: (T[c] / tot if tot else 0.0) for c in range(4)}
    mean_dur = {CLASS_NAMES[c]: (T[c] / N[c] if N[c] else None) for c in range(4)}
    rates = {}
    for (a, b), n in trans.items():
        p = n / N[a]
        rates[f'{CLASS_NAMES[a]}->{CLASS_NAMES[b]}'] = p / (T[a] / N[a])
    return dict(probability=prob, n_instances={CLASS_NAMES[c]: N[c] for c in range(4)},
                total_time_min={CLASS_NAMES[c]: T[c] for c in range(4)},
                mean_duration_min=mean_dur, transition_rate_per_min=rates,
                n_switches=sum(trans.values()))


# ==========================================================================
def _scalebar(ax, length, label, loc=(0.05, 0.06)):
    x0, y0 = ax.transLimits.inverted().transform(loc)
    ax.plot([x0, x0 + length], [y0, y0], color='k', lw=2.5, solid_capstyle='butt')
    ax.text(x0 + length / 2, y0, label, ha='center', va='bottom', fontsize=8)


def fig_trajectories(tracks):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2),
                             gridspec_kw={'width_ratios': [1, 1, 1.05]})
    for ax, tid in zip(axes[:2], sorted(tracks)):
        p = tracks[tid]['pos']; t = tracks[tid]['t_min']
        pts = p.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap='viridis', linewidth=1.1)
        lc.set_array(t[:-1] / 60.0)
        ax.add_collection(lc)
        ax.plot(*p[0], 'o', ms=7, mfc='w', mec='k', mew=1.4, zorder=5)
        ax.plot(*p[-1], 's', ms=7, mfc='k', mec='k', zorder=5)
        ax.set_aspect('equal'); ax.autoscale_view()
        ax.margins(0.08)
        ax.set_title(f'TRACK {tid}   (N = {len(p)} frames = 12 h)')
        ax.set_xlabel('X (pixel)'); ax.set_ylabel('Y (pixel)')
        ax.invert_yaxis()
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('経過時間 (h)')
        _scalebar(ax, 50, '50 px')
    ax = axes[2]
    for tid in sorted(tracks):
        p = tracks[tid]['pos']
        ax.plot(p[:, 0], p[:, 1], lw=0.9, color=TCOL[tid], label=f'TRACK {tid}')
        ax.plot(*p[0], 'o', ms=6, mfc='w', mec=TCOL[tid], mew=1.5)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.legend(frameon=False)
    ax.set_title('両粒子の相対位置（同一視野）')
    ax.set_xlabel('X (pixel)'); ax.set_ylabel('Y (pixel)')
    _scalebar(ax, 100, '100 px')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig01_trajectories.png'); plt.close(fig)


def fig_thresholds(thr):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = thr.durations
    for ax, xmax in zip(axes, [150, d[-1]]):
        for c in range(4):
            lo = np.quantile(thr.slopes[:, c, :], 0.01, axis=1)
            hi = np.quantile(thr.slopes[:, c, :], 0.99, axis=1)
            md = np.median(thr.slopes[:, c, :], axis=1)
            ax.fill_between(d, lo, hi, color=CLASS_COLORS[c], alpha=0.22)
            ax.plot(d, md, color=CLASS_COLORS[c], lw=1.4, label=CLASS_NAMES[c])
        for k, lab in enumerate(['immobile | confined', 'confined | free', 'free | directed']):
            ax.plot(d, thr.thr[:, k], 'k--', lw=1.1)
        ax.set_xlim(20, xmax); ax.set_xlabel('セグメント長 (frames)')
        ax.set_ylabel('MSS slope')
        ax.set_xscale('log' if xmax > 200 else 'linear')
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_title('分類閾値（破線）と各運動クラスの MSS slope 分布\n1–99 パーセンタイル帯, 20–150 frames')
    axes[1].set_title('全 duration グリッド（対数軸, 20–760 frames）')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig02_thresholds.png'); plt.close(fig)


def fig_msd_mss(tracks, results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    for tid in sorted(tracks):
        p = tracks[tid]['pos']
        taus, msd, npair = msd_curve(p, max_lag=len(p) // 4)
        ax.loglog(taus, msd, color=TCOL[tid], lw=1.8, label=f'TRACK {tid}')
    for a, st, lab in [(1.0, '-', r'$\propto\tau^{1}$ (Brownian)'),
                       (2.0, ':', r'$\propto\tau^{2}$ (directed)')]:
        t = np.array([1, 180.])
        ax.loglog(t, 90 * (t / t[0]) ** a, 'k', ls=st, lw=1, alpha=.6, label=lab)
    ax.axvspan(1, MAX_LAG_CAP, color='0.85', alpha=.5, zorder=0)
    ax.text(5, 2e4, f'MSS 解析範囲\nτ ≤ {MAX_LAG_CAP} min', fontsize=8, color='0.35')
    ax.set_xlabel('時間ラグ τ (min)'); ax.set_ylabel(r'MSD ($\mathrm{px}^2$)')
    ax.set_title('時間平均 MSD'); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for tid in sorted(tracks):
        p = tracks[tid]['pos']
        taus, msd, _ = msd_curve(p, max_lag=len(p) // 4)
        la = np.gradient(np.log(msd), np.log(taus))
        ax.semilogx(taus, la, color=TCOL[tid], lw=1.6, label=f'TRACK {tid}')
    ax.axhline(1, color='k', ls='-', lw=.9, alpha=.6)
    ax.axhline(2, color='k', ls=':', lw=.9, alpha=.6)
    ax.axvspan(1, MAX_LAG_CAP, color='0.85', alpha=.5, zorder=0)
    ax.set_ylim(-0.2, 2.4)
    ax.set_xlabel('時間ラグ τ (min)'); ax.set_ylabel(r'局所指数 $d\log$MSD$/d\log\tau$')
    ax.set_title('MSD の局所スケーリング指数'); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    m = np.arange(7)
    for tid in sorted(tracks):
        a = results[tid]['whole_alphas']
        ax.plot(m, a, 'o-', color=TCOL[tid], ms=5,
                label=f'TRACK {tid}  $S_{{MSS}}$={results[tid]["whole_mss"]:.3f}')
    ax.plot(m, 0.5 * m, 'k-', lw=1, alpha=.6, label='Brownian (slope 0.5)')
    ax.plot(m, 1.0 * m, 'k:', lw=1, alpha=.6, label='directed (slope 1.0)')
    ax.set_xlabel('モーメント次数 m'); ax.set_ylabel(r'スケーリング指数 $\alpha_m$')
    ax.set_title('Moment Scaling Spectrum（軌跡全体）')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig03_msd_mss.png'); plt.close(fig)


def fig_step1(tracks, results):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.6), sharex=True)
    for j, tid in enumerate(sorted(tracks)):
        info = results[tid]['step1']; b = results[tid]['bounds']
        n = len(info['mpd'])
        ax = axes[0, j]
        ax.plot(np.arange(n), info['mpd'], lw=.7, color='0.65', label='MPD')
        ax.plot(np.arange(n), info['smpd'], lw=1.5, color=TCOL[tid],
                label=f'SMPD (Gauss σ=2)')
        for x in b[1:-1]:
            ax.axvline(x, color='r', lw=1, alpha=.8)
        ax.set_ylabel('MPD (pixel)')
        ax.set_title(f'TRACK {tid} — Step 1: 局所運動記述子と切替点\n'
                     f'初期ピーク {info["n_initial_peaks"]} → KS統合 {info["n_ks_merged"]} '
                     f'→ 最終 {len(b)-1} セグメント', fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax = axes[1, j]
        ax.plot(np.arange(1, n), info['dsmpd'], lw=.8, color='0.4')
        ax.axhline(info['thr'], color='orange', ls='--', lw=1.2,
                   label='上位5% 閾値')
        for x in b[1:-1]:
            ax.axvline(x, color='r', lw=1, alpha=.8)
        ax.set_xlabel('時間 (min)'); ax.set_ylabel(r'$\Delta$SMPD')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig04_step1.png'); plt.close(fig)


def fig_classified(tracks, results):
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 0.42, 1.15], hspace=0.42)
    for j, tid in enumerate(sorted(tracks)):
        p = tracks[tid]['pos']; segs = results[tid]['segs']
        ax = fig.add_subplot(gs[0, j])
        for s in segs:
            sub = p[max(0, s['start'] - 1):s['end']]
            ax.plot(sub[:, 0], sub[:, 1], color=CLASS_COLORS[s['cls']], lw=1.5)
        ax.plot(*p[0], 'o', ms=8, mfc='w', mec='k', mew=1.5, zorder=5)
        ax.plot(*p[-1], 's', ms=8, mfc='k', mec='k', zorder=5)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.margins(.08)
        ax.set_title(f'TRACK {tid} — DC-MSS 分類結果'); ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')
        _scalebar(ax, 50, '50 px')

        axt = fig.add_subplot(gs[1, j])
        for s in segs:
            axt.axvspan(s['start'], s['end'], color=CLASS_COLORS[s['cls']],
                        alpha=.92, lw=0)
            if s['len'] > 35:
                axt.text((s['start'] + s['end']) / 2, .5,
                         f"{CLASS_NAMES[s['cls']][:4]}\n{s['len']}", ha='center',
                         va='center', fontsize=7, color='w', weight='bold')
        axt.set_xlim(0, len(p)); axt.set_yticks([]); axt.grid(False)
        axt.set_xlabel('時間 (min)')
        axt.set_title('運動形態タイムライン', fontsize=9)

        axd = fig.add_subplot(gs[2, j])
        disp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        axd.plot(np.arange(1, len(p)), disp, lw=.7, color='0.45')
        for s in segs:
            axd.axvspan(s['start'], s['end'], color=CLASS_COLORS[s['cls']],
                        alpha=.16, lw=0)
        axd.set_xlim(0, len(p)); axd.set_xlabel('時間 (min)')
        axd.set_ylabel('|Δr| (px/min)')
        axd.set_title('1分間変位', fontsize=9)
    handles = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=4, label=CLASS_NAMES[c])
               for c in [0, 1, 2, 3, -1]]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(.5, -0.02))
    fig.savefig(f'{FIG}/fig05_classified.png'); plt.close(fig)


def fig_segment_params(seg_df):
    d = seg_df[seg_df.cls != 'unclassified']
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    order = [c for c in ['immobile', 'confined', 'free', 'directed']
             if (d.cls == c).any()]
    cmap = {v: CLASS_COLORS[k] for k, v in CLASS_NAMES.items()}
    specs = [('D', r'拡散係数 $D$ (px$^2$/min)', True),
             ('conf_radius', '位置広がり半径 R (px)', True),
             ('dur_min', 'セグメント継続時間 (min)', False),
             ('straightness', '直線性 (正味変位/経路長)', False)]
    for ax, (col, lab, logy) in zip(axes, specs):
        for i, c in enumerate(order):
            sub = d[d.cls == c][col].dropna()
            if not len(sub):
                continue
            jit = (np.random.default_rng(i).random(len(sub)) - .5) * .3
            ax.scatter(np.full(len(sub), i) + jit, sub, s=np.clip(
                d[d.cls == c].dur_min.values[:len(sub)] * .9, 12, 220),
                color=cmap[c], alpha=.7, edgecolor='k', linewidth=.5)
            ax.hlines(np.median(sub), i - .28, i + .28, color='k', lw=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylabel(lab)
        if logy:
            ax.set_yscale('log')
    axes[0].set_title('（マーカー面積 ∝ セグメント長）', fontsize=8, loc='left')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig06_segment_params.png'); plt.close(fig)


def fig_state_diagram(state):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    coords = {'immobile': (0, 0), 'confined': (1, 0.9),
              'free': (2, 0), 'directed': (3, 0.9)}
    for ax, tid in zip(axes, sorted(state)):
        st = state[tid]
        pr = st['probability']
        for c, (x, y) in coords.items():
            p = pr.get(c, 0)
            if p <= 0:
                continue
            r = .13 + .40 * np.sqrt(p)
            ax.add_patch(Circle((x, y), r, color=CLASS_COLORS[
                {v: k for k, v in CLASS_NAMES.items()}[c]], alpha=.85, zorder=3))
            ax.text(x, y, f'{p*100:.0f}%', ha='center', va='center',
                    color='w', fontsize=10, weight='bold', zorder=4)
            ax.text(x, y - r - .13, c, ha='center', va='top', fontsize=8.5)
        for key, rate in st['transition_rate_per_min'].items():
            a, b = key.split('->')
            (x1, y1), (x2, y2) = coords[a], coords[b]
            ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2), connectionstyle='arc3,rad=0.28',
                arrowstyle='-|>', mutation_scale=13, lw=1 + 220 * rate,
                color='0.3', shrinkA=26, shrinkB=26, zorder=2))
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + .30, f'{rate:.4f}',
                    ha='center', fontsize=7.5, color='0.25')
        ax.set_xlim(-.7, 3.7); ax.set_ylim(-1.0, 1.9)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'TRACK {tid} — 運動状態の占有率と遷移速度 '
                     f'(min$^{{-1}}$)\n切替回数 = {st["n_switches"]}', fontsize=9.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig07_state_diagram.png'); plt.close(fig)


def fig_qc(tracks, results, seg_df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for tid in sorted(tracks):
        step = np.linalg.norm(np.diff(tracks[tid]['pos'], axis=0), axis=1)
        ax.hist(step, bins=60, histtype='step', lw=1.6, color=TCOL[tid],
                density=True, label=f'TRACK {tid} (中央値 {np.median(step):.1f})')
    ax.set_xlabel('1分間変位 |Δr| (px)'); ax.set_ylabel('確率密度')
    ax.set_title('ステップサイズ分布'); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    d = seg_df.dropna(subset=['CE']).reset_index(drop=True)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.scatter(r['CE'], i, s=60, color=CLASS_COLORS[
            {v: k for k, v in CLASS_NAMES.items()}[r['cls']]], edgecolor='k', lw=.5)
        ax.text(r['CE'] + .02, i, f"T{int(r['track'])}-s{int(r['seg'])}", fontsize=7,
                va='center')
    ax.axvline(0.7, color='k', ls='--', lw=1)
    ax.text(0.71, -0.6, '閾値導出時の参照値 CE=0.7', fontsize=7.5, rotation=90,
            va='bottom')
    ax.set_xlabel(r'confinement extent  CE $=\sqrt{4D\Delta t}/R$')
    ax.set_yticks([]); ax.set_title('閉じ込め度の事後 QC')
    ax.set_ylim(-1, max(1, len(d)))

    ax = axes[2]
    labels, vals = [], []
    for tid in sorted(tracks):
        c = results[tid]['cve']
        labels.append(f'TRACK {tid}')
        vals.append(c['sigma2'])
    ax.bar(labels, vals, color=[TCOL[t] for t in sorted(tracks)], width=.5)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel(r'CVE 推定 $\sigma_p^2$ (px$^2$)')
    ax.set_title('局在誤差の共分散推定\n（負 = 運動に対し検出限界以下）', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig08_qc.png'); plt.close(fig)


if __name__ == '__main__':
    main()
