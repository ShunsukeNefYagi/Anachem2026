"""MSS による分類の中身と、方向相関による独立評価を並べた図。

DC-MSS は MSS 曲線の傾き S_MSS だけで free / directed を決めている。
その S_MSS の値そのものと、分類に使った継続時間依存の閾値、
そして MSS が捨てている方向情報 (VACF/DACF) を同じ図に並べる。

  A. MSS スペクトル alpha_m vs m (これが「MSS」そのもの)
  B. S_MSS と継続時間依存の分類閾値
  C. 方向相関 C(1), Cu(1) と free 帰無の 95% 区間

使い方: python3 mss_vs_direction.py [paper|recal]
"""
import json
import os
import sys

import numpy as np

import slide_common as S
from mss_core import MSSThresholds, mss_analysis, MOMENT_ORDERS   # noqa: E402
from seg_direction import (analyse, SEGMENTS as _unused)          # noqa: E402,F401

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'recal'

# 分割ごとに、その分割を作ったときの MSS 設定と閾値テーブルを使う
CONF = {
    # 'paper' は derive_thresholds.py を先に走らせて生成する必要がある（同梱していない）
    'paper': dict(thr=os.path.join(S.ROOT, 'results', 'dcmss',
                                   'mss_thresholds.npz'),
                  lag_frac=4, lag_cap=30,
                  note='論文既定 (tau_max = N/4, sigma_p/Sigma_D = 0)'),
    'recal': dict(thr=os.path.join(S.ROOT, 'data', 'calibration',
                                   'thr_sig0.14_lf3.npz'),
                  lag_frac=3, lag_cap=30,
                  note='再較正 (tau_max = N/3, sigma_p/Sigma_D = 0.14)'),
}[VARIANT]

if not os.path.exists(CONF['thr']):
    raise SystemExit(
        f"Threshold table not found: {CONF['thr']}\n"
        "For VARIANT='paper', run analysis/dcmss/derive_thresholds.py first.")

SEGS = {8: S.VARIANTS[VARIANT], 5: S.SEGMENTS_T5}
CLASS_OF = {0: 'immobile', 1: 'confined', 2: 'free', 3: 'directed'}


def mss_of_segment(pos, i0, i1, thr):
    r = mss_analysis(np.asarray(pos, float)[i0:i1],
                     lag_cap=CONF['lag_cap'], lag_frac=CONF['lag_frac'])
    s = float(r['mss_slope'][0])
    L = i1 - i0
    cls = thr.classify(s, L)
    return dict(t0=i0, t1=i1, len=L, mss_slope=s,
                alphas=[float(a) for a in r['alphas'][0]],
                gen_D2=float(r['gen_D'][0, 2]),
                max_lag=int(r['max_lag']),
                cls=int(cls), cls_name=CLASS_OF.get(cls, 'unclassified'),
                thresholds=[float(t) for t in thr.thresholds_at(L)],
                p_mis=float(thr.p_misclass(s, L, cls)))


def main():
    tracks, _ = S.get_tracks()
    thr = MSSThresholds(CONF['thr'])

    rows = []
    for tid in (8, 5):
        for i0, i1, cls_report in SEGS[tid]:
            m = mss_of_segment(tracks[tid]['pos'], i0, i1, thr)
            d = analyse(tracks[tid]['pos'], i0, i1)
            m.update(track=tid, cls_report=cls_report,
                     C1=d['vacf'][1], Cu1=d['dacf'][1],
                     C1_free_lo=d['cal']['free']['C1']['q025'],
                     C1_free_hi=d['cal']['free']['C1']['q975'],
                     Cu1_free_lo=d['cal']['free']['Cu1']['q025'],
                     Cu1_free_hi=d['cal']['free']['Cu1']['q975'],
                     C_const_drift=d['C_const_drift'],
                     DE=d['DE'], drift_speed=d['drift']['speed'],
                     forward_frac=d['angles']['forward_frac'])
            rows.append(m)

    with open(f'{S.OUT}/mss_vs_direction_{VARIANT}.json', 'w') as f:
        json.dump(rows, f, indent=2)

    # ---------------- 表 ----------------
    print('=' * 104)
    print(f'MSS による分類と方向相関  [分割: {VARIANT}]   {CONF["note"]}')
    print('=' * 104)
    print(f"{'区間':>20}{'N':>5}{'tau_max':>8}{'S_MSS':>8}"
          f"{'閾値 c|f':>10}{'閾値 f|d':>10}{'分類':>11}{'p_mis':>9}"
          f"{'C(1)':>8}{'Cu(1)':>8}{'DE':>7}")
    for r in rows:
        lab = f"T{r['track']} {r['t0']}-{r['t1']}"
        print(f"{lab:>20}{r['len']:>5}{r['max_lag']:>8}{r['mss_slope']:>8.3f}"
              f"{r['thresholds'][1]:>10.3f}{r['thresholds'][2]:>10.3f}"
              f"{r['cls_name']:>11}{r['p_mis']:>9.4f}"
              f"{r['C1']:>8.3f}{r['Cu1']:>8.3f}{r['DE']:>7.3f}")

    print('\n  分類は S_MSS が「閾値 f|d」を超えたかどうかだけで決まる。')
    for r in rows:
        if r['track'] != 8:
            continue
        margin = r['mss_slope'] - r['thresholds'][2]
        print(f"   T8 {r['t0']:>3}-{r['t1']:<3}: S_MSS - 閾値(f|d) = {margin:+.3f}"
              f"   -> {r['cls_name']}")

    # ---------------- 図 ----------------
    plt = S.slide_style()
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.4))
    m = MOMENT_ORDERS.astype(float)
    t8 = [r for r in rows if r['track'] == 8]
    t5 = [r for r in rows if r['track'] == 5]

    def style(r, k_free=[0]):
        if r['track'] == 5:
            return '#9AA0A6', ':', 'TRACK 5  free  0–12 h'
        ls = ['-', '--', '-.'][k_free[0] % 3]
        lab = (f"{r['cls_report']}  {r['t0'] / 60:.1f}–{r['t1'] / 60:.1f} h"
               f"   S$_{{MSS}}$={r['mss_slope']:.3f}")
        col = S.COL_BLUE if r['cls_report'] == 'directed' else S.COL_RED
        k_free[0] += 1
        return col, ls, lab

    styles = [style(r) for r in rows]

    # A. MSS スペクトル
    ax = axes[0]
    ax.plot(m, m * 0.5, color=S.COL_WHITE, lw=1.2, alpha=0.45)
    ax.plot(m, m * 1.0, color=S.COL_WHITE, lw=1.2, alpha=0.45, ls='--')
    ax.text(6.05, 3.05, 'free\n(slope 0.5)', color=S.COL_WHITE, alpha=0.7,
            fontsize=11.5, ha='right', va='top')
    ax.text(6.05, 5.85, 'directed\n(slope 1)', color=S.COL_WHITE, alpha=0.7,
            fontsize=11.5, ha='right', va='top')
    for r, (col, ls, lab) in zip(rows, styles):
        ax.plot(m, r['alphas'], color=col, ls=ls, lw=2.2, marker='o', ms=4.5,
                label=lab)
    ax.set_xlabel('Moment order  m')
    ax.set_ylabel(r'Scaling exponent  $\alpha_m$')
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 6.6)
    ax.legend(fontsize=10.5, loc='upper left', handlelength=1.8)

    # B. S_MSS と継続時間依存閾値
    ax = axes[1]
    dur = np.arange(20, 740, 5)
    tt = np.array([thr.thresholds_at(d) for d in dur])
    ax.plot(dur, tt[:, 2], color=S.COL_BLUE, lw=1.8, alpha=0.85)
    ax.plot(dur, tt[:, 1], color=S.COL_WHITE, lw=1.8, alpha=0.55)
    ax.fill_between(dur, tt[:, 2], 1.05, color=S.COL_BLUE, alpha=0.13, lw=0)
    ax.text(560, 0.72, 'directed', color=S.COL_BLUE, fontsize=13, ha='right')
    ax.text(560, 0.47, 'free', color=S.COL_WHITE, fontsize=13, ha='right',
            alpha=0.8)
    ax.text(560, 0.16, 'confined', color=S.COL_WHITE, fontsize=13, ha='right',
            alpha=0.55)
    for r, (col, ls, lab) in zip(rows, styles):
        ax.plot(r['len'], r['mss_slope'], marker='o', ms=13, color=col,
                mec=S.COL_WHITE, mew=1.6, ls='none', zorder=5)
        dy = 15 if r['mss_slope'] >= 0.60 else -26
        ax.annotate(f"{r['t0']}–{r['t1']}", (r['len'], r['mss_slope']),
                    textcoords='offset points', xytext=(0, dy),
                    color=S.COL_WHITE, fontsize=10.5, ha='center')
    ax.set_xlabel('Segment duration (min)')
    ax.set_ylabel(r'MSS slope  S$_{MSS}$')
    ax.set_xlim(0, 760)
    ax.set_ylim(0, 1.0)

    # C. 方向相関
    ax = axes[2]
    x = np.arange(len(rows))
    for i, (r, (col, ls, lab)) in enumerate(zip(rows, styles)):
        ax.bar(i, r['Cu1'], width=0.60, color=col, alpha=0.95, zorder=2)
        ax.fill_between([i - 0.38, i + 0.38], r['Cu1_free_lo'],
                        r['Cu1_free_hi'], color='#B8BCC2', alpha=0.42, lw=0,
                        zorder=4)
        for y in (r['Cu1_free_lo'], r['Cu1_free_hi']):
            ax.plot([i - 0.38, i + 0.38], [y] * 2, color=S.COL_WHITE, lw=1.3,
                    alpha=0.8, zorder=5)
        ax.plot([i - 0.38, i + 0.38], [r['C_const_drift']] * 2,
                color='#FFD400', lw=2.2, ls=(0, (2, 1.5)), zorder=6)
    ax.axhline(0, color=S.COL_WHITE, lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['t0']}–{r['t1']}" if r['track'] == 8
                        else 'T5\n0–720' for r in rows], fontsize=11)
    ax.set_xlabel('Segment (min)')
    ax.set_ylabel(r'Directional correlation  C$_u$(1)')
    ax.set_ylim(-0.36, 0.72)
    ax.plot([], [], color='#B8BCC2', lw=8, alpha=0.42,
            label='free null 95% CI')
    ax.plot([], [], color='#FFD400', lw=2, ls=(0, (2, 1.5)),
            label='constant-drift prediction')
    ax.legend(fontsize=11, loc='upper right')

    fig.tight_layout()
    name = f'{S.FIGS}/S4_mss_vs_direction_{VARIANT}.png'
    fig.savefig(name)
    fig.savefig(name.replace('.png', '.pdf'))
    plt.close(fig)
    print(f'\n  {name}')


if __name__ == '__main__':
    main()
