"""DC-MSS で分割した各区間における方向相関解析。

q7_direction.py は「軌跡全体」と「固定 60 分窓」に対してのみ方向相関を
計算する。ここでは再較正済み DC-MSS が返した区間
(analysis/dcmss/apply_recalibrated.py の分割) を単位として同じ解析を行う。

計算する量 (すべて区間ごと)
  A. 速度自己相関 VACF  C(n) = <v(t).v(t+n)> / <|v|^2>
  B. 方向自己相関 DACF  Cu(n) = <u(t).u(t+n)>       (大きさに鈍感)
  C. turning angle の前方散乱率と <cos theta>
  D. 平均速度ベクトル (ドリフト) とその有意性
  E. 較正: 同じ N・同じ「ドリフトを除いた D」をもつ free / confined / directed
     シミュレーションでの 95% 区間
  F. 一定ドリフト仮説の検定
     ドリフト v が一定なら n >= 1 で C(n) = DE^2/(1+DE^2) の *平坦* な値をとる。
     観測 C(n) がこれを超えて減衰するなら、一定ドリフトでは説明できない
     持続性 (方向が緩やかに変わる輸送) が残っていることを意味する。
  G. directed 区間を 3 等分し、ドリフト方向が一定かどうかを見る。
"""
import json

import numpy as np

import slide_common as S
from q7_direction import vacf, dacf, angle_stats, drift_estimate   # noqa: E402
import common as C                                                 # noqa: E402

rng = np.random.default_rng(20260830)
NSIM = 2000
MAXLAG = 20

import sys

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'recal'
SEGMENTS = {5: S.SEGMENTS_T5, 8: S.VARIANTS[VARIANT]}


def d_driftfree(pos):
    """平均速度を除いた D = <|dr - <dr>|^2>/4 [px^2/min]。"""
    v = np.diff(np.asarray(pos, float), axis=0)
    mv = v.mean(axis=0)
    return float(((v - mv) ** 2).sum(axis=1).mean() / 4.0)


def calibrate(N, D, v_drift, nsim=NSIM):
    """区間長 N・拡散係数 D に合わせた較正。confined は CE=0.16 相当の R。"""
    R = np.sqrt(4 * D) / 0.16
    conds = {
        'free': C.sim_free(N, D, rng, nsim),
        'confined': C.sim_confined(N, D, R, rng, nsim),
        'directed (obs. v)': C.sim_directed(N, D, v_drift, rng, nsim),
    }
    out = {}
    for name, P in conds.items():
        c1 = np.array([vacf(P[i], 1)[1] for i in range(nsim)])
        u1 = np.array([dacf(P[i], 1)[1] for i in range(nsim)])
        ff = np.array([angle_stats(P[i])['forward_frac'] for i in range(nsim)])
        f = lambda a: dict(mean=float(a.mean()), sd=float(a.std()),
                           q025=float(np.quantile(a, 0.025)),
                           q975=float(np.quantile(a, 0.975)))
        out[name] = dict(C1=f(c1), Cu1=f(u1), forward_frac=f(ff))
    return out


def corr_time(cu, nmax=10):
    """Cu(n) = exp(-n/tau) の片対数フィット (Cu>0 の範囲のみ)。"""
    n = np.arange(1, nmax + 1)
    y = np.asarray(cu[1:nmax + 1], float)
    ok = np.isfinite(y) & (y > 1e-3)
    if ok.sum() < 3:
        return np.nan
    a = np.polyfit(n[ok], np.log(y[ok]), 1)[0]
    return float(-1.0 / a) if a < 0 else np.nan


def analyse(pos, i0, i1):
    seg = np.asarray(pos, float)[i0:i1]
    dr = drift_estimate(seg)
    D = d_driftfree(seg)
    DE2 = dr['speed'] ** 2 / (4 * D)
    cv = vacf(seg, MAXLAG)
    cu = dacf(seg, MAXLAG)
    return dict(
        t0=i0, t1=i1, n=i1 - i0,
        D_driftfree=D,
        D_apparent=float(np.sum(np.diff(seg, axis=0) ** 2, axis=1).mean() / 4),
        drift=dr, DE=float(np.sqrt(DE2)),
        C_const_drift=float(DE2 / (1 + DE2)),      # 一定ドリフトの予言 (平坦)
        vacf=[float(x) for x in cv],
        dacf=[float(x) for x in cu],
        angles=angle_stats(seg),
        tau_corr=corr_time(cu),
        cal=calibrate(i1 - i0, D, dr['speed']),
    )


def thirds(pos, i0, i1):
    """区間を 3 等分し、それぞれのドリフト方向・速さを返す。"""
    edges = np.linspace(i0, i1, 4).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        d = drift_estimate(np.asarray(pos, float)[a:b])
        out.append(dict(t0=int(a), t1=int(b), speed=d['speed'],
                        direction_deg=d['direction_deg'], z2=d['z2']))
    return out


# ---------------------------------------------------------------------------
def sliding_dacf(pos, W=61):
    """スライディング窓の Cu(1) と前方散乱率。VACF と違い大きさに鈍感なので、
    「速い」ことと「まっすぐ」なことを分離できる。"""
    p = np.asarray(pos, float)
    N = len(p)
    h = W // 2
    cu, ff, ctr = [], [], []
    for i in range(h, N - h):
        seg = p[i - h:i + h + 1]
        cu.append(dacf(seg, 1)[1])
        ff.append(angle_stats(seg)['forward_frac'])
        ctr.append(i)
    return np.array(ctr), np.array(cu), np.array(ff)


def figure(res, tracks):
    """スライディング窓の方向相関 C_u(1) の時間変化 (TRACK 8 のみ)。

    DC-MSS が directed と判定した区間をピンクで塗る。凡例は置かない。
    """
    dirs = [(a, b) for a, b, c in SEGMENTS[8] if c == 'directed']

    plt = S.slide_style()
    fig, ax = plt.subplots(figsize=(10.5, 7.0))          # 横縦比 3:2

    ax.axhline(0, color=S.COL_WHITE, lw=1.0, alpha=0.5, zorder=2)

    ctr, cu, _ = sliding_dacf(tracks[8]['pos'])
    ax.plot(ctr / 60, cu, color=S.COL_RED, lw=2.6, zorder=4)

    ax.set_xlabel('Time (h)', labelpad=8)
    ax.set_ylabel(r'Directional correlation  C$_u$(1)', labelpad=8)
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(0, 13, 2))
    lo, hi = np.nanmin(cu), np.nanmax(cu)
    ax.set_ylim(lo - 0.06, hi + 0.06)

    fig.tight_layout()
    name = f'{S.FIGS}/S3_segment_direction_{VARIANT}.png'
    fig.savefig(name)
    fig.savefig(name.replace('.png', '.pdf'))
    plt.close(fig)
    print(f'  {name}   directed {len(dirs)} 区間: '
          f'{[(a, b) for a, b in dirs]}')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    tracks, _ = S.get_tracks()
    res, extra = {}, {}
    for tid, segs in SEGMENTS.items():
        pos = tracks[tid]['pos']
        res[str(tid)] = [analyse(pos, i0, i1) for i0, i1, _ in segs]
        for k, (i0, i1, cls) in enumerate(segs):
            if cls == 'directed':
                extra[f'{tid}-{k}'] = thirds(pos, i0, i1)

    with open(f'{S.OUT}/segment_direction_{VARIANT}.json', 'w') as f:
        json.dump(dict(segments=res, thirds=extra), f, indent=2)

    W = 94
    print('=' * W)
    print(f'区間ごとの方向相関解析  [分割: {VARIANT}]')
    print('=' * W)
    hdr = (f"{'区間':>18}{'N':>5}{'D_df':>8}{'|v|':>8}{'DE':>7}"
           f"{'C(1)':>8}{'Cu(1)':>8}{'fwd':>7}{'tau':>7}")
    for tid, segs in SEGMENTS.items():
        print(f'\nTRACK {tid}')
        print(hdr)
        for r, (_, _, cls) in zip(res[str(tid)], segs):
            lab = f"{r['t0']}-{r['t1']} {cls}"
            print(f"{lab:>18}{r['n']:>5}{r['D_driftfree']:>8.2f}"
                  f"{r['drift']['speed']:>8.3f}{r['DE']:>7.3f}"
                  f"{r['vacf'][1]:>8.3f}{r['dacf'][1]:>8.3f}"
                  f"{r['angles']['forward_frac']:>7.3f}"
                  f"{r['tau_corr']:>7.2f}")

    print('\n' + '=' * W)
    print('較正: 同じ N・同じ D の帰無モデルにおける 95% 区間')
    print('=' * W)
    for tid, segs in SEGMENTS.items():
        for r, (_, _, cls) in zip(res[str(tid)], segs):
            print(f"\nTRACK {tid}  {r['t0']}-{r['t1']} min ({cls}), "
                  f"N={r['n']}, D={r['D_driftfree']:.2f}")
            print(f"  {'':>20}{'C(1)':>10}{'95% CI':>22}"
                  f"{'Cu(1)':>10}{'95% CI':>22}")
            print(f"  {'観測':>20}{r['vacf'][1]:>10.3f}{'':>22}"
                  f"{r['dacf'][1]:>10.3f}")
            for name, c in r['cal'].items():
                ci1 = f"[{c['C1']['q025']:+.3f},{c['C1']['q975']:+.3f}]"
                ci2 = f"[{c['Cu1']['q025']:+.3f},{c['Cu1']['q975']:+.3f}]"
                print(f"  {name:>20}{c['C1']['mean']:>10.3f}{ci1:>22}"
                      f"{c['Cu1']['mean']:>10.3f}{ci2:>22}")
            print(f"  {'一定ドリフトの予言':>19}{r['C_const_drift']:>10.3f}"
                  f"{'(n>=1 で平坦)':>22}")

    print('\n' + '=' * W)
    print('directed 区間を 3 等分したときのドリフト方向')
    print('=' * W)
    for key, ts in extra.items():
        print(f'  segment {key}:')
        for t in ts:
            print(f"    {t['t0']:>4}-{t['t1']:>4} min  "
                  f"|v| = {t['speed']:.3f} px/min "
                  f"({t['speed'] * S.PX_UM:.4f} um/min)  "
                  f"方向 = {t['direction_deg']:+7.1f} deg  z2 = {t['z2']:.1f}")

    figure(res, tracks)
