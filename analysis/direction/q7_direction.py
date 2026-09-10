"""追加解析②: 方向相関解析 (directional correlation analysis)

MSD/MSS は変位の「大きさ」のスケーリングしか見ないため、
方向情報を捨てている。方向相関はそれを直接見る独立な手法であり、
  - 方向性輸送 (drift)     -> 正の速度自己相関、turning angle が 0 度に偏る
  - 閉じ込め / 局在        -> 負の速度自己相関 (跳ね返り)、180 度に偏る
  - 自由拡散               -> 相関ゼロ、turning angle 一様
を区別できる。局在誤差も lag-1 に負の相関を作るので、その寄与も評価する。

計算する量
  7-A. 速度自己相関 VACF: C(n) = <v(t)·v(t+n)> / <|v|^2>,  v(t) = r(t+1) - r(t)
  7-B. 方向自己相関 (単位ベクトル): Cu(n) = <u(t)·u(t+n)> = <cos theta(t, t+n)>
  7-C. turning angle 分布と非対称性
  7-D. 60 分区間ごとの C(1), Cu(1), 前方散乱率
  7-E. 実測ドリフト速度の推定と、VACF から期待される値との一致
すべて正解既知シミュレーションで較正する。
"""
import json
import numpy as np
import common as C

rng = np.random.default_rng(2718)
NSIM = 3000
MAXLAG = 20


def velocities(pos):
    return np.diff(np.asarray(pos, float), axis=0)


def vacf(pos, maxlag=MAXLAG):
    """規格化速度自己相関 C(n) = <v(t).v(t+n)> / <|v|^2>"""
    v = velocities(pos)
    denom = float((v ** 2).sum(axis=1).mean())
    out = []
    for n in range(0, maxlag + 1):
        if n == 0:
            out.append(1.0)
            continue
        if len(v) - n < 5:
            out.append(np.nan)
            continue
        out.append(float((v[:-n] * v[n:]).sum(axis=1).mean() / denom))
    return np.array(out)


def dacf(pos, maxlag=MAXLAG):
    """方向 (単位ベクトル) 自己相関 <cos theta>。大きさの変動に鈍感。"""
    v = velocities(pos)
    nrm = np.linalg.norm(v, axis=1)
    ok = nrm > 1e-12
    u = np.zeros_like(v)
    u[ok] = v[ok] / nrm[ok, None]
    out = []
    for n in range(0, maxlag + 1):
        if n == 0:
            out.append(1.0)
            continue
        if len(u) - n < 5:
            out.append(np.nan)
            continue
        m = ok[:-n] & ok[n:]
        out.append(float((u[:-n][m] * u[n:][m]).sum(axis=1).mean()))
    return np.array(out)


def turning_angles(pos):
    """連続する変位ベクトル間の角度 [-pi, pi]。"""
    v = velocities(pos)
    a = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(a)
    return (d + np.pi) % (2 * np.pi) - np.pi


def angle_stats(pos):
    th = turning_angles(pos)
    if len(th) < 5:
        return dict(n=len(th), mean_cos=np.nan, forward_frac=np.nan,
                    backward_frac=np.nan, fb_ratio=np.nan)
    c = np.cos(th)
    fwd = float((np.abs(th) < np.pi / 2).mean())
    bwd = 1.0 - fwd
    return dict(n=int(len(th)), mean_cos=float(c.mean()),
                forward_frac=fwd, backward_frac=bwd,
                fb_ratio=float(fwd / bwd) if bwd > 0 else np.nan)


def drift_estimate(pos):
    """平均速度ベクトルとその有意性 (速度成分の標準誤差ベース)。"""
    v = velocities(pos)
    mv = v.mean(axis=0)
    n = len(v)
    se = v.std(axis=0, ddof=1) / np.sqrt(n)
    speed = float(np.linalg.norm(mv))
    # 2 自由度の Hotelling 風統計量
    z2 = float(((mv / se) ** 2).sum())
    return dict(vx=float(mv[0]), vy=float(mv[1]), speed=speed,
                se_x=float(se[0]), se_y=float(se[1]), z2=z2,
                direction_deg=float(np.degrees(np.arctan2(mv[1], mv[0]))))


# ---------------------------------------------------------------------------
def calibrate(N, D=20.0, label=''):
    """N frames の各運動形態における C(1), Cu(1), <cos theta>, 前方散乱率。"""
    conds = {
        'free': C.sim_free(N, D, rng, NSIM),
        'confined R=20 (CE=0.32)': C.sim_confined(N, D, 20.0, rng, NSIM),
        'confined R=40 (CE=0.16)': C.sim_confined(N, D, 40.0, rng, NSIM),
        'directed v=2 (DE=0.22)': C.sim_directed(N, D, 2.0, rng, NSIM),
        'directed v=6 (DE=0.67)': C.sim_directed(N, D, 6.0, rng, NSIM),
    }
    # 局在誤差のみ (immobile + noise) : lag-1 に -0.5 の相関を作る
    conds['pure localization noise'] = C.sim_immobile(N, 3.0, rng, NSIM)
    out = {}
    for name, P in conds.items():
        c1, u1, mc, ff = [], [], [], []
        for i in range(P.shape[0]):
            c1.append(vacf(P[i], 1)[1])
            u1.append(dacf(P[i], 1)[1])
            s = angle_stats(P[i])
            mc.append(s['mean_cos']); ff.append(s['forward_frac'])
        f = lambda a: (float(np.mean(a)), float(np.std(a)),
                       float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
        out[name] = dict(
            C1=dict(zip(('mean', 'sd', 'q025', 'q975'), f(c1))),
            Cu1=dict(zip(('mean', 'sd', 'q025', 'q975'), f(u1))),
            mean_cos=dict(zip(('mean', 'sd', 'q025', 'q975'), f(mc))),
            forward_frac=dict(zip(('mean', 'sd', 'q025', 'q975'), f(ff))))
    return out


if __name__ == '__main__':
    tracks, _ = C.get_tracks()

    whole = {}
    for tid in sorted(tracks):
        p = tracks[tid]['pos']
        whole[tid] = dict(
            vacf=[float(v) for v in vacf(p, MAXLAG)],
            dacf=[float(v) for v in dacf(p, MAXLAG)],
            angles=angle_stats(p), drift=drift_estimate(p))

    wins = []
    for tid in sorted(tracks):
        for k, (i0, i1, seg) in enumerate(C.windows(tracks[tid]['pos'])):
            wins.append(dict(track=tid, win=k, t0=i0, t1=i1,
                             D=C.D_from_msd1(seg),
                             vacf=[float(v) for v in vacf(seg, 10)],
                             dacf=[float(v) for v in dacf(seg, 10)],
                             angles=angle_stats(seg),
                             drift=drift_estimate(seg)))

    cal60 = calibrate(60)
    cal720 = calibrate(720)

    with open(f'{C.OUT}/q7_direction.json', 'w') as f:
        json.dump(dict(whole={str(k): v for k, v in whole.items()},
                       windows=wins, cal_N60=cal60, cal_N720=cal720), f, indent=2)

    print('=' * 96)
    print('7-A/B. 軌跡全体 (720 frames) の速度自己相関 VACF と方向自己相関 DACF')
    print('=' * 96)
    print(f"{'lag n':>7}" + ''.join(f"{f'T{t} VACF':>11}{f'T{t} DACF':>11}"
                                    for t in sorted(whole)))
    for n in list(range(0, 11)) + [15, 20]:
        row = f"{n:>7}"
        for t in sorted(whole):
            row += f"{whole[t]['vacf'][n]:>11.4f}{whole[t]['dacf'][n]:>11.4f}"
        print(row)

    print('\n  較正 (N=720):')
    print(f"{'condition':>26}{'C(1) 平均':>12}{'95% CI':>20}"
          f"{'Cu(1) 平均':>12}{'95% CI':>20}")
    for k, v in cal720.items():
        ci = f"[{v['C1']['q025']:+.4f},{v['C1']['q975']:+.4f}]"
        ciu = f"[{v['Cu1']['q025']:+.4f},{v['Cu1']['q975']:+.4f}]"
        print(f"{k:>26}{v['C1']['mean']:>+12.4f}{ci:>20}"
              f"{v['Cu1']['mean']:>+12.4f}{ciu:>20}")

    print()
    print('=' * 96)
    print('7-C. turning angle と正味ドリフト (軌跡全体)')
    print('=' * 96)
    for t in sorted(whole):
        a, d = whole[t]['angles'], whole[t]['drift']
        print(f"  TRACK {t}: <cos theta> = {a['mean_cos']:+.4f}, "
              f"前方散乱率 = {a['forward_frac']:.4f} (等方なら 0.5), "
              f"前方/後方比 = {a['fb_ratio']:.3f}")
        print(f"           平均速度 = ({d['vx']:+.4f}, {d['vy']:+.4f}) px/min, "
              f"|v| = {d['speed']:.4f}, 方向 = {d['direction_deg']:+.1f} deg, "
              f"z^2 = {d['z2']:.1f}")
    fr = cal720['free']['forward_frac']
    print(f"\n  較正 (free, N=720): 前方散乱率 = {fr['mean']:.4f} "
          f"95% CI [{fr['q025']:.4f}, {fr['q975']:.4f}]")
    fr6 = cal720['directed v=6 (DE=0.67)']['forward_frac']
    print(f"  較正 (directed DE=0.67): 前方散乱率 = {fr6['mean']:.4f}")

    print()
    print('=' * 96)
    print('7-D. 60 分区間ごとの方向相関')
    print('=' * 96)
    v = cal60['free']
    print(f"  較正 (free, N=60): C(1) 95% CI "
          f"[{v['C1']['q025']:+.3f}, {v['C1']['q975']:+.3f}], "
          f"前方散乱率 95% CI [{v['forward_frac']['q025']:.3f}, "
          f"{v['forward_frac']['q975']:.3f}]")
    print(f"{'trk':>4}{'win':>4}{'t(min)':>10}{'C(1)':>9}{'C(2)':>9}{'C(3)':>9}"
          f"{'Cu(1)':>9}{'<cos>':>9}{'前方率':>9}{'|v| px/min':>12}{'flag':>6}")
    lo, hi = v['C1']['q025'], v['C1']['q975']
    flo, fhi = v['forward_frac']['q025'], v['forward_frac']['q975']
    nf = 0
    for w in wins:
        fl = ''
        if not (lo <= w['vacf'][1] <= hi):
            fl += 'C'
        if not (flo <= w['angles']['forward_frac'] <= fhi):
            fl += 'F'
        if fl:
            nf += 1
        print(f"{w['track']:>4}{w['win']:>4}{f'{w['t0']}-{w['t1']}':>10}"
              f"{w['vacf'][1]:>+9.3f}{w['vacf'][2]:>+9.3f}{w['vacf'][3]:>+9.3f}"
              f"{w['dacf'][1]:>+9.3f}{w['angles']['mean_cos']:>+9.3f}"
              f"{w['angles']['forward_frac']:>9.3f}{w['drift']['speed']:>12.3f}{fl:>6}")
    print(f"\n  方向相関が較正 95% 区間の外にある区間: {nf}/24 (期待値 ~2.3)")
    print(f'\nsaved -> {C.OUT}/q7_direction.json')
