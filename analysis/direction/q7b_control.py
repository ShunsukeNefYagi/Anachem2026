"""7 の対照: 観測された正の速度自己相関はアーティファクトではないか。

検討する交絡
  (a) 欠損フレームの線形内挿 — 内挿区間は人工的な直線になり、正の相関を作る。
      TRACK 5 は 6 フレームを内挿している。内挿点に触れるステップを除外して再計算する。
  (b) 局在誤差 — これは lag-1 に *負* の相関を作るので、正の相関を弱める方向に働く。
      すなわち観測値は真の持続性の下限である。局在誤差を加えたときの減衰を定量化する。
  (c) TRACK 8 の directed 期 (140-374 min) を除いても持続性が残るか。
"""
import json
import numpy as np
import common as C
from q7_direction import vacf, dacf, angle_stats, drift_estimate

rng = np.random.default_rng(4242)


def vacf_masked(pos, valid, maxlag=10):
    """valid[i] = False の位置に触れるステップを除外した VACF。"""
    p = np.asarray(pos, float)
    v = np.diff(p, axis=0)
    # ステップ i は位置 i と i+1 を使う
    step_ok = valid[:-1] & valid[1:]
    denom = float((v[step_ok] ** 2).sum(axis=1).mean())
    out = [1.0]
    for n in range(1, maxlag + 1):
        m = step_ok[:-n] & step_ok[n:]
        if m.sum() < 5:
            out.append(np.nan)
            continue
        out.append(float((v[:-n][m] * v[n:][m]).sum(axis=1).mean() / denom))
    return np.array(out)


def noise_attenuation(D=20.0, N=720, v_drift=1.0):
    """局在誤差 sigma_p が VACF(1) をどれだけ引き下げるか。"""
    out = []
    for ratio in (0.0, 0.14, 0.28, 0.42, 0.70):
        sigma = ratio * np.sqrt(4 * D)          # sigma_p / Sigma_D = ratio
        c1f, c1d = [], []
        for _ in range(400):
            p = C.sim_free(N, D, rng, 1)[0]
            p = p + rng.normal(0, sigma, p.shape) if sigma > 0 else p
            c1f.append(vacf(p, 1)[1])
            q = C.sim_directed(N, D, v_drift, rng, 1)[0]
            q = q + rng.normal(0, sigma, q.shape) if sigma > 0 else q
            c1d.append(vacf(q, 1)[1])
        out.append(dict(sigma_ratio=ratio, sigma_px=float(sigma),
                        C1_free=float(np.mean(c1f)),
                        C1_directed=float(np.mean(c1d))))
    return out


if __name__ == '__main__':
    tracks, rep = C.get_tracks()
    res = {}

    print('=' * 92)
    print('7b-A. 内挿点の影響 (TRACK 5 は 6 フレームを線形内挿している)')
    print('=' * 92)
    for tid in sorted(tracks):
        p = tracks[tid]['pos']
        interp = tracks[tid]['interpolated']
        valid = ~interp
        full = vacf(p, 10)
        mask = vacf_masked(p, valid, 10)
        res[str(tid)] = dict(n_interp=int(interp.sum()),
                             vacf_all=[float(v) for v in full],
                             vacf_excl_interp=[float(v) for v in mask])
        print(f"\n  TRACK {tid} (内挿フレーム数 = {int(interp.sum())})")
        print(f"    {'lag':>5}{'全点':>12}{'内挿除外':>12}{'差':>10}")
        for n in range(1, 8):
            print(f"    {n:>5}{full[n]:>+12.4f}{mask[n]:>+12.4f}"
                  f"{mask[n]-full[n]:>+10.4f}")

    print()
    print('=' * 92)
    print('7b-B. 局在誤差は正の相関を弱める (すなわち観測値は下限)')
    print('=' * 92)
    att = noise_attenuation()
    res['noise_attenuation'] = att
    print(f"  {'sigma_p/Sigma_D':>16}{'sigma_p (px)':>14}"
          f"{'C(1) free':>12}{'C(1) directed(v=1)':>21}")
    for a in att:
        print(f"  {a['sigma_ratio']:>16.2f}{a['sigma_px']:>14.2f}"
              f"{a['C1_free']:>+12.4f}{a['C1_directed']:>+21.4f}")
    print("\n  → 局在誤差は C(1) を必ず負方向へ引く。実測が正である以上、")
    print("     真の方向持続性は観測値以上であり、かつ局在誤差は小さい。")

    print()
    print('=' * 92)
    print('7b-C. TRACK 8: directed 期 (140-374 min) を除いた残りの持続性')
    print('=' * 92)
    p8 = tracks[8]['pos']
    parts = {'0-140 min': p8[0:140], '140-374 min (directed)': p8[140:374],
             '374-720 min': p8[374:720]}
    res['track8_parts'] = {}
    print(f"  {'区間':>24}{'C(1)':>9}{'C(2)':>9}{'C(3)':>9}{'C(5)':>9}"
          f"{'前方率':>9}{'|v| px/min':>12}{'z^2':>8}")
    for k, seg in parts.items():
        c = vacf(seg, 6)
        a = angle_stats(seg)
        d = drift_estimate(seg)
        res['track8_parts'][k] = dict(vacf=[float(x) for x in c],
                                      angles=a, drift=d)
        print(f"  {k:>24}{c[1]:>+9.3f}{c[2]:>+9.3f}{c[3]:>+9.3f}{c[5]:>+9.3f}"
              f"{a['forward_frac']:>9.3f}{d['speed']:>12.3f}{d['z2']:>8.1f}")

    # TRACK 5 も前半 / 後半で
    print()
    p5 = tracks[5]['pos']
    print(f"  {'TRACK 5 区間':>24}{'C(1)':>9}{'C(2)':>9}{'C(3)':>9}{'C(5)':>9}"
          f"{'前方率':>9}{'|v| px/min':>12}{'z^2':>8}")
    res['track5_parts'] = {}
    for k, seg in {'0-360 min (前半)': p5[0:360], '360-720 min (後半)': p5[360:720]}.items():
        c = vacf(seg, 6)
        a = angle_stats(seg)
        d = drift_estimate(seg)
        res['track5_parts'][k] = dict(vacf=[float(x) for x in c], angles=a, drift=d)
        print(f"  {k:>24}{c[1]:>+9.3f}{c[2]:>+9.3f}{c[3]:>+9.3f}{c[5]:>+9.3f}"
              f"{a['forward_frac']:>9.3f}{d['speed']:>12.3f}{d['z2']:>8.1f}")

    with open(f'{C.OUT}/q7b_control.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\nsaved -> {C.OUT}/q7b_control.json')
