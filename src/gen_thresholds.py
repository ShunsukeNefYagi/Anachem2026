"""Δt = 1 min のデータ向けに MSS 分類閾値を作り直す。

    python src/gen_thresholds.py      (~7 min; writes results/calibration/*.npz)

The paper uses thr_sig0.14_lf3.npz (shipped as data/thr_sig0.14_lf3.npz).  All
eight tables are generated because the random-number stream is shared between
them: generating only one would not reproduce it.

論文 (Vega et al. 2018) の閾値は
  - segment duration 20-500 frames
  - tau_max = N/4 (30 frames で cap)
  - localization error なし
  - confined: CE = 0.7, directed: DE = 1.0
という条件で作られている。本データは撮像間隔が約 2000 倍違うため、
minimum segment length を短くしたい。segment length を変えると MSS slope の
分布幅が変わるので、閾値もセットで作り直す必要がある。

本スクリプトは
  duration  : 6-760 frames（論文の下限 20 より下まで延長）
  lag_frac  : tau_max = N/4 と N/3 の両方
  sigma_p   : 0, 0.14, 0.28, 0.42 x Sigma_D（論文 Fig. S3 と同じ水準）
について MSS slope 分布を求める。
"""
import numpy as np, time, sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mss_core import (sim_immobile, sim_confined, sim_free, sim_directed,
                      add_localization_noise, MOMENT_ORDERS)

# Written to results/, not data/, so a re-run never overwrites the shipped
# threshold table that the published classification was made with.
OUT = os.path.join(ROOT, 'results', 'calibration')
os.makedirs(OUT, exist_ok=True)

N_SIM = 10000
D_REF = 1.0
DT = 1.0
SIGMA_RATIOS = [0.0, 0.14, 0.28, 0.42]
LAG_FRACS = [4, 3]
LAG_CAP = 30
CE_REF, DE_REF = 0.7, 1.0
SEEDS = {'immobile': 101, 'free': 202, 'confined': 303, 'directed': 404}

DURATIONS = np.unique(np.concatenate([
    np.arange(6, 41, 1),        # 論文の下限 20 より下まで細かく
    np.arange(42, 101, 2),
    np.arange(110, 201, 10),
    np.arange(240, 761, 40),
]))


def mss_slopes_multi(pos, lag_fracs, lag_cap=LAG_CAP):
    """1 回のモーメント計算から複数の lag_frac の MSS slope を得る。"""
    n_tr, N, _ = pos.shape
    max_lag = min(N // min(lag_fracs), lag_cap)
    if max_lag < 2:
        return {lf: np.full(n_tr, np.nan) for lf in lag_fracs}
    n_m = len(MOMENT_ORDERS)
    log_mu = np.empty((n_tr, n_m, max_lag))
    for i, tau in enumerate(range(1, max_lag + 1)):
        d = np.linalg.norm(pos[:, tau:, :] - pos[:, :-tau, :], axis=2)
        p = np.ones_like(d)
        for j in range(n_m):
            m = p.mean(axis=1)
            log_mu[:, j, i] = np.log(np.maximum(m, 1e-300))
            p *= d
    m = MOMENT_ORDERS.astype(float)
    mc = m - m.mean()
    out = {}
    for lf in lag_fracs:
        k = min(N // lf, lag_cap)
        if k < 2:
            out[lf] = np.full(n_tr, np.nan)
            continue
        x = np.log(4.0 * np.arange(1, k + 1))
        xc = x - x.mean()
        alphas = (log_mu[:, :, :k] * xc).sum(axis=2) / (xc ** 2).sum()
        out[lf] = (alphas * mc).sum(axis=1) / (mc ** 2).sum()
    return out


def build(tag, gen, sigma_ratios, durations=DURATIONS, n_sim=N_SIM, seed=0):
    """1 つの運動クラスについて、全 duration / sigma_p / lag_frac の slope を得る。
    戻り値: dict[(sigma_ratio, lag_frac)] -> (n_dur, n_sim)"""
    rng = np.random.default_rng(seed)
    res = {(sr, lf): np.full((len(durations), n_sim), np.nan, np.float32)
           for sr in sigma_ratios for lf in LAG_FRACS}
    sigma_D = np.sqrt(2 * D_REF * DT)
    t0 = time.time()
    for i, N in enumerate(durations):
        base = gen(n_sim, int(N), rng)
        for sr in sigma_ratios:
            pos = add_localization_noise(base, sr * sigma_D, rng) if sr > 0 else base
            sl = mss_slopes_multi(pos, LAG_FRACS)
            for lf in LAG_FRACS:
                res[(sr, lf)][i] = sl[lf].astype(np.float32)
        if i % 20 == 0:
            print(f'  [{tag}] {i+1}/{len(durations)} N={N} {time.time()-t0:.0f}s',
                  flush=True)
    return res


GENS = {
    # immobile は「真の位置は静止、見かけの動きは局在誤差のみ」なので
    # MSS slope 分布は sigma_p に依存しない (sigma_p=1 で代表させる)
    'immobile': (lambda n, N, rng: sim_immobile(n, N, sigma_p=1.0, rng=rng), [0.0]),
    'free':     (lambda n, N, rng: sim_free(n, N, D=D_REF, dt=DT, rng=rng), SIGMA_RATIOS),
    'confined': (lambda n, N, rng: sim_confined(n, N, D=D_REF, dt=DT, CE=CE_REF, rng=rng),
                 SIGMA_RATIOS),
    'directed': (lambda n, N, rng: sim_directed(n, N, D=D_REF, dt=DT, DE=DE_REF, rng=rng),
                 SIGMA_RATIOS),
}

if __name__ == '__main__':
    t0 = time.time()
    store = {}
    for k, (gen, srs) in GENS.items():
        print(f'--- {k} ---', flush=True)
        store[k] = build(k, gen, srs, seed=SEEDS[k])   # 再現性のため固定シード

    # (sigma_ratio, lag_frac) ごとに 1 ファイルにまとめる
    for sr in SIGMA_RATIOS:
        for lf in LAG_FRACS:
            slopes = np.stack([
                store['immobile'][(0.0, lf)],                     # sigma 非依存
                store['confined'][(sr, lf)],
                store['free'][(sr, lf)],
                store['directed'][(sr, lf)],
            ], axis=1)                                            # (n_dur, 4, n_sim)
            meta = dict(sigma_ratio=sr, lag_frac=lf, lag_cap=LAG_CAP,
                        CE=CE_REF, DE=DE_REF, D=D_REF, dt=DT, n_sim=N_SIM)
            np.savez_compressed(
                os.path.join(OUT, f'thr_sig{sr:.2f}_lf{lf}.npz'),
                durations=DURATIONS, slopes=slopes, n_sim=N_SIM,
                lag_cap=LAG_CAP, lag_frac=lf, meta=np.array(meta, dtype=object))
    print(f'done in {time.time()-t0:.0f}s -> {OUT}')
