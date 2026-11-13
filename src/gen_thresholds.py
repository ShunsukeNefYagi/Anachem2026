"""Regenerate the MSS classification thresholds for data with dt = 1 min.

    python src/gen_thresholds.py      (~7 min; writes results/calibration/*.npz)

The paper uses thr_sig0.14_lf3.npz (shipped as data/thr_sig0.14_lf3.npz).  All
eight tables are generated because the random-number stream is shared between
them: generating only one would not reproduce it.

The thresholds of Vega et al. (2018) were derived for
  - segment durations of 20-500 frames
  - tau_max = N/4 (capped at 30 frames)
  - no localization error
  - confined: CE = 0.7, directed: DE = 1.0
Our frame interval is about 2000 times longer than theirs, so we want a shorter
minimum segment length.  Changing the segment length changes the spread of the
MSS-slope distributions, so the thresholds have to be regenerated with it.

This script computes MSS-slope distributions for
  duration  : 6-760 frames (extended below the paper's lower limit of 20)
  lag_frac  : both tau_max = N/4 and N/3
  sigma_p   : 0, 0.14, 0.28, 0.42 x Sigma_D (the same levels as the paper's Fig. S3)
"""
import os
import time

import numpy as np

from mss_core import (MOMENT_ORDERS, add_localization_noise, sim_confined, sim_directed,
                      sim_free, sim_immobile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    np.arange(6, 41, 1),        # fine steps, reaching below the paper's lower limit of 20
    np.arange(42, 101, 2),
    np.arange(110, 201, 10),
    np.arange(240, 761, 40),
]))


def mss_slopes_multi(pos, lag_fracs, lag_cap=LAG_CAP):
    """MSS slopes for several lag_frac values from a single moment computation."""
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
    """MSS slopes of one motion class for every duration / sigma_p / lag_frac.
    Returns: dict[(sigma_ratio, lag_frac)] -> (n_dur, n_sim)"""
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
    # For immobile, the true position is fixed and all apparent motion is
    # localization error, so the MSS-slope distribution does not depend on
    # sigma_p (sigma_p = 1 is used as the representative value)
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
        store[k] = build(k, gen, srs, seed=SEEDS[k])   # fixed seed for reproducibility

    # one file per (sigma_ratio, lag_frac)
    for sr in SIGMA_RATIOS:
        for lf in LAG_FRACS:
            slopes = np.stack([
                store['immobile'][(0.0, lf)],                     # independent of sigma
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
