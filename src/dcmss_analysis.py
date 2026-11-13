"""DC-MSS segmentation and motion classification of one trajectory.

Method: Vega AR, Freeman SA, Grinstein S, Jaqaman K. "Multistep Track Segmentation
and Motion Classification for Transient Mobility Analysis", Biophys. J. 114,
1018-1025 (2018).  Implementation: mss_core.py (independent Python implementation).

Conditions used in the paper (recalibrated for 1-min sampling, see README):
    MPD window 5 positions, Gaussian sigma 2 frames, switch-point percentile 97.5,
    minimum classifiable segment 20 positions, MSS max lag = min(N // 3, 30),
    thresholds from simulations with localization error sigma_p / Sigma_D = 0.14
    (data/thr_sig0.14_lf3.npz, made by gen_thresholds.py).
"""
import os

import numpy as np

from mss_core import CLASS_NAMES, DCMSSConfig, MSSThresholds, dc_mss, mss_analysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = os.path.join(ROOT, 'data', 'thr_sig0.14_lf3.npz')
CONFIG = DCMSSConfig(window=5, sigma=2.0, peak_pctl=97.5, min_seg=20,
                     lag_frac=3, name='paper')
PX_UM = 10.0 / 222.0          # um / px


def load_thresholds(path=THRESHOLDS, cfg=CONFIG):
    thr = MSSThresholds(path)
    assert thr.lag_frac == cfg.lag_frac and thr.lag_cap == cfg.lag_cap, \
        'threshold table does not match the MSS lag settings'
    return thr


def direction_autocorrelation(pos, lag=1):
    """C_u(lag) = <u(t).u(t+lag)>, u = unit vector of a 1-frame displacement.
    Pairs containing a zero-length displacement are excluded."""
    v = np.diff(np.asarray(pos, float), axis=0)
    n = np.linalg.norm(v, axis=1)
    ok = n > 1e-12
    u = np.zeros_like(v)
    u[ok] = v[ok] / n[ok, None]
    m = ok[:-lag] & ok[lag:]
    return float((u[:-lag][m] * u[lag:][m]).sum(axis=1).mean()) if m.any() else np.nan


def analyse(pos, thr=None, cfg=CONFIG):
    """Run DC-MSS on pos (N, 2) [px].  Returns (rows, result).

    rows: one dict per final segment; bounds are half-open [start, end_exclusive)
    frame indices.  gen_D2_px2 is the generalised coefficient of the 2nd moment
    (mu_2 = D_2 (4 tau)^alpha_2); it is a diffusion coefficient only if alpha_2 = 1.
    """
    pos = np.asarray(pos, float)
    thr = thr or load_thresholds(cfg=cfg)
    res = dc_mss(pos, thr, cfg)
    rows = []
    for k, s in enumerate(res['segs'], 1):
        a, b = int(s['start']), int(s['end'])
        sub = pos[a:b]
        step = np.linalg.norm(np.diff(sub, axis=0), axis=1)
        t = thr.thresholds_at(b - a)
        m = mss_analysis(sub, lag_frac=cfg.lag_frac, lag_cap=cfg.lag_cap)
        rows.append(dict(
            segment=k, start=a, end_exclusive=b, n_positions=b - a,
            motion_class=CLASS_NAMES[s['cls']], mss_slope=s['mss_slope'],
            thr_free_directed=t[2], thr_confined_free=t[1], thr_immobile_confined=t[0],
            max_lag=int(m['max_lag']), alpha2=float(m['alphas'][0, 2]),
            gen_D2_px2=s['D'], conf_radius_px=s['conf_radius'],
            v_net_um_per_min=s['v_net'] * PX_UM,
            straightness=float(np.linalg.norm(sub[-1] - sub[0]) / step.sum()),
            Cu_lag1=direction_autocorrelation(sub, 1)))
    return rows, res


def plot(pos, rows, path, title=''):
    """Trajectory coloured by class (diffusive black, directed red) and S_MSS per
    segment with its length-dependent free|directed threshold."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    color = {'directed': '#d62728'}
    p = np.asarray(pos, float) * PX_UM
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.8),
                                   gridspec_kw=dict(width_ratios=[1, 1.4]))
    for r in rows:
        a, b = r['start'], r['end_exclusive']
        seg = p[a:min(b + 1, len(p))]       # +1 joins consecutive segments visually
        c = color.get(r['motion_class'], 'black')
        ax0.plot(seg[:, 0], seg[:, 1], color=c, lw=0.8)
        ax1.plot([a, b], [r['mss_slope']] * 2, color=c, lw=2)
        ax1.plot([a, b], [r['thr_free_directed']] * 2, color='0.4', ls='--', lw=0.8)
    for r in rows[1:]:
        ax1.axvline(r['start'], color='0.6', ls=':', lw=0.8)
    ax0.plot(*p[0], 'o', mfc='white', mec='black', ms=5)
    ax0.set(aspect='equal', xlabel='x (µm)', ylabel='y (µm)', title=title)
    ax0.invert_yaxis()
    ax1.set(xlim=(0, len(p)), xlabel='Time (min)', ylabel='$S_{MSS}$')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
