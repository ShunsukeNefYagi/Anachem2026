"""analysis2 共通ユーティリティ.

査読コメントの検証に必要な最小限の道具立て:
  - two.csv の読み込み (analysis/prepare_data.py を再利用)
  - 60 分窓への分割
  - MSD の単一べき乗則フィット (点数可変) と適合度指標
  - 窓ごとの MSS 解析

単位: 長さ = px, 時間 = frame = 1 min
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))     # repository root
DCMSS = os.path.join(ROOT, 'analysis', 'dcmss')
sys.path.insert(0, DCMSS)

from prepare_data import load_tracks          # noqa: E402
import mss_core as mc                          # noqa: E402

OUT = os.path.join(ROOT, 'results', 'direction')
FIGS = os.path.join(OUT, 'figs')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

WINDOW_MIN = 60          # 査読コメントが問題にしている 60 分区間
N_FIT_POINTS = 10        # 論文が用いた MSD フィット点数 (tau = 1..10 min)


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------
def get_tracks():
    tracks, report = load_tracks(os.path.join(ROOT, 'data', 'two.csv'))
    return tracks, report


def windows(pos, win=WINDOW_MIN):
    """非重複の 60 分窓に分割し [(i0, i1, pos_seg), ...] を返す。"""
    N = len(pos)
    out = []
    for i0 in range(0, N - win + 1, win):
        out.append((i0, i0 + win, pos[i0:i0 + win]))
    return out


# ---------------------------------------------------------------------------
# MSD と単一べき乗則フィット
# ---------------------------------------------------------------------------
def msd(pos, max_lag):
    """時間平均 MSD。returns taus, msd, n_pairs"""
    pos = np.asarray(pos, float)
    N = len(pos)
    max_lag = int(min(max_lag, N - 1))
    taus = np.arange(1, max_lag + 1)
    m = np.array([np.mean(np.sum((pos[t:] - pos[:-t]) ** 2, axis=1)) for t in taus])
    npair = np.array([N - t for t in taus], float)
    return taus, m, npair


def powerlaw_fit(taus, msd_vals):
    """log-log 単一べき乗則フィット MSD = K tau^alpha.

    返り値 dict:
      alpha, logK, r2      : 通常の最小二乗の結果と決定係数
      rmse_log             : log10 残差の RMS
      max_absres_log       : log10 残差の最大絶対値
      curv                 : 2次項を足したときの曲率係数 (log-log 空間)
      curv_t               : 曲率係数の t 値 (|t| > 2 で曲率が有意)
      p_curv               : 曲率係数の両側 p 値 (点の独立性を仮定した参考値)
    """
    from scipy import stats
    x = np.log(np.asarray(taus, float))
    y = np.log(np.asarray(msd_vals, float))
    n = len(x)

    # 1次 (単一べき乗則)
    A1 = np.vstack([x, np.ones(n)]).T
    c1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    yhat = A1 @ c1
    res = y - yhat
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # 2次 (曲率の検定)
    xc = x - x.mean()
    A2 = np.vstack([xc ** 2, xc, np.ones(n)]).T
    c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    res2 = y - A2 @ c2
    dof = n - 3
    s2 = float((res2 ** 2).sum()) / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(A2.T @ A2)
    se_curv = float(np.sqrt(cov[0, 0]))
    t_curv = float(c2[0] / se_curv) if se_curv > 0 else np.nan
    p_curv = float(2 * stats.t.sf(abs(t_curv), dof)) if dof > 0 else np.nan

    return dict(alpha=float(c1[0]), logK=float(c1[1]), r2=float(r2),
                rmse_log=float(np.sqrt((res ** 2).mean())),
                max_absres_log=float(np.abs(res).max()),
                curv=float(c2[0]), curv_t=t_curv, p_curv=p_curv,
                n_points=int(n))


def alpha_of(pos, n_fit=N_FIT_POINTS):
    taus, m, _ = msd(pos, n_fit)
    return powerlaw_fit(taus, m)


# ---------------------------------------------------------------------------
# MSS
# ---------------------------------------------------------------------------
def mss_of(pos, lag_cap=None, lag_frac=4):
    """1本の軌跡の MSS。alphas(m=0..6), MSS slope, および MSS 曲線の曲率を返す。"""
    r = mc.mss_analysis(pos, lag_cap=lag_cap, lag_frac=lag_frac)
    alphas = r['alphas'][0]
    m = mc.MOMENT_ORDERS.astype(float)
    # MSS 曲線の直線からのずれ (自己相似なら alpha_m は m の直線)
    A = np.vstack([m, np.ones(len(m))]).T
    c, *_ = np.linalg.lstsq(A, alphas, rcond=None)
    resid = alphas - A @ c
    # 2次項
    mc_ = m - m.mean()
    A2 = np.vstack([mc_ ** 2, mc_, np.ones(len(m))]).T
    c2, *_ = np.linalg.lstsq(A2, alphas, rcond=None)
    return dict(mss_slope=float(r['mss_slope'][0]),
                alphas=alphas, gen_D=r['gen_D'][0],
                max_lag=int(r['max_lag']),
                mss_resid_rms=float(np.sqrt((resid ** 2).mean())),
                mss_curv=float(c2[0]))


# ---------------------------------------------------------------------------
# 空間的な広がりの指標
# ---------------------------------------------------------------------------
def radius_gyration(pos):
    p = np.asarray(pos, float)
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def max_pairwise_distance(pos):
    p = np.asarray(pos, float)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return float(d.max())


def net_over_path(pos):
    p = np.asarray(pos, float)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path = step.sum()
    net = np.linalg.norm(p[-1] - p[0])
    return float(net / path) if path > 0 else np.nan


def D_from_msd1(pos):
    """D = MSD(1)/4 [px^2/min]"""
    p = np.asarray(pos, float)
    d2 = np.sum(np.diff(p, axis=0) ** 2, axis=1)
    return float(d2.mean() / 4.0)


# ---------------------------------------------------------------------------
# シミュレーション (較正用)
# ---------------------------------------------------------------------------
def sim_free(N, D, rng, n=1):
    s = np.sqrt(2.0 * D)          # dt = 1
    st = rng.normal(0, s, size=(n, N - 1, 2))
    p = np.concatenate([np.zeros((n, 1, 2)), np.cumsum(st, axis=1)], axis=1)
    return p


def sim_confined(N, D, R, rng, n=1):
    """半径 R の円板内に閉じ込められた拡散 (径方向反射)。"""
    s = np.sqrt(2.0 * D)
    p = np.zeros((n, N, 2))
    for t in range(1, N):
        cand = p[:, t - 1] + rng.normal(0, s, size=(n, 2))
        r = np.linalg.norm(cand, axis=1)
        bad = r > R
        if bad.any():
            cand[bad] *= ((2 * R - r[bad]) / r[bad])[:, None]
        p[:, t] = cand
    return p


def sim_directed(N, D, v, rng, n=1):
    s = np.sqrt(2.0 * D)
    th = rng.uniform(0, 2 * np.pi, size=(n, 1))
    drift = np.concatenate([np.cos(th), np.sin(th)], axis=1)[:, None, :] * v
    st = rng.normal(0, s, size=(n, N - 1, 2)) + drift
    p = np.concatenate([np.zeros((n, 1, 2)), np.cumsum(st, axis=1)], axis=1)
    return p


def sim_immobile(N, sigma, rng, n=1):
    return rng.normal(0, sigma, size=(n, N, 2))
