"""
MSS (Moment Scaling Spectrum) 解析と DC-MSS (divide-and-conquer MSS) の実装。

参考:
  Vega AR, Freeman SA, Grinstein S, Jaqaman K.
  "Multistep Track Segmentation and Motion Classification for Transient Mobility
   Analysis", Biophysical Journal 114, 1018-1025 (2018).
  および Supporting Material (Note S1, Fig. S1, S2, S4).

単位系: 長さ = pixel, 時間 = frame (本データでは 1 frame = 1 min)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp

MOMENT_ORDERS = np.arange(0, 7)          # 0th - 6th moment (Note S1, step 1)
MAX_LAG_CAP = 30                         # 120 frame 超のトラックでは lag を 30 に制限
LAG_FRAC = 4                             # tau_max = segment duration / LAG_FRAC
MIN_SEG = 20                             # MSS 分類が可能な最小セグメント長 [frames]
WINDOW = 11                              # Initial Track Segmentation の rolling window
GAUSS_SIGMA = 2.0                        # MPD(t) 平滑化の Gaussian kernel [frames]
PEAK_PCTL = 95.0                         # DSMPD ピークの上位 5%
KS_ALPHA = 0.05                          # 隣接セグメント統合の KS 検定有意水準

CLASS_NAMES = {-1: 'unclassified', 0: 'immobile', 1: 'confined',
               2: 'free', 3: 'directed'}
CLASS_COLORS = {-1: '#000000', 0: '#8B5A2B', 1: '#1f4fd8',
                2: '#00c8d7', 3: '#e01ec8'}


# --------------------------------------------------------------------------
# MSS 解析
# --------------------------------------------------------------------------
def mss_analysis(pos, lag_cap=None, lag_frac=None):
    """バッチ版 MSS 解析。

    pos      : ndarray, shape (n_tracks, N, 2)
    lag_cap  : 時間ラグの上限 (既定 MAX_LAG_CAP=30)。None で既定値。
    lag_frac : tau_max = N / lag_frac (既定 LAG_FRAC=4)。None で既定値。
    returns dict with
        mss_slope : (n_tracks,)   MSS 曲線 alpha_m vs m の傾き
        alphas    : (n_tracks, 7) 各モーメント次数のスケーリング指数
        gen_D     : (n_tracks, 7) 一般化拡散係数 D_m
        (D_2 = gen_D[:,2] が MSD 由来の拡散係数)
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[None, :, :]
    n_tr, N, _ = pos.shape
    lf = LAG_FRAC if lag_frac is None else lag_frac
    max_lag = min(N // lf, MAX_LAG_CAP if lag_cap is None else lag_cap)
    if max_lag < 2:
        raise ValueError(f'track too short for MSS: N={N}')

    n_m = len(MOMENT_ORDERS)
    log_mu = np.empty((n_tr, n_m, max_lag))
    for i, tau in enumerate(range(1, max_lag + 1)):
        d = np.linalg.norm(pos[:, tau:, :] - pos[:, :-tau, :], axis=2)
        p = np.ones_like(d)
        for j in range(n_m):
            # mu_m(tau) = < |dr(tau)|^m >
            log_mu[:, j, i] = np.log(p.mean(axis=1))
            p *= d

    # mu_m(tau) = D_m * (4 tau)^alpha_m  ->  log mu_m = log D_m + alpha_m log(4 tau)
    x = np.log(4.0 * np.arange(1, max_lag + 1))
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    alphas = (log_mu * xc).sum(axis=2) / denom
    intercept = log_mu.mean(axis=2) - alphas * x.mean()
    gen_D = np.exp(intercept)

    # MSS 曲線: alpha_m vs m の直線回帰 (m = 0..6, 切片あり)
    m = MOMENT_ORDERS.astype(float)
    mc = m - m.mean()
    mss_slope = (alphas * mc).sum(axis=1) / (mc ** 2).sum()

    return dict(mss_slope=mss_slope, alphas=alphas, gen_D=gen_D, max_lag=max_lag)


def msd_curve(pos, max_lag=None):
    """時間平均 MSD (2次モーメント)。pos: (N,2)"""
    pos = np.asarray(pos, float)
    N = len(pos)
    if max_lag is None:
        max_lag = N // 4
    taus = np.arange(1, max_lag + 1)
    msd = np.array([np.mean(np.sum((pos[t:] - pos[:-t]) ** 2, axis=1))
                    for t in taus])
    n_pairs = np.array([N - t for t in taus])
    return taus, msd, n_pairs


def fit_D_sigma(pos, n_lag=5):
    """MSD(tau) = 4 D tau + 4 sigma_p^2 の線形フィット。
    D [px^2/frame], sigma_p [px] (負の切片なら sigma_p = nan)。"""
    taus, msd, _ = msd_curve(pos, max_lag=min(n_lag, max(2, len(pos) // 4)))
    A = np.vstack([taus, np.ones_like(taus, dtype=float)]).T
    coef, *_ = np.linalg.lstsq(A, msd, rcond=None)
    D = coef[0] / 4.0
    b = coef[1]
    sigma = np.sqrt(b / 4.0) if b > 0 else np.nan
    return D, sigma, b


def confinement_radius(pos):
    """位置分散から閉じ込め半径を推定。
    2次元一様円板 (半径 R) では共分散行列の固有値平均 = R^2/4 なので
    R = sqrt((d+2) * mean(eig))  (d=2)。u-track と同じ定義。"""
    pos = np.asarray(pos, float)
    if len(pos) < 3:
        return np.nan
    cov = np.cov(pos.T)
    ev = np.linalg.eigvalsh(cov)
    return float(np.sqrt(4.0 * ev.mean()))


# --------------------------------------------------------------------------
# シミュレーション (閾値導出用)
# --------------------------------------------------------------------------
def sim_immobile(n, N, sigma_p=1.0, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, sigma_p, size=(n, N, 2))


def sim_free(n, N, D=1.0, dt=1.0, rng=None):
    rng = rng or np.random.default_rng()
    step = rng.normal(0.0, np.sqrt(2 * D * dt), size=(n, N, 2))
    step[:, 0, :] = 0.0
    return np.cumsum(step, axis=1)


def sim_confined(n, N, D=1.0, dt=1.0, CE=0.7, rng=None):
    """閉じ込め拡散。CE = sqrt(4 D dt) / R (confinement extent)。
    境界は半径 R の円で、はみ出した分は径方向に反射させる。"""
    rng = rng or np.random.default_rng()
    R = np.sqrt(4 * D * dt) / CE
    s = np.sqrt(2 * D * dt)
    pos = np.zeros((n, N, 2))
    cur = np.zeros((n, 2))
    steps = rng.normal(0.0, s, size=(n, N, 2))
    for t in range(1, N):
        nxt = cur + steps[:, t, :]
        r = np.linalg.norm(nxt, axis=1)
        out = r > R
        if out.any():
            scale = (2 * R - r[out]) / r[out]
            nxt[out] = nxt[out] * scale[:, None]
            # 極端な跳ね返り(2R超)は円内に丸める
            r2 = np.linalg.norm(nxt, axis=1)
            bad = r2 > R
            if bad.any():
                nxt[bad] = nxt[bad] / r2[bad][:, None] * R
        cur = nxt
        pos[:, t, :] = cur
    return pos


def sim_directed(n, N, D=1.0, dt=1.0, DE=1.0, rng=None):
    """ドリフト付き拡散。DE = v dt / sqrt(4 D dt) (drift extent)。"""
    rng = rng or np.random.default_rng()
    v = DE * np.sqrt(4 * D * dt) / dt
    theta = rng.uniform(0, 2 * np.pi, size=(n, 1))
    drift = np.concatenate([np.cos(theta), np.sin(theta)], axis=1) * v * dt
    step = rng.normal(0.0, np.sqrt(2 * D * dt), size=(n, N, 2))
    step += drift[:, None, :]
    step[:, 0, :] = 0.0
    return np.cumsum(step, axis=1)


SIMULATORS = {0: sim_immobile, 1: sim_confined, 2: sim_free, 3: sim_directed}


# --------------------------------------------------------------------------
# 閾値テーブル
# --------------------------------------------------------------------------
class MSSThresholds:
    """継続時間依存の MSS slope 分類閾値 + 誤分類確率計算。

    npz には
      durations : (nd,)
      slopes    : (nd, 4, nsim)  各 duration / class の MSS slope サンプル
    が入っている。
    """

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.durations = z['durations']
        self.lag_cap = int(z['lag_cap']) if 'lag_cap' in z else MAX_LAG_CAP
        self.lag_frac = int(z['lag_frac']) if 'lag_frac' in z else LAG_FRAC
        self.meta = dict(z['meta'].item()) if 'meta' in z else {}
        self.slopes = np.sort(z['slopes'], axis=2)   # 昇順ソート済み
        self.nsim = self.slopes.shape[2]
        # tau_max = N/lag_frac が 2 未満になる duration は MSS 回帰が定義できず
        # NaN になる。そこを除外し、分類可能な最小長を記録する。
        ok = np.isfinite(self.slopes).all(axis=(1, 2))
        if not ok.all():
            self.durations = self.durations[ok]
            self.slopes = self.slopes[ok]
        self.min_classifiable = int(self.durations[0])
        self.thr = self._build_thresholds()          # (nd, 3): imm|conf, conf|free, free|dir

    # -- 閾値導出 (Note S1) ------------------------------------------------
    @staticmethod
    def _balanced_threshold(lower, upper):
        """lower(低移動度クラス)と upper(高移動度クラス)の分布から
        P(lower > t) = P(upper < t) となる t と、その共通誤分類確率を返す。"""
        cand = np.unique(np.concatenate([lower, upper]))
        p_lo = 1.0 - np.searchsorted(lower, cand, side='right') / len(lower)
        p_up = np.searchsorted(upper, cand, side='left') / len(upper)
        diff = p_lo - p_up
        k = int(np.argmin(np.abs(diff)))
        t = cand[k]
        return float(t), float(max(p_lo[k], p_up[k]))

    def _build_thresholds(self):
        nd = len(self.durations)
        thr = np.zeros((nd, 3))
        for i in range(nd):
            imm, con, fre, dir_ = (self.slopes[i, c] for c in range(4))
            pairs = [(imm, con), (con, fre), (fre, dir_)]
            # Regime 2 のフォールバック値 (誤分類確率 0.01 基準)
            regime2 = [np.quantile(imm, 0.99),    # immobile を confined と誤る確率 0.01
                       np.quantile(fre, 0.01),    # free を confined と誤る確率 0.01
                       np.quantile(fre, 0.99)]    # free を directed と誤る確率 0.01
            for k, (lo, up) in enumerate(pairs):
                t, p = self._balanced_threshold(lo, up)
                thr[i, k] = t if p > 0.01 else regime2[k]
            # 単調性を保証
            thr[i] = np.maximum.accumulate(thr[i])
        return thr

    def thresholds_at(self, dur):
        d = np.clip(dur, self.durations[0], self.durations[-1])
        return np.array([np.interp(d, self.durations, self.thr[:, k])
                         for k in range(3)])

    def classify(self, mss_slope, dur, min_seg=None):
        if dur < (MIN_SEG if min_seg is None else min_seg):
            return -1
        if dur < self.min_classifiable or not np.isfinite(mss_slope):
            return -1
        t = self.thresholds_at(dur)
        return int(np.searchsorted(t, mss_slope, side='right'))

    # -- 誤分類確率 (Fig. S4b) --------------------------------------------
    def _cdf(self, cls, dur, s):
        """duration dur における class cls の MSS slope 分布の CDF を s で評価。
        duration はグリッド上の近傍2点から線形内挿。"""
        d = np.clip(dur, self.durations[0], self.durations[-1])
        j = int(np.clip(np.searchsorted(self.durations, d) - 1,
                        0, len(self.durations) - 2))
        d0, d1 = self.durations[j], self.durations[j + 1]
        w = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
        c0 = np.searchsorted(self.slopes[j, cls], s, side='right') / self.nsim
        c1 = np.searchsorted(self.slopes[j + 1, cls], s, side='right') / self.nsim
        return (1 - w) * c0 + w * c1

    def p_misclass(self, mss_slope, dur, cls):
        """Fig. S4b 方式: 与えられた slope 値に対し、隣接クラスの分布のうち
        「そのクラス側に食い込む」裾の面積を誤分類確率とする。"""
        s = mss_slope
        if cls == 0:
            return self._cdf(1, dur, s)
        if cls == 1:
            return (1 - self._cdf(0, dur, s)) + self._cdf(2, dur, s)
        if cls == 2:
            return (1 - self._cdf(1, dur, s)) + self._cdf(3, dur, s)
        if cls == 3:
            return 1 - self._cdf(2, dur, s)
        return 1.0


# --------------------------------------------------------------------------
# DC-MSS
# --------------------------------------------------------------------------
def max_pairwise_distance_series(pos, window=WINDOW):
    """rolling window 内の最大点間距離 MPD(t)。窓の中心フレームに値を割り当て、
    端は最も近い値で埋めて長さ N の系列にする。"""
    pos = np.asarray(pos, float)
    N = len(pos)
    half = window // 2
    vals = np.empty(N - window + 1)
    for i in range(N - window + 1):
        w = pos[i:i + window]
        diff = w[:, None, :] - w[None, :, :]
        vals[i] = np.sqrt((diff ** 2).sum(axis=2)).max()
    out = np.empty(N)
    out[half:N - half] = vals
    out[:half] = vals[0]
    out[N - half:] = vals[-1]
    return out


def initial_segmentation(pos, window=WINDOW, sigma=GAUSS_SIGMA,
                         peak_pctl=PEAK_PCTL, ks_alpha=KS_ALPHA,
                         min_seg=MIN_SEG, verbose=False):
    """DC-MSS Step 1。境界インデックス配列 (0 ... N) を返す。"""
    N = len(pos)
    mpd = max_pairwise_distance_series(pos, window)
    smpd = gaussian_filter1d(mpd, sigma, mode='nearest')

    num = np.abs(np.diff(smpd))
    den = (smpd[1:] + smpd[:-1]) / 2.0
    dsmpd = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    # 局所極大 かつ 上位 5%
    thr = np.percentile(dsmpd, peak_pctl)
    is_peak = np.zeros(len(dsmpd), bool)
    is_peak[1:-1] = (dsmpd[1:-1] >= dsmpd[:-2]) & (dsmpd[1:-1] >= dsmpd[2:])
    cand = np.where(is_peak & (dsmpd >= thr))[0] + 1        # 切替点フレーム

    bounds = [0] + [int(c) for c in cand] + [N]
    bounds = sorted(set(bounds))

    # 短いセグメント (min_seg-2sigma <= L < min_seg) を min_seg に伸ばす
    bounds = _extend_short_segments(bounds, N, min_seg, sigma)

    # 隣接セグメントの点間距離分布を KS 検定で比較し p>0.05 なら統合
    bounds, n_merged = _ks_merge(pos, bounds, ks_alpha)

    info = dict(mpd=mpd, smpd=smpd, dsmpd=dsmpd, thr=thr,
                n_initial_peaks=len(cand), n_ks_merged=n_merged)
    if verbose:
        print(f'  initial peaks={len(cand)}, KS merged={n_merged}, '
              f'segments={len(bounds)-1}')
    return np.array(bounds), info


def _extend_short_segments(bounds, N, min_seg=MIN_SEG, sigma=GAUSS_SIGMA):
    """min_seg-2*sigma <= L < min_seg のセグメントを min_seg まで伸ばす。
    隣接セグメントが min_seg 未満になる場合は伸ばさない。"""
    b = list(bounds)
    lo = int(min_seg - 2 * sigma)
    changed = True
    while changed:
        changed = False
        lens = np.diff(b)
        for i, L in enumerate(lens):
            if lo <= L < min_seg:
                need = min_seg - L
                # 右隣を削る
                if i + 2 < len(b) and (b[i + 2] - b[i + 1]) - need >= min_seg:
                    b[i + 1] += need
                    changed = True
                    break
                # 左隣を削る
                if i - 1 >= 0 and (b[i] - b[i - 1]) - need >= min_seg:
                    b[i] -= need
                    changed = True
                    break
    return b


def _pairwise_dist(pos, i0, i1, cap=20000, rng=None):
    seg = pos[i0:i1]
    diff = seg[:, None, :] - seg[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    iu = np.triu_indices(len(seg), k=1)
    d = d[iu]
    if len(d) > cap:
        rng = rng or np.random.default_rng(0)
        d = rng.choice(d, cap, replace=False)
    return d


def _ks_merge(pos, bounds, alpha=KS_ALPHA):
    b = list(bounds)
    n_merged = 0
    changed = True
    while changed and len(b) > 2:
        changed = False
        for i in range(len(b) - 2):
            d1 = _pairwise_dist(pos, b[i], b[i + 1])
            d2 = _pairwise_dist(pos, b[i + 1], b[i + 2])
            if len(d1) < 2 or len(d2) < 2:
                continue
            if ks_2samp(d1, d2).pvalue > alpha:
                del b[i + 1]
                n_merged += 1
                changed = True
                break
    return b, n_merged


def classify_segments(pos, bounds, thresholds, min_seg=MIN_SEG):
    """DC-MSS Step 2。"""
    cap = getattr(thresholds, 'lag_cap', None)
    frac = getattr(thresholds, 'lag_frac', None)
    segs = []
    for i in range(len(bounds) - 1):
        i0, i1 = int(bounds[i]), int(bounds[i + 1])
        L = i1 - i0
        if L >= min_seg:
            r = mss_analysis(pos[i0:i1], lag_cap=cap, lag_frac=frac)
            s = float(r['mss_slope'][0])
            cls = thresholds.classify(s, L, min_seg)
            D2 = float(r['gen_D'][0, 2])
        else:
            s, cls, D2 = np.nan, -1, np.nan
        segs.append(dict(start=i0, end=i1, len=L, mss_slope=s, cls=cls, D=D2))
    return segs


def _reclassify(pos, i0, i1, thresholds, min_seg=MIN_SEG):
    L = i1 - i0
    if L < min_seg:
        return -1, np.nan, np.nan
    r = mss_analysis(pos[i0:i1], lag_cap=getattr(thresholds, 'lag_cap', None),
                     lag_frac=getattr(thresholds, 'lag_frac', None))
    s = float(r['mss_slope'][0])
    return thresholds.classify(s, L, min_seg), s, float(r['gen_D'][0, 2])


def final_segmentation(pos, segs, thresholds, max_iter=200, min_seg=MIN_SEG):
    """DC-MSS Step 3: 隣接セグメントの統合判定。"""
    segs = [dict(s) for s in segs]
    for _ in range(max_iter):
        merged = False
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            case1 = (a['cls'] == -1) ^ (b['cls'] == -1)
            case2 = (a['cls'] == b['cls']) and a['cls'] != -1
            if not (case1 or case2):
                continue
            i0, i1 = a['start'], b['end']
            new_cls, new_s, new_D = _reclassify(pos, i0, i1, thresholds, min_seg)
            if new_cls == -1:
                continue
            ref = a['cls'] if a['cls'] != -1 else b['cls']
            accept = False
            if new_cls <= ref:
                # 同じか低移動度のクラス -> 受理 (長く観るほど閉じ込めが見える)
                accept = True
            else:
                p_new = thresholds.p_misclass(new_s, i1 - i0, new_cls)
                if case1:
                    known = a if a['cls'] != -1 else b
                    p_sep = thresholds.p_misclass(known['mss_slope'],
                                                  known['len'], known['cls'])
                else:
                    w1 = a['len'] / (a['len'] + b['len'])
                    w2 = 1 - w1
                    p_sep = (w1 * thresholds.p_misclass(a['mss_slope'], a['len'], a['cls'])
                             + w2 * thresholds.p_misclass(b['mss_slope'], b['len'], b['cls']))
                accept = p_new <= p_sep
            if accept:
                segs[i:i + 2] = [dict(start=i0, end=i1, len=i1 - i0,
                                      mss_slope=new_s, cls=new_cls, D=new_D)]
                merged = True
                break
        if not merged:
            break
    return segs


class DCMSSConfig:
    """DC-MSS の調整可能パラメータ。既定値は Vega et al. 2018 の論文値。

    論文の実験は 33 Hz 撮像なので、frame 数で定義されたパラメータは
    そのままでは撮像間隔の違う実験に転用できない。本クラスはそれらを
    明示的に外に出し、シミュレーションによる再較正を可能にする。
    """

    def __init__(self, window=WINDOW, sigma=GAUSS_SIGMA, peak_pctl=PEAK_PCTL,
                 min_seg=MIN_SEG, ks_alpha=KS_ALPHA, lag_cap=MAX_LAG_CAP,
                 lag_frac=LAG_FRAC, name=None):
        self.window = int(window)
        self.sigma = float(sigma)
        self.peak_pctl = float(peak_pctl)
        self.min_seg = int(min_seg)
        self.ks_alpha = float(ks_alpha)
        self.lag_cap = int(lag_cap)
        self.lag_frac = int(lag_frac)
        self.name = name or (f'W{self.window}_s{self.sigma:g}_p{self.peak_pctl:g}'
                             f'_L{self.min_seg}_f{self.lag_frac}')

    def as_dict(self):
        return dict(window=self.window, sigma=self.sigma, peak_pctl=self.peak_pctl,
                    min_seg=self.min_seg, ks_alpha=self.ks_alpha,
                    lag_cap=self.lag_cap, lag_frac=self.lag_frac, name=self.name)

    def __repr__(self):
        return f'DCMSSConfig({self.name})'


PAPER_CFG = DCMSSConfig(name='paper')


def dc_mss(pos, thresholds, cfg=None, verbose=False):
    """DC-MSS 全体。pos: (N,2)"""
    cfg = cfg or PAPER_CFG
    bounds, info = initial_segmentation(
        pos, window=cfg.window, sigma=cfg.sigma, peak_pctl=cfg.peak_pctl,
        ks_alpha=cfg.ks_alpha, min_seg=cfg.min_seg, verbose=verbose)
    segs0 = classify_segments(pos, bounds, thresholds, cfg.min_seg)
    segs = final_segmentation(pos, segs0, thresholds, min_seg=cfg.min_seg)
    # セグメントごとの付随量
    for s in segs:
        sub = pos[s['start']:s['end']]
        s['conf_radius'] = confinement_radius(sub)
        net = np.linalg.norm(sub[-1] - sub[0])
        s['v_net'] = net / max(1, s['len'] - 1)
        if s['len'] >= cfg.min_seg:
            D_lin, sig, _ = fit_D_sigma(sub)
            s['D_lin'] = D_lin
            s['sigma_p_fit'] = sig
        else:
            s['D_lin'] = np.nan
            s['sigma_p_fit'] = np.nan
    return dict(bounds=bounds, segs_initial=segs0, segs=segs, step1=info,
                cfg=cfg.as_dict())


# --------------------------------------------------------------------------
# 局在誤差付きシミュレーション
# --------------------------------------------------------------------------
def add_localization_noise(pos, sigma_p, rng=None):
    """各観測位置に独立な Gaussian ノイズ N(0, sigma_p) を加える。"""
    if sigma_p <= 0:
        return pos
    rng = rng or np.random.default_rng()
    return pos + rng.normal(0.0, sigma_p, size=pos.shape)


def sigma_from_ratio(ratio, D, dt=1.0):
    """sigma_p / Sigma_D = ratio となる sigma_p を返す。
    Sigma_D = sqrt(2 D dt) は 1 次元ステップサイズの標準偏差。"""
    return ratio * np.sqrt(2 * D * dt)


def sim_switching(n, durations, classes, D=1.0, dt=1.0, CE=0.7, DE=1.0,
                  sigma_p=0.0, rng=None):
    """運動形態が切り替わる軌跡を生成する。

    durations : 各区間の長さのリスト
    classes   : 各区間の運動クラス (0=immobile,1=confined,2=free,3=directed)
    戻り値    : (pos, switch_points)  pos は (n, sum(durations), 2)
    """
    rng = rng or np.random.default_rng()
    N = int(sum(durations))
    pos = np.zeros((n, N, 2))
    cur = np.zeros((n, 2))
    i = 0
    switches = []
    for L, c in zip(durations, classes):
        L = int(L)
        if c == 0:
            # immobile: 真の位置は動かない。見かけの揺らぎは局在誤差のみ。
            sub = np.repeat(cur[:, None, :], L, axis=1)
        else:
            sub = SIMULATORS[c](n, L, D=D, dt=dt, rng=rng, **(
                {'CE': CE} if c == 1 else {'DE': DE} if c == 3 else {}))
            sub = sub - sub[:, :1, :] + cur[:, None, :]
        pos[:, i:i + L, :] = sub
        cur = sub[:, -1, :]
        i += L
        if i < N:
            switches.append(i)
    if sigma_p > 0:
        pos = add_localization_noise(pos, sigma_p, rng)
    return pos, switches
