"""Portable extraction of the five analysis definitions in provenance/dcmss3.py.
Function/class bodies are unchanged; site-specific plotting/data imports are omitted.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mss_core import MSSThresholds, classify_segments, confinement_radius, max_pairwise_distance_series, _reclassify, _ks_merge, CLASS_NAMES

def extend_short(bounds, min_seg, floor):
    """floor <= L < min_seg の区間を min_seg まで伸ばす (隣を削る)。

    mss_core._extend_short_segments と同じ手続きだが、下限を
    min_seg - 2*sigma に固定せず引数で与える。
    """
    b = list(bounds)
    changed = True
    while changed:
        changed = False
        for i, L in enumerate(np.diff(b)):
            if not (floor <= L < min_seg):
                continue
            need = min_seg - L
            if i + 2 < len(b) and (b[i + 2] - b[i + 1]) - need >= min_seg:
                b[i + 1] += need
                changed = True
                break
            if i - 1 >= 0 and (b[i] - b[i - 1]) - need >= min_seg:
                b[i] -= need
                changed = True
                break
    return b

def step1(pos, window, sigma, pctl, min_seg, floor, ks_alpha=0.05):
    p = np.asarray(pos, float)
    N = len(p)
    smpd = gaussian_filter1d(max_pairwise_distance_series(p, window), sigma,
                             mode='nearest')
    num = np.abs(np.diff(smpd))
    den = (smpd[1:] + smpd[:-1]) / 2.0
    ds = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    thr_v = np.percentile(ds, pctl)
    peak = np.zeros(len(ds), bool)
    peak[1:-1] = (ds[1:-1] >= ds[:-2]) & (ds[1:-1] >= ds[2:])
    cand = np.where(peak & (ds >= thr_v))[0] + 1
    b = sorted(set([0] + [int(c) for c in cand] + [N]))
    b = extend_short(b, min_seg, floor)
    b, _ = _ks_merge(p, b, ks_alpha)
    return np.array(b)

class Cfg:
    def __init__(self, window, sigma, pctl, min_seg, lag_frac, floor,
                 guard=True):
        self.window, self.sigma, self.peak_pctl = window, sigma, pctl
        self.min_seg, self.lag_frac, self.floor = min_seg, lag_frac, floor
        self.guard = guard
        self.ks_alpha, self.lag_cap = 0.05, 30

    def key(self):
        return (self.window, self.sigma, self.peak_pctl, self.min_seg,
                self.lag_frac, self.floor, self.guard)

    def __repr__(self):
        return (f'W{self.window}_s{self.sigma:g}_p{self.peak_pctl:g}'
                f'_L{self.min_seg}_f{self.lag_frac}_e{self.floor}'
                f'{"_G" if self.guard else ""}')

def step3(pos, segs, thr, min_seg, guard):
    """mss_core.final_segmentation と同じだが Case 1 ガードを持つ。

    ガード: 分類不能な短片 (< min_seg) との統合が、既に分類されている相手の
    クラスを変えてしまう場合は統合しない。論文の規則は「統合後のクラスが
    同じか低移動度なら無条件受理」だが、これだと 11 frames の分類不能片が
    74 frames の directed 区間を free に塗り替えてしまう。p_misclass は
    区間が長いほど小さくなるため、この降格を止める働きをしない。
    """
    segs = [dict(s) for s in segs]
    for _ in range(300):
        merged = False
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            case1 = (a['cls'] == -1) ^ (b['cls'] == -1)
            case2 = (a['cls'] == b['cls']) and a['cls'] != -1
            if not (case1 or case2):
                continue
            i0, i1 = a['start'], b['end']
            new_cls, new_s, new_D = _reclassify(pos, i0, i1, thr, min_seg)
            if new_cls == -1:
                continue
            ref = a['cls'] if a['cls'] != -1 else b['cls']
            if case1 and guard and new_cls != ref:
                continue
            if new_cls <= ref:
                accept = True
            else:
                p_new = thr.p_misclass(new_s, i1 - i0, new_cls)
                if case1:
                    k = a if a['cls'] != -1 else b
                    p_sep = thr.p_misclass(k['mss_slope'], k['len'], k['cls'])
                else:
                    w1 = a['len'] / (a['len'] + b['len'])
                    p_sep = (w1 * thr.p_misclass(a['mss_slope'], a['len'], a['cls'])
                             + (1 - w1) * thr.p_misclass(b['mss_slope'], b['len'],
                                                         b['cls']))
                accept = p_new <= p_sep
            if accept:
                segs[i:i + 2] = [dict(start=i0, end=i1, len=i1 - i0,
                                      mss_slope=new_s, cls=new_cls, D=new_D)]
                merged = True
                break
        if not merged:
            break
    return segs

def dc_mss3(pos, thr, cfg):
    b = step1(pos, cfg.window, cfg.sigma, cfg.peak_pctl, cfg.min_seg,
              cfg.floor, cfg.ks_alpha)
    segs0 = classify_segments(pos, b, thr, cfg.min_seg)
    segs = step3(pos, segs0, thr, cfg.min_seg, cfg.guard)
    for s in segs:
        sub = np.asarray(pos, float)[s['start']:s['end']]
        s['conf_radius'] = confinement_radius(sub)
        s['v_net'] = np.linalg.norm(sub[-1] - sub[0]) / max(1, s['len'] - 1)
    return dict(bounds=b, segs_initial=segs0, segs=segs)
