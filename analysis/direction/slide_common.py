"""スライド用図の共通設定。

スケール: 222 px = 10 um  ->  1 px = 0.045045 um
時間    : 1 frame = 1 min
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))     # repository root
ANALYSIS = os.path.join(ROOT, 'analysis', 'dcmss')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

OUT = os.path.join(ROOT, 'results', 'direction')
FIGS = os.path.join(OUT, 'figs')
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

PX_UM = 10.0 / 222.0            # um / px
DT_MIN = 1.0                    # min / frame

# 添付図の配色 (黒背景)
COL_RED = '#CC2229'             # TRACK 8 の基本色 (= 添付図の TRACK 8)
COL_BLUE = '#2E7BC4'            # TRACK 5 の色 (S3/S5 の directed 強調にも使う)
COL_PINK = '#FF4FA3'            # 軌跡図の directed 区間
COL_WHITE = '#FFFFFF'
BG = '#000000'

# 再較正済み DC-MSS の採用条件 (analysis/dcmss/apply_recalibrated.py と同一)
CFG_WINDOW = 5
CFG_SIGMA = 2.0

# TRACK 8 の区間。報告書には 2 通りの分割が併記されている。
#   'recal' : 本データ条件へ再較正した主解析
#             analysis/dcmss/apply_recalibrated.py の出力、directed 1 つ
#   'paper' : Vega et al. 既定パラメータ、analysis/dcmss/run_analysis.py の出力、
#             directed 2 つ
SEGMENTS_T8_RECAL = [(0, 140, 'free'), (140, 374, 'directed'), (374, 720, 'free')]
SEGMENTS_T8_PAPER = [(0, 139, 'free'), (139, 166, 'directed'),
                     (166, 271, 'free'), (271, 373, 'directed'),
                     (373, 720, 'free')]
#   'new'   : 短区間を取りこぼさない条件 (slides/dcmss3.py, W11_s2_p95_L20_f3_e12)
#             S_MSS(t) の閾値超え解析が見つけた 3 イベントすべてを directed で捉える
SEGMENTS_T8_NEW = [(0, 48, 'free'), (48, 68, 'directed'),
                   (68, 139, 'free'), (139, 166, 'directed'),
                   (166, 271, 'free'), (271, 373, 'directed'),
                   (373, 720, 'free')]
SEGMENTS_T5 = [(0, 720, 'free')]

VARIANTS = {'recal': SEGMENTS_T8_RECAL, 'paper': SEGMENTS_T8_PAPER,
            'new': SEGMENTS_T8_NEW}
LABELS = {'recal': ('free (0–2.3 h, 6.2–12 h)', 'directed (2.3–6.2 h)'),
          'paper': ('free', 'directed (2.3–2.8 h, 4.5–6.2 h)'),
          'new': ('free', 'directed (0.8–1.1, 2.3–2.8, 4.5–6.2 h)')}

# 既定は主解析 (再較正版)
SEGMENTS_T8 = SEGMENTS_T8_RECAL


def slide_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': ['Helvetica Neue', 'Helvetica', 'Arial',
                        'Hiragino Sans', 'DejaVu Sans'],
        'font.size': 16,
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'text.color': COL_WHITE, 'axes.labelcolor': COL_WHITE,
        'xtick.color': COL_WHITE, 'ytick.color': COL_WHITE,
        'axes.edgecolor': COL_WHITE, 'axes.linewidth': 1.4,
        'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'axes.unicode_minus': False,
        'figure.dpi': 130, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'legend.frameon': False, 'legend.labelcolor': COL_WHITE,
    })
    return plt


def get_tracks():
    from prepare_data import load_tracks
    tracks, report = load_tracks(os.path.join(ROOT, 'data', 'two.csv'))
    return tracks, report


def smoothed_mpd(pos, window=CFG_WINDOW, sigma=CFG_SIGMA):
    """SMPD(t) [px]。DC-MSS Step 1 と同一の定義。"""
    from scipy.ndimage import gaussian_filter1d
    from mss_core import max_pairwise_distance_series
    mpd = max_pairwise_distance_series(np.asarray(pos, float), window)
    return gaussian_filter1d(mpd, sigma, mode='nearest')


def mpd_to_velocity(smpd_px, window=CFG_WINDOW):
    """SMPD [px] -> 局所速度 [um/min]。

    幅 window frames の窓が張る時間は (window-1) * DT_MIN [min] なので、
    等速直線運動 v に対して MPD = v * (window-1) * DT_MIN が成り立つ。
    したがって v = MPD * PX_UM / ((window-1) * DT_MIN)。
    """
    return np.asarray(smpd_px, float) * PX_UM / ((window - 1) * DT_MIN)
