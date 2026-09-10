"""
Vega et al. 2018 Note S1 に従い、MSS slope の分類閾値を
シミュレーションから導出する。

各 duration / motion class について N_SIM 本のトラックを生成し
MSS slope 分布を得る。参照パラメータは論文と同じ:
  immobile : sigma_p 任意 (分布はパラメータ非依存)
  free     : D = 1 px^2/frame
  confined : confinement extent CE = sqrt(4 D dt)/R = 0.7
  directed : drift extent      DE = v dt/sqrt(4 D dt) = 1.0
"""
import numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mss_core import SIMULATORS, mss_analysis

N_SIM = 10000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'results', 'dcmss', 'mss_thresholds.npz')

durations = np.unique(np.concatenate([
    np.arange(20, 151, 1),      # 論文と同じ細かいグリッド
    np.arange(155, 401, 5),
    np.arange(410, 761, 10),    # 本データのトラック長 (720 frames) をカバー
]))

rng = np.random.default_rng(20260819)
slopes = np.empty((len(durations), 4, N_SIM), dtype=np.float32)

t_start = time.time()
for i, N in enumerate(durations):
    for c in range(4):
        pos = SIMULATORS[c](N_SIM, int(N), rng=rng)
        slopes[i, c] = mss_analysis(pos)['mss_slope'].astype(np.float32)
    if i % 20 == 0 or i == len(durations) - 1:
        print(f'[{i+1}/{len(durations)}] N={N}  '
              f'elapsed={time.time()-t_start:.0f}s', flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, durations=durations, slopes=slopes, n_sim=N_SIM)
print('saved', OUT, f'({os.path.getsize(OUT)/1e6:.1f} MB), '
      f'total {time.time()-t_start:.0f}s')
