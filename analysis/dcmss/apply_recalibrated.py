"""再較正した DC-MSS 条件を実測軌跡へ適用し、数値結果と図を保存する。"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))     # repository root
sys.path.insert(0, HERE)

from mss_core import (CLASS_COLORS, CLASS_NAMES, DCMSSConfig, MSSThresholds,
                      dc_mss)
from prepare_data import load_tracks

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'results', 'dcmss')
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)

SWEEP_PATH = os.path.join(DATA, 'recalib', 'sweep_segmentation.json')
THRESHOLD_DIR = os.path.join(DATA, 'calibration')
FIG_PATH = os.path.join(OUT, 'figs', 'fig15_recalibration.png')
SEG_PATH = os.path.join(OUT, 'recalibrated_segments.csv')
SUMMARY_PATH = os.path.join(OUT, 'recalibrated_summary.json')
VALIDATION_PATH = os.path.join(OUT, 'validation_selected.json')

SELECTED = DCMSSConfig(window=5, sigma=2.0, peak_pctl=97.5, min_seg=20,
                       lag_frac=3, name='recalibrated')
BASELINE = DCMSSConfig(window=11, sigma=2.0, peak_pctl=95.0, min_seg=20,
                       lag_frac=3, name='baseline_shape_lf3')
PRIMARY_SIGMA_RATIO = 0.14


def aggregate_sweep(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['window'], row['sigma'], row['pctl'], row['min_seg'])
        grouped[key].append(row)
    output = []
    for key, group in grouped.items():
        switching = [r for r in group if r['dataset'].startswith('switch_')]
        homogeneous = [r for r in group if r['dataset'].startswith('homog_')
                       and r['dataset'] != 'homog_free_Dt']
        dt_free = next(r for r in group if r['dataset'] == 'homog_free_Dt')
        output.append({
            'key': key,
            'switch_tpr': float(np.mean([r['frac_1'] for r in switching])),
            'switch_accuracy': float(np.mean([r['accuracy'] for r in switching])),
            'switch_error_median': float(np.mean(
                [r['sw_err_median'] for r in switching])),
            'homogeneous_fpr': float(np.mean([1 - r['frac_0']
                                               for r in homogeneous])),
            'homogeneous_accuracy': float(np.mean([r['accuracy']
                                                    for r in homogeneous])),
            'free_Dt_fpr': float(1 - dt_free['frac_0']),
            'free_Dt_accuracy': float(dt_free['accuracy']),
        })
    return output


def segment_rows(track_id, pos, segs):
    rows = []
    for index, seg in enumerate(segs, 1):
        sub = pos[seg['start']:seg['end']]
        steps = np.linalg.norm(np.diff(sub, axis=0), axis=1)
        net = np.linalg.norm(sub[-1] - sub[0])
        straightness = net / steps.sum() if steps.sum() else np.nan
        de = (seg['v_net'] / np.sqrt(4 * seg['D'])
              if seg['D'] == seg['D'] and seg['D'] > 0 else np.nan)
        rows.append({
            'track': track_id,
            'seg': index,
            'start_min': seg['start'],
            'end_min': seg['end'],
            'dur_min': seg['len'],
            'cls': CLASS_NAMES[seg['cls']],
            'mss_slope': seg['mss_slope'],
            'D_px2_per_min': seg['D'],
            'conf_radius_px': seg['conf_radius'],
            'v_net_px_per_min': seg['v_net'],
            'straightness': straightness,
            'DE': de,
        })
    return rows


def classify_for_sensitivity(tracks):
    """sigma_p/Sigma_D を振ったときの分類の感度。

    公開リポジトリには主解析で使う thr_sig0.14_lf3.npz のみを同梱している。
    残る 3 条件は analysis/dcmss/gen_thresholds.py で生成でき、生成物を
    data/calibration/ に置けばこのループが自動的に拾う。
    """
    output = {}
    for ratio in (0.0, 0.14, 0.28, 0.42):
        path = os.path.join(THRESHOLD_DIR, f'thr_sig{ratio:.2f}_lf3.npz')
        if not os.path.exists(path):
            print(f'  [skip] sigma ratio {ratio:.2f}: {path} not present '
                  '(run gen_thresholds.py to create it)')
            continue
        threshold = MSSThresholds(path)
        output[f'{ratio:.2f}'] = {}
        for track_id, track in tracks.items():
            segs = dc_mss(track['pos'], threshold, SELECTED)['segs']
            output[f'{ratio:.2f}'][str(track_id)] = [
                {'start': int(s['start']), 'end': int(s['end']),
                 'class': CLASS_NAMES[s['cls']]}
                for s in segs
            ]
    return output


def plot_classified(ax, pos, segs, title):
    for seg in segs:
        sub = pos[seg['start']:seg['end']]
        points = sub.reshape(-1, 1, 2)
        lines = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection(LineCollection(
            lines, colors=CLASS_COLORS[seg['cls']], linewidths=1.8))
    ax.scatter(pos[0, 0], pos[0, 1], s=28, c='black', marker='o', zorder=3,
               label='start')
    ax.scatter(pos[-1, 0], pos[-1, 1], s=36, c='black', marker='x', zorder=3,
               label='end')
    ax.autoscale()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.grid(alpha=0.2)


def make_figure(aggregates, tracks, primary_results, validation=None):
    selected_key = (SELECTED.window, SELECTED.sigma, SELECTED.peak_pctl,
                    SELECTED.min_seg)
    baseline_key = (BASELINE.window, BASELINE.sigma, BASELINE.peak_pctl,
                    BASELINE.min_seg)
    selected = next(r for r in aggregates if r['key'] == selected_key)
    baseline = next(r for r in aggregates if r['key'] == baseline_key)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    ax = axes[0, 0]
    x = [r['homogeneous_fpr'] for r in aggregates]
    y = [r['switch_tpr'] for r in aggregates]
    c = [r['free_Dt_fpr'] for r in aggregates]
    scatter = ax.scatter(x, y, c=c, cmap='viridis_r', s=24, alpha=0.8)
    for item, marker, color, label in ((baseline, 'X', '#d1495b', 'baseline'),
                                        (selected, '*', '#f4a261', 'selected')):
        ax.scatter(item['homogeneous_fpr'], item['switch_tpr'], marker=marker,
                   c=color, s=145, edgecolor='black', linewidth=0.7, label=label)
    ax.set_xlabel('False-positive rate: stationary tracks')
    ax.set_ylabel('One-switch detection rate')
    ax.set_title('A  Segmentation-parameter sweep (108 settings)')
    ax.legend(frameon=False)
    cb = fig.colorbar(scatter, ax=ax, pad=0.02)
    cb.set_label('False-positive rate: free D(t)')
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    names = ['Switch\nTPR', 'Stationary\nFPR', 'Free D(t)\nFPR']
    if validation:
        baseline_eval = validation['configs']['baseline']['aggregate']
        selected_eval = validation['configs']['selected']['aggregate']
        panel_title = 'B  Independent validation (1,000 per dataset)'
    else:
        baseline_eval, selected_eval = baseline, selected
        panel_title = 'B  Performance of selected setting'
    base_values = [baseline_eval['switch_tpr'],
                   baseline_eval['homogeneous_fpr'],
                   baseline_eval['free_Dt_fpr']]
    sel_values = [selected_eval['switch_tpr'],
                  selected_eval['homogeneous_fpr'],
                  selected_eval['free_Dt_fpr']]
    xx = np.arange(3)
    ax.bar(xx - 0.18, base_values, 0.36, label='baseline', color='#9aa6ac')
    ax.bar(xx + 0.18, sel_values, 0.36, label='selected', color='#2a9d8f')
    ax.set_xticks(xx, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction')
    ax.set_title(panel_title)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)

    plot_classified(axes[1, 0], tracks[5]['pos'], primary_results[5]['segs'],
                    'C  TRACK 5: free, 0–720 min')
    plot_classified(axes[1, 1], tracks[8]['pos'], primary_results[8]['segs'],
                    'D  TRACK 8: free / directed / free')
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    with open(SWEEP_PATH, encoding='utf-8') as f:
        sweep_rows = json.load(f)
    aggregates = aggregate_sweep(sweep_rows)
    tracks, _ = load_tracks()
    threshold = MSSThresholds(os.path.join(
        THRESHOLD_DIR, f'thr_sig{PRIMARY_SIGMA_RATIO:.2f}_lf3.npz'))

    results = {}
    rows = []
    state = {}
    for track_id, track in tracks.items():
        result = dc_mss(track['pos'], threshold, SELECTED)
        results[track_id] = result
        track_rows = segment_rows(track_id, track['pos'], result['segs'])
        rows.extend(track_rows)
        durations = defaultdict(int)
        for row in track_rows:
            durations[row['cls']] += row['dur_min']
        state[str(track_id)] = {
            'n_segments': len(track_rows),
            'n_switches': len(track_rows) - 1,
            'duration_min': dict(durations),
            'fraction': {key: value / len(track['pos'])
                         for key, value in durations.items()},
        }

    frame = pd.DataFrame(rows)
    frame.to_csv(SEG_PATH, index=False)

    selected_key = (SELECTED.window, SELECTED.sigma, SELECTED.peak_pctl,
                    SELECTED.min_seg)
    baseline_key = (BASELINE.window, BASELINE.sigma, BASELINE.peak_pctl,
                    BASELINE.min_seg)
    selected_metrics = dict(next(r for r in aggregates
                                 if r['key'] == selected_key))
    baseline_metrics = dict(next(r for r in aggregates
                                 if r['key'] == baseline_key))
    for metrics in (selected_metrics, baseline_metrics):
        metrics['key'] = list(metrics['key'])

    summary = {
        'selected_config': SELECTED.as_dict(),
        'primary_sigma_p_over_SigmaD': PRIMARY_SIGMA_RATIO,
        'sweep_n_settings': len(aggregates),
        'sweep_n_trajectories': len(aggregates) * 9 * 100,
        'selected_metrics': selected_metrics,
        'baseline_metrics': baseline_metrics,
        'state': state,
        'localization_error_sensitivity': classify_for_sensitivity(tracks),
    }
    validation = None
    if os.path.exists(VALIDATION_PATH):
        with open(VALIDATION_PATH, encoding='utf-8') as f:
            validation = json.load(f)
        summary['independent_validation'] = validation
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    make_figure(aggregates, tracks, results, validation)
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(FIG_PATH)


if __name__ == '__main__':
    main()
