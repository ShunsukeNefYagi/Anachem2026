# Trajectory analysis for long-term NIR single-particle tracking with Au42 quantum needles

Analysis code accompanying:

> **Long-term single-particle tracking by NIR imaging using Au42 (gold) quantum needles**
> Shunsuke Yagi, Shinjiro Takano, Ryo Nishiyama, Ryosuke Oketani, Tatsuya Tsukuda, Kotaro Hiramatsu
> *bioRxiv* (2026). doi:[10.64898/2026.06.24.734378](https://doi.org/10.64898/2026.06.24.734378)

If you use this code, please cite both the paper and the software — see [`CITATION.cff`](CITATION.cff).

---

## 1. Overview

The paper demonstrates near-infrared single-particle tracking (SPT) of atomically
precise gold quantum needles, Au42(PET)32, inside living HEK293 cells, with
continuous observation for up to 12 h at 1 frame / min (720 frames). This
repository contains the trajectory analysis applied to those recordings — the
part that turns per-frame particle positions into the reported statements about
transitions between **directional** and **diffusive** transport.

Three pipelines, run in this order:

| # | Pipeline | Directory | Question it answers |
|---|---|---|---|
| 1 | **Linking** | [`analysis/linking/`](analysis/linking/) | Given per-frame detections, which detections belong to the same particle — and how much of the apparent motion is the microscope rather than the sample? |
| 2 | **DC-MSS** | [`analysis/dcmss/`](analysis/dcmss/), [`analysis/dcmss_T4/`](analysis/dcmss_T4/) | Within one trajectory, where does the motion type change, and what is each segment (immobile / confined / free / directed)? |
| 3 | **Directional correlation** | [`analysis/direction/`](analysis/direction/) | Do velocity and direction autocorrelations independently support the DC-MSS classes? MSD and MSS use only displacement *magnitude*; this uses the direction they discard. |

Two recordings are analysed and both are included:

- **`data/two.csv`** — a TrackMate track export of two long-lived particles
  (TRACK 5 and TRACK 8, 720 frames each). This is the input to pipelines 2 and 3.
- **`data/linking/laser1traj.xml`** — a TrackMate *detection-only* export
  (9,663 spots over 720 frames, ~13 particles per frame) of a second field of
  view. This is the input to pipeline 1, whose output feeds the DC-MSS analysis
  of track T4 in [`analysis/dcmss_T4/`](analysis/dcmss_T4/).

Units throughout: **length = pixel** (1 px = 10/222 µm = 0.045045 µm),
**time = frame** (1 frame = 1 min). Conversion to µm is applied only at the
reporting stage.

### The DC-MSS method

Segmentation and classification follow **DC-MSS** (divide-and-conquer moment
scaling spectrum):

> Vega AR, Freeman SA, Grinstein S, Jaqaman K. *Multistep Track Segmentation and
> Motion Classification for Transient Mobility Analysis.* Biophysical Journal
> **114**, 1018–1025 (2018). doi:10.1016/j.bpj.2018.01.012

This is an independent reimplementation in [`analysis/dcmss/mss_core.py`](analysis/dcmss/mss_core.py),
not the authors' original MATLAB code, and it is **not** parameter-identical to
the original — see [§7 Deviations](#7-deviations-from-the-original-dc-mss) before
comparing numbers with that paper.

---

## 2. Repository layout

```
.
├── README.md
├── LICENSE                    MIT
├── CITATION.cff               how to cite the code and the paper
├── requirements.txt           pinned pip environment
├── environment.yml            equivalent conda environment
├── pyproject.toml             project metadata + the same pins
│
├── analysis/
│   ├── linking/               detections → drift decomposition → trajectories
│   ├── dcmss/                 MSS/DC-MSS implementation + threshold calibration
│   ├── dcmss_T4/              frozen, self-verifying DC-MSS reproduction for T4
│   └── direction/             directional-correlation analysis
│
├── data/                      minimal inputs (see §4)
│   ├── two.csv
│   ├── linking/
│   ├── dcmss_T4/
│   ├── calibration/
│   └── recalib/
│
└── results/                   all generated output (git-ignored, recreated by the scripts)
```

Every script writes only into `results/`. Nothing writes into `data/`, so a
re-run can never silently modify a published input.

---

## 3. Environment setup

Verified on **Python 3.12.7** (macOS 15, arm64) with NumPy 2.3.3, SciPy 1.16.1,
pandas 2.2.3, Matplotlib 3.10.6.

### pip

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate au42-spt
```

There is nothing to install from this repository itself — the analysis is a set
of ordered scripts, not an importable library. Run each script from **its own
directory**; each one locates `data/` and `results/` relative to its own file, so
no paths need editing.

---

## 4. Data included

Deposited here is the minimum needed to reproduce the reported trajectory
analysis, ~18 MB in total.

| Path | Size | What it is |
|---|---|---|
| `data/two.csv` | 273 KB | TrackMate track export, 2 particles × 720 frames. Input to DC-MSS and directional correlation. |
| `data/linking/laser1traj.xml` | 4.3 MB | TrackMate detection-only export, 9,663 spots / 720 frames. Input to `s1_parse_xml.py`. |
| `data/linking/spot_intensity_square_roi.csv` | 693 KB | Per-spot background-corrected integrated intensity. Product of `s2_intensity.py`; shipped because that step needs the raw image stack. |
| `data/linking/legacy/track_summary_traj.csv` | 60 KB | Track summary from an earlier, uncorrected run. Used only by `s5_compare.py` as a reference point. |
| `data/dcmss_T4/T4_tracks_vib.csv` | 104 KB | Track T4 after removal of the high-frequency common component. Input to `analysis/dcmss_T4/reproduce.py`. |
| `data/dcmss_T4/T4_tracks_raw.csv` | 97 KB | The same track, uncorrected. |
| `data/dcmss_T4/T4_coordinate_correction.csv` | 31 KB | Frame-by-frame `raw − vib` difference, so the correction applied to T4 is auditable. |
| `data/dcmss_T4/expected_T4_segments.csv` | 901 B | Reference segmentation of T4. `reproduce.py` asserts against it. |
| `data/calibration/thr_sig0.14_lf3.npz` | 12 MB | Monte-Carlo MSS-slope distributions (89 segment lengths × 4 classes × 10,000 tracks) used to classify segments. Regenerable, but ~7 min of compute; shipped so the published classification can be reproduced exactly. |
| `data/recalib/sweep_segmentation.json` | 280 KB | Result of a 108-setting segmentation-parameter sweep, summarised by `apply_recalibrated.py`. |

### What is *not* included

- **The raw image stacks.** `laser1traj.tif` is 2.9 GB (720 frames × 1431²).
  `s2_intensity.py` is the only script that reads it, and it exits with a clear
  message if the file is absent; its output is shipped instead. Available from
  the corresponding author on request.
- **Detection.** Spots were detected in [TrackMate](https://imagej.net/plugins/trackmate/)
  v8.1.5 (Fiji), and its XML/CSV exports are the entry point here. Detection
  parameters are in the paper's Methods.
- **Three of the four threshold tables** (`thr_sig0.00/0.28/0.42_lf3.npz`).
  They are used only for the sensitivity check in `apply_recalibrated.py`, which
  skips missing ones with a printed note. Regenerate with `gen_thresholds.py`.

### A note on `20260825_005`

Comments and plot legends in `analysis/linking/s3*.py` refer to `20260825_005`.
That is our internal identifier for a **separate control measurement on the same
microscope**: immobilised particles tracked for 12 h at 1 frame / min. It supplies
the instrument's measured drift (12 h: x −0.635 px, y +1.757 px) and its 8.00 min
vibration line (amplitude 3.742 px), which `s3d_validate.py` uses as a
known-magnitude reference to decide whether an observed common component can be
the instrument at all. The numbers are hard-coded in `s3d_validate.py`; that
recording is not otherwise part of this repository.

---

## 5. Reproducing the analysis

### 5.1 Quickest check — the frozen T4 reproduction (~5 s)

```bash
cd analysis/dcmss_T4
python3 reproduce.py
```

This is self-verifying: it asserts that the segment boundaries and class labels
match `data/dcmss_T4/expected_T4_segments.csv` and that the MSS slopes agree to
< 1e-10, then prints `PASS`. Expected output:

```
 segment  start_frame  end_frame_exclusive class_label    S_MSS  max_lag_min
       1            0                   48        free 0.459563         16.0
       2           48                   68    directed 0.852339          6.0
       3           68                  113        free 0.655466         15.0
       4          113                  419    directed 0.733390         30.0
       5          419                  720        free 0.576018         30.0
PASS: reference boundaries/classes reproduced; max slope error 2.22e-16
```

Note that segment 3 sits 0.0013 below the free|directed threshold (0.656748).
Class labels this close to a boundary should not be read as settled physical
states.

### 5.2 Linking pipeline (roughly 20–30 min on a laptop)

Run from `analysis/linking/`, in order:

```bash
cd analysis/linking
python3 s1_parse_xml.py              # XML → results/linking/spots_raw.csv
# s2_intensity.py is skipped: needs the raw image stack (see §4)
python3 s3_drift.py                  # naive common-motion estimate + its own null
python3 s3b_diagnose.py              # is the common component the stage, or biology?
python3 s3c_rigid.py                 # restrict "common" to spatially uniform translation
python3 s3d_validate.py              # cross-validate; compare against measured instrument drift
python3 s3e_decompose.py             # split into instrument (<30 min) and sample (≥30 min)
python3 s4_track.py raw              # link, uncorrected coordinates
python3 s4_track.py vib              # link, instrument vibration removed   ← used downstream
python3 s4_track.py full             # link, all common motion removed
python3 s4_track.py common           # same correction as full, named for s6   ← s6 requires this
python3 s5_compare.py                # compare the three coordinate systems
python3 s6_common_drift_analysis.py  # full-band common-displacement removal, evaluated separately
```

`s6_common_drift_analysis.py` reads `tracks_common.csv` and
`track_summary_common.csv`, so it fails with `FileNotFoundError` if the `common`
variant was not linked. `common` applies the same correction as `full`; it is a
separate output name so that `s5_compare.py`'s three-way comparison and `s6`'s
analysis do not overwrite each other's files.

`s4_track.py` also accepts `--bootstrap` to add the linking-uncertainty resampling
(slower).

**End-to-end check.** `results/linking/tracks_vib.csv` regenerates the shipped T4
coordinates:

```bash
cd ../..
python3 - <<'EOF'
import pandas as pd, numpy as np
new = pd.read_csv('results/linking/tracks_vib.csv')
ref = pd.read_csv('data/dcmss_T4/T4_tracks_vib.csv')
g = new[new.TRACK_ID == 4].sort_values('FRAME')
for c in ('X_px', 'Y_px'):
    print(c, np.abs(g[c].to_numpy() - ref[c].to_numpy()).max())
EOF
```

Both differences are ~2e-13 px, i.e. CSV float round-trip only.

### 5.3 DC-MSS on the two-particle recording (~10 min)

```bash
cd analysis/dcmss
python3 apply_recalibrated.py        # ~5 s — the main analysis; uses the shipped thresholds
python3 derive_thresholds.py         # ~9 min — only needed for run_analysis.py below
python3 run_analysis.py              # DC-MSS with the original paper's parameters
```

`apply_recalibrated.py` is the primary result and needs no threshold generation.
It should print, for TRACK 8:

```
free      0–140 min
directed  140–374 min
free      374–720 min
```

and a single `free` segment spanning 0–720 min for TRACK 5.

### 5.4 Directional correlation (~2 min)

```bash
cd analysis/direction
python3 q7_direction.py              # whole-track and fixed 60-min windows
python3 q7b_control.py               # controls: interpolation, localization error, drift
python3 seg_direction.py             # per-DC-MSS-segment direction correlation
python3 mss_vs_direction.py          # MSS spectrum and direction correlation side by side
```

---

## 6. What each script does

### 6.1 `analysis/linking/` — positions to trajectories

The non-obvious part of this pipeline is that a naive ensemble-median drift
correction is actively harmful here. With only ~13 particles per frame, the
median of independent Brownian steps does not average to zero; accumulating it
manufactures a drift of tens of pixels that was never there. Worse, the largest
genuine common component in this sample is **biology** — particles converging on
a localised intracellular domain, which is one of the paper's findings — and
subtracting it would erase the result. Steps `s3` through `s3e` exist to separate
those three things before any linking is done.

| Script | Role |
|---|---|
| `common.py` | Shared constants and the linking algorithm itself. Priority-greedy matching with gap filling: cost `C = w_d·d/δmax + w_i·(r−1)/(Rmax−1)`; tracks with more detections are assigned first so short tracks cannot steal spots; the search radius relaxes by γ per consecutive missing frame. δmax = 3.0 µm, (w_d, w_i) = (0.7, 0.3), Rmax = 2.0, Gmax = 3, γ = 0.3, minimum 3 detections. |
| `s1_parse_xml.py` | Parses the TrackMate detection XML into `spots_raw.csv`. Coordinates are in pixels (`pixelwidth="1.0"` in the export). |
| `s2_intensity.py` | Background-corrected integrated intensity per spot, from a 0.4 µm square ROI, with background from the top-left 100×100 px of each frame. Run *before* any coordinate correction: intensity is a property of the raw pixel at that time, so it must be sampled at the detected position. **Needs the raw image stack**; its output is shipped instead. |
| `s3_drift.py` | Estimates the common component and, crucially, its own noise floor. Splits contributing particles into random halves A/B per frame: `Cov(med_A, med_B)` is an unbiased estimate of the common variance, while `(med_A − med_B)/2` is a noise-only series whose cumulative sum is exactly the spurious drift this estimator produces when there is none. Wiener shrinkage then passes only the frequency bands where signal exceeds that floor. |
| `s3b_diagnose.py` | Tests whether the detected common component is the stage or the sample: rigid-translation consistency across sub-fields, affine decomposition (divergence and rotation of the displacement field), the existence of near-stationary tracks in raw coordinates, and convergence toward an accumulation centre. |
| `s3c_rigid.py` | Restricts "common" to the spatially uniform part, by splitting particles at the per-frame median position instead of randomly. Rigid translation enters both halves equally; position-dependent collective flow does not. |
| `s3d_validate.py` | Two independent checks: cross-validation (estimate drift from group A, apply to group B, see whether B's displacement variance falls) and comparison against the same microscope's measured 12 h drift and 8 min vibration line. |
| `s3e_decompose.py` | Splits the common component at a 30 min period into instrument (high-frequency mechanical vibration) and sample (low-frequency collective transport), and emits `raw`, `vib`, and `full` coordinate variants. `vib` — vibration removed, biology retained — is the one used downstream. |
| `s4_track.py` | Applies `common.track()` to a chosen coordinate variant. `python3 s4_track.py {raw\|vib\|full\|common} [--bootstrap]`. With `--bootstrap`, resamples δmax to report per-track stability. |
| `s5_compare.py` | Compares the three coordinate systems against each other and against the earlier uncorrected run. |
| `s6_common_drift_analysis.py` | Evaluates full-band common-displacement removal while holding the raw link pairs fixed, so that particles changing identity under re-linking cannot be mistaken for improvement. |

### 6.2 `analysis/dcmss/` — segmentation and classification

| Script | Role |
|---|---|
| `mss_core.py` | The implementation. MSS analysis (moments q = 0…6, γ(q) by OLS of log μq on log 4τ, then S_MSS as the slope of γ(q) vs q), the four motion simulators used for calibration, `MSSThresholds` (length-dependent thresholds and model misclassification probabilities), and the three DC-MSS steps: initial segmentation from the smoothed maximum-pairwise-distance series, MSS classification, and merging of adjacent segments. |
| `prepare_data.py` | Loads `data/two.csv`, drops duplicate frames keeping the highest-QUALITY spot, and linearly interpolates missing frames (DC-MSS assumes a uniform time series). TRACK 8 has 4 duplicated frames; TRACK 5 has 6 missing frames (0.83%), at most 2 consecutive. Reports exactly what it changed. |
| `derive_thresholds.py` | Generates MSS-slope distributions under the *original paper's* conditions (τmax = N/4, no localization error), 217 durations × 4 classes × 10,000 tracks. ~9 min, writes a 30 MB npz to `results/dcmss/`. Needed only by `run_analysis.py`. |
| `gen_thresholds.py` | Generates the recalibrated tables actually used here: durations 6–760, τmax = N/4 and N/3, σp/Σ_D ∈ {0, 0.14, 0.28, 0.42}. ~7 min for all eight files. Fixed seeds per class (immobile 101, free 202, confined 303, directed 404). Writes to `results/calibration/` so the shipped table is never overwritten. **Regenerating a single file by trimming this script will not reproduce it** — the random-number consumption order would change. |
| `run_analysis.py` | DC-MSS with the original paper's default parameters, plus per-track MSD curves, D and σ estimates, and the covariance-based localization-error estimator of Vestergaard et al. (2014). Requires `derive_thresholds.py` first. |
| `apply_recalibrated.py` | **The main DC-MSS analysis.** Applies the selected recalibrated condition (window = 5, σ = 2.0, peak percentile = 97.5, min segment = 20, τmax = N/3, σp/Σ_D = 0.14) to both tracks, summarises the 108-setting parameter sweep, and reports sensitivity across σp ratios. |

### 6.3 `analysis/dcmss_T4/` — the frozen T4 reproduction

A deliberately narrow, self-verifying package for the single trajectory (T4) that
the paper's segmentation figure is built on. It starts from corrected coordinates
and reruns only the DC-MSS steps, checking its result against a stored reference.

| File | Role |
|---|---|
| `dc_mss.py` | The five definitions (`extend_short`, `step1`, `Cfg`, `step3`, `dc_mss3`) used for T4, extracted verbatim from the working script with site-specific plotting and data imports removed. |
| `config.json` | Every fixed parameter, in one auditable place. |
| `reproduce.py` | Reruns the analysis, records each stage (candidate boundaries → after extension → after KS merge → final), asserts equality with the reference, and writes diagnostics, the moment spectrum, the threshold grid, and two figures. |

The T4 parameters differ from `apply_recalibrated.py` (window 11 vs 5, percentile
95 vs 97.5, short-segment floor 12): they are the condition chosen for that
recording, and `config.json` records them so the two are not confused.

### 6.4 `analysis/direction/` — the independent check

MSD and MSS scaling use only the magnitude of displacement. Direction
autocorrelation uses the information they discard, and separates the same three
regimes on a different principle: directed transport gives positive velocity
autocorrelation with turning angles biased toward 0°; confinement gives negative
autocorrelation (rebound) biased toward 180°; free diffusion gives zero
correlation and a uniform turning-angle distribution. Localization error also
produces a negative lag-1 correlation, so observed positive values are a *lower*
bound on true persistence.

| Script | Role |
|---|---|
| `common.py` | Track loading, 60-min windowing, single-power-law MSD fits with goodness-of-fit, per-window MSS. |
| `slide_common.py` | Scale constants, plotting style, and the three published TRACK 8 segmentations (`recal`, `paper`, `new`) as named variants, so a figure states which segmentation it used. |
| `q7_direction.py` | Velocity autocorrelation `C(n)`, direction autocorrelation `Cu(n) = <cos θ>`, turning-angle asymmetry, per-60-min-window values, and drift-speed estimation — each calibrated against ground-truth simulations. |
| `q7b_control.py` | Controls for the confounds: linear interpolation of missing frames (recomputes with interpolation-touching steps excluded), localization error, and whether persistence survives removing TRACK 8's directed phase. |
| `seg_direction.py` | The same quantities per DC-MSS segment rather than per fixed window, with 95% intervals from matched simulations, a test of the constant-drift prediction (`C(n)` flat at `DE²/(1+DE²)`), and a thirds-split of each directed segment to see whether the drift direction holds. |
| `mss_vs_direction.py` | Puts S_MSS, its duration-dependent threshold, and the direction correlations on one figure — showing what the classification actually turned on. `python3 mss_vs_direction.py {recal\|paper}`; `paper` needs `derive_thresholds.py` first. |

---

## 7. Deviations from the original DC-MSS

This implementation departs from Vega et al. (2018) in ways that matter when
comparing numbers:

- Classification lag changed from **N/4 to N/3**.
- Thresholds recalibrated to include a **localization-error ratio σp/Σ_D = 0.14**;
  the original assumes none.
- Short-segment extension floor lowered from **16 to 12 positions**.
- Duration grid extended below the original lower bound of 20 (down to 6) and up
  to 760, to cover 720-frame tracks.
- `S = 0.5` is **not** used as a fixed free/directed boundary; all three
  boundaries are duration-dependent and read from the calibration table.

Two interpretive cautions carried over from the analysis notes:

- The KS test used to merge adjacent segments compares distributions of
  **pairwise distances, which are not independent**. Its p-value is used here as
  an operating rule reproducing the original procedure, not as a calibrated error
  rate.
- `p_mis` is the tail area of the neighbouring class's empirical distribution. It
  is not the posterior probability that a given segment is misclassified, and a
  simulated `p_mis = 0` does not mean the error probability is zero.

Segmentation by displacement magnitude, MSS class assignment, and verification of
single self-similar behaviour are three different questions. A segment that was
not split is not thereby guaranteed to be internally homogeneous in α.

---

## 8. License

MIT — see [`LICENSE`](LICENSE). You may use, modify, and redistribute this code,
including commercially, provided the copyright notice and license text are
retained. The code is provided without warranty.

The data in `data/` is released under the same terms for the purpose of
reproducing and building on the published analysis; please cite the paper.

---

## 9. Contact

Questions about the code or requests for the raw image stacks: Shunsuke Yagi
(Department of Chemistry, Kyushu University) — `@`.
