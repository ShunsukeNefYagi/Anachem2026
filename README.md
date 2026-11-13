# Linking and DC-MSS analysis of a long-term Adot trajectory

Code for the single-particle trajectory analysis of **Adot (A)** (Fig. 4C, 4D) in

> *Long-term single-particle tracking by NIR imaging using Au42 gold quantum needles* (2026).

The pipeline goes from per-frame spot detections to a segmented, motion-classified
trajectory:

```
TrackMate detections ──► linking (displacement + intensity cost) ──► particle A ──► DC-MSS
  data/laser1traj.xml       src/linking.py                                         src/dcmss_analysis.py
```

Result for particle A (1 frame = 1 min, 720 frames):

| Segment (min) | Class | S_MSS (this linking) | S_MSS (TrackMate track, published figure) | free/directed threshold |
|---|---|---|---|---|
| 0–140 | free (diffusive) | 0.568 | 0.568 | 0.620 |
| 140–374 | **directed** | 0.689 | 0.689 | 0.605 |
| 374–720 | free (diffusive) | 0.458 | 0.459 | 0.594 |

Bounds are half-open `[start, end)`: 140–374 contains the positions at 140, …, 373 min.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py              # linking + DC-MSS, ~5 s
python verify_reproduction.py       # re-run and compare with the reported values, ~10 s
```

`run_pipeline.py` writes everything to `results/`:

| File | Content |
|---|---|
| `results/linking/tracks.csv` | all linked tracks (one row per spot, with link cost and confidence) |
| `results/particleA_track.csv` | particle A |
| `results/dcmss/particleA_segments.csv` | DC-MSS segments and per-segment values |
| `results/dcmss/particleA_stages.json` | boundaries after each DC-MSS step |
| `results/dcmss/particleA_dcmss.png` | trajectory coloured by class, S_MSS per segment |

## Repository layout

```
run_pipeline.py          detections -> linking -> particle A -> DC-MSS
verify_reproduction.py   re-runs the pipeline and checks it against reference/
src/
  linking.py             TrackMate XML parser and the linking algorithm
  spot_intensity.py      spot intensities from the raw image stack (optional)
  tracks_io.py           trajectory loading, duplicate removal, gap interpolation
  mss_core.py            MSS analysis, threshold tables, DC-MSS steps 1-3, simulators
  dcmss_analysis.py      DC-MSS conditions used in the paper, per-segment table, figure
  gen_thresholds.py      Monte-Carlo generation of the MSS classification thresholds
data/
  laser1traj.xml                   TrackMate detections, all spots (9,663 spots, 720 frames)
  spot_intensity_square_roi.csv    background-corrected intensity of each spot
  thr_sig0.14_lf3.npz              MSS slope distributions for classification
  particleA_trackmate.csv          particle A as linked by TrackMate (see below)
reference/
  linking_tracks.csv               linking output of the original analysis run
  figure3_segment_summary.csv      per-segment values used for the published figure
```

## Data

* **Detections** (`data/laser1traj.xml`): TrackMate v8.1.5 export of the
  Laplacian-of-Gaussian detections in every frame (detection only, no tracks).
  Coordinates are in pixels; 222 px = 10 µm. All spots in the field are needed, not
  only those of particle A, because linking assigns spots competitively.
* **Spot intensities** (`data/spot_intensity_square_roi.csv`): integrated intensity in a
  0.4 µm square ROI around each spot minus the background (mean of the top-left
  100×100 px of frame 0). Made by `src/spot_intensity.py` from the raw 720-frame stack
  (2.9 GB, not included; available from the authors on request).
* **Thresholds** (`data/thr_sig0.14_lf3.npz`): MSS-slope distributions of 10,000
  simulated tracks for each of 89 segment lengths (6–760 positions) and each of the four
  motion classes. Regenerate with `python src/gen_thresholds.py` (~7 min).
* **`data/particleA_trackmate.csv`**: the same particle linked by TrackMate's own
  tracker (rows of TrackMate track 8). It differs from our linking by ≈0.1 px in
  position (median); 4 frames contain two spots, of which the higher-QUALITY one is
  kept. `python run_pipeline.py --trackmate` analyses it as well.

## Methods

### Linking (`src/linking.py`)

The cost of linking the last spot of a track to a spot in the next frame is

    C = w_d · d / d_max + w_i · (r − 1) / (R_max − 1)

where *d* is the displacement and *r* = max(I₁, I₂)/min(I₁, I₂) the intensity ratio.

| Parameter | Value |
|---|---|
| maximum displacement d_max | 3.0 µm |
| maximum intensity ratio R_max | 2.0 |
| weights (w_d, w_i) | (0.7, 0.3) |
| gap closing G_max | 3 frames; search radius × (1 + 0.3·gap) |
| minimum track length | 3 detections |

Tracks are processed in order of their number of detections (longer first) and each
takes its lowest-cost candidate. The confidence of a link is 1 − C_best/C_second.
Particle A is the track whose frame-0 spot is at (1327.7, 255.7) px; it is detected in
all 720 frames (no gaps, mean link confidence 0.996). Coordinates are not drift
corrected.

### DC-MSS (`src/mss_core.py`, `src/dcmss_analysis.py`)

DC-MSS follows Vega et al., *Biophys. J.* **114**, 1018–1025 (2018),
doi:10.1016/j.bpj.2018.01.012. This is an independent Python implementation, not the
authors' MATLAB code.

1. **Segmentation.** Maximum pairwise distance in a rolling window, Gaussian smoothing,
   candidate switch points at local maxima of the normalised change above a percentile,
   extension of segments of 16–19 positions to 20, merging of neighbours whose
   pairwise-distance distributions do not differ (KS, α = 0.05).
2. **Classification.** For each segment ≥ 20 positions, moments of order 0–6 of the
   displacement, μ_m(τ) ∝ τ^{α_m}, then S_MSS = slope of α_m vs m. The class
   (immobile / confined / free / directed) follows from segment-length-dependent
   thresholds derived from simulated tracks.
3. **Refinement.** Adjacent segments of which one is unclassified (< 20 positions) or
   both have the same class are merged and reclassified. A merge is accepted if the
   merged class has the same or lower mobility, or otherwise if its model
   misclassification probability is not larger than that of the separate segments.

Conditions, and how they differ from the defaults of Vega et al. (which were set for
33 Hz imaging; here 1 frame = 1 min):

| | Vega et al. | this analysis |
|---|---|---|
| MPD window | 11 positions | **5** |
| switch-point percentile | 95 | **97.5** |
| Gaussian σ | 2 frames | 2 frames |
| minimum segment | 20 positions | 20 positions |
| MSS max lag | min(N/4, 30) | **min(N/3, 30)** |
| localization error in threshold simulations | none | **σ_p/Σ_D = 0.14** |
| threshold segment lengths | 20–500 | **6–760** |

Thresholds come from simulated immobile, confined (CE = 0.7), free and directed
(DE = 1.0) tracks (`src/gen_thresholds.py`, fixed seeds). In
`particleA_segments.csv`, `gen_D2_px2` is the generalised coefficient of the 2nd moment
(μ₂ = D₂(4τ)^α₂); it equals a diffusion coefficient only when α₂ = 1.

## Reproducibility

`python verify_reproduction.py` re-runs the pipeline and checks:

1. **Linking**: every spot is assigned to the same track as in the original analysis run
   (`reference/linking_tracks.csv`, 207 tracks).
2. **DC-MSS on the TrackMate track of particle A**: boundaries, classes, S_MSS and the
   other per-segment values equal those of the published figure
   (`reference/figure3_segment_summary.csv`) to < 1e-10.
3. **DC-MSS on particle A from this linking**: same segments and classes as the paper.
4. **Threshold table**: checksum; with `--regen-thresholds`, regenerated bit-identically.

With `--tif <laser1traj.tif>` it also recomputes the spot intensities from the raw stack.

## License

To be added.
