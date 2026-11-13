# Linking and DC-MSS analysis of a long-term Adot trajectory

Code for the single-particle trajectory analysis of **Adot (A)** (Fig. 4C, 4D) in

> *Long-term single-particle tracking by NIR imaging using Au42 gold quantum needles* (2026).

```
TrackMate detections ──► linking ──► particle A ──► DC-MSS
  data/laser1traj.xml     src/linking.py              src/dcmss_analysis.py
```

Result for particle A (1 frame = 1 min, 720 frames):

| Segment (min) | Class | S_MSS |
|---|---|---|
| 0–139 | free (diffusive) | 0.568 |
| 140–373 | **directed** | 0.689 |
| 374–720 | free (diffusive) | 0.458 |

DC-MSS follows Vega et al., *Biophys. J.* **114**, 1018–1025 (2018),
doi:10.1016/j.bpj.2018.01.012 (independent Python implementation).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py              # linking + DC-MSS
python verify_reproduction.py       # re-run and compare with reference/
```

Outputs are written to `results/` (linked tracks, particle A trajectory, DC-MSS
segments and figure). Segment tables report lengths in µm and time in min.

`verify_reproduction.py` is a reproducibility check: it confirms that the code
regenerates the stored outputs of the original analysis run in `reference/`. It
does not by itself validate the linking or the DC-MSS classification.

## Repository layout

```
run_pipeline.py          detections -> linking -> particle A -> DC-MSS
verify_reproduction.py   re-runs the pipeline and checks it against reference/
src/                     linking, MSS / DC-MSS analysis, threshold generation
data/                    TrackMate detections, spot intensities, MSS threshold table
reference/               published results (linked tracks, particle A segments), for verification
```

The raw image stack (2.9 GB) is not included; it is available from the authors on
request.

## Citation

If you use this code, please cite the paper above. Citation metadata is in
[`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

## License

[MIT](LICENSE) © 2026 Shunsuke Yagi, Shinjiro Takano, Ryo Nishiyama,
Ryosuke Oketani, Tatsuya Tsukuda, Kotaro Hiramatsu
