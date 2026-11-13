"""Background-corrected integrated intensity of every detected spot.

    python src/spot_intensity.py --tif path/to/laser1traj.tif

Needs the raw 720-frame image stack (2.9 GB, not distributed).  Its output is
shipped as data/spot_intensity_square_roi.csv and is the intensity used by the
linking cost (linking.py), so this step only needs to be re-run from raw images.

Method
  - square ROI of side 0.4 um centred on each spot (half side 4.44 px -> usually 9x9 px)
  - pixel-centre convention: POSITION_X = (col + 0.5) * pixelwidth -> col = X_px - 0.5
  - background b = mean of the top-left 100x100 px of frame 0 [counts/px];
    I_corrected = sum(ROI) - b * n_pixels
Output: results/linking/spot_intensity_square_roi.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from linking import parse_trackmate_xml
from units import PIXEL_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'linking')
ROI_SIZE_UM = 0.4
BG_REGION = 100


def main():
    import tifffile

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tif', required=True, help='raw image stack (720 frames)')
    tif_path = ap.parse_args().tif
    os.makedirs(OUT, exist_ok=True)

    df = parse_trackmate_xml()
    df['cx_px'] = df['X_px'] - 0.5
    df['cy_px'] = df['Y_px'] - 0.5
    df['x_um'] = df['X_px'] * PIXEL_SIZE
    df['y_um'] = df['Y_px'] * PIXEL_SIZE
    half_px = (ROI_SIZE_UM / 2.0) / PIXEL_SIZE
    by_frame = {f: g for f, g in df.groupby('FRAME')}

    records = []
    with tifffile.TiffFile(tif_path) as tif:
        pages = tif.pages
        h, w = pages[0].shape
        bg_ref = float(pages[0].asarray()[:BG_REGION, :BG_REGION].mean())
        for fi in range(len(pages)):
            g = by_frame.get(fi)
            if g is None:
                continue
            img = pages[fi].asarray()
            for _, spot in g.iterrows():
                cx, cy = float(spot['cx_px']), float(spot['cy_px'])
                c0 = max(0, int(np.floor(cx - half_px)))
                c1 = min(w, int(np.ceil(cx + half_px)) + 1)
                r0 = max(0, int(np.floor(cy - half_px)))
                r1 = min(h, int(np.ceil(cy + half_px)) + 1)
                yy, xx = np.ogrid[r0:r1, c0:c1]
                mask = (np.abs(xx - cx) <= half_px) & (np.abs(yy - cy) <= half_px)
                pixels = img[r0:r1, c0:c1][mask].astype(np.float64)
                n_px, total = int(pixels.size), float(pixels.sum())
                records.append({
                    'spot_id': int(spot['ID']), 'frame': fi,
                    'x_um': round(float(spot['x_um']), 4),
                    'y_um': round(float(spot['y_um']), 4),
                    'cx_px': round(cx, 3), 'cy_px': round(cy, 3),
                    'roi_side_um': ROI_SIZE_UM, 'n_pixels': n_px,
                    'integrated_intensity': total,
                    'bg_per_px': round(bg_ref, 4),
                    'bg_corrected_intensity': round(total - bg_ref * n_px, 2),
                })
            if fi % 100 == 0:
                print(f'  frame {fi}/{len(pages)}', flush=True)

    res = pd.DataFrame(records)
    res.to_csv(os.path.join(OUT, 'spot_intensity_square_roi.csv'), index=False)
    print(f'Saved {len(res)} spots -> {OUT}/spot_intensity_square_roi.csv')


if __name__ == '__main__':
    main()
