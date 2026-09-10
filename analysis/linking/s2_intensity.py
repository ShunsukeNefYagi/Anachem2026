"""
s2_intensity.py — 輝点ごとの積分輝度（BG 補正済み）

ドリフト／振動除去より前に実行する。理由:
  輝度は「その時刻の生画像のその画素」から取る量であり、座標の後処理（ドリフト補正）を
  適用した座標で切り出すと、実際に粒子が写っていない場所を測ることになる。
  ドリフト補正は解析座標系の話、輝度は検出座標系の話なので分離する。

手法:
  - 各スポット中心を中心とする一辺 0.4 µm の正方形 ROI（half = 4.44 px → 通常 9×9 = 81 px）
  - ピクセル中心の規約: POSITION_X = (col + 0.5) * pixelwidth → col = POSITION_X - 0.5
  - BG は frame 0 の左上 100×100 px の平均 [counts/px]、I_BG = I_raw - b * N_px

追加: 褪色（bleaching）の評価用に、フレームごとの背景も同じ左上 100×100 px で記録する。

出力: out/spot_intensity_square_roi.csv
"""
import os
import numpy as np
import pandas as pd
import tifffile

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
TIF  = os.path.join(ROOT, 'data', 'linking', 'laser1traj.tif')
OUT  = os.path.join(ROOT, 'results', 'linking')
os.makedirs(OUT, exist_ok=True)

ROI_SIZE_UM = 0.4
PIXEL_SIZE  = 10.0 / 222.0
BG_REGION   = 100

# The 720-frame image stack (2.9 GB) is too large to distribute with the code.
# Its product, spot_intensity_square_roi.csv, is shipped in data/linking/ so the
# rest of the pipeline (s3*, s4) runs without it. Request the stack from the
# corresponding author to re-run this step from the raw images.
if not os.path.exists(TIF):
    raise SystemExit(
        f'Image stack not found: {TIF}\n'
        'This step needs the raw 720-frame TIFF, which is not distributed with '
        'the repository (2.9 GB). Use the precomputed '
        'data/linking/spot_intensity_square_roi.csv instead, or obtain the stack '
        'from the corresponding author.')

df = pd.read_csv(os.path.join(OUT, 'spots_raw.csv'))
print(f'Spots: {len(df)}  frames {int(df.FRAME.min())}-{int(df.FRAME.max())}', flush=True)

df['cx_px'] = df['X_px'] - 0.5
df['cy_px'] = df['Y_px'] - 0.5
df['x_um']  = df['X_px'] * PIXEL_SIZE
df['y_um']  = df['Y_px'] * PIXEL_SIZE

half_px = (ROI_SIZE_UM / 2.0) / PIXEL_SIZE
print(f'ROI: {ROI_SIZE_UM} um square (half = {half_px:.3f} px)', flush=True)

by_frame = {f: g for f, g in df.groupby('FRAME')}

records, bg_frames = [], []
with tifffile.TiffFile(TIF) as tif:
    pages = tif.pages
    n_frames = len(pages)
    h, w = pages[0].shape
    print(f'TIF: {n_frames} frames, {h}x{w}, {pages[0].dtype}', flush=True)

    bg_ref = float(pages[0].asarray()[:BG_REGION, :BG_REGION].mean())
    print(f'BG(frame 0, top-left {BG_REGION}x{BG_REGION}): {bg_ref:.4f} counts/px', flush=True)

    for fi in range(n_frames):
        img = pages[fi].asarray()
        bg_frames.append(dict(frame=fi,
                              bg_local=float(img[:BG_REGION, :BG_REGION].mean()),
                              img_median=float(np.median(img))))
        g = by_frame.get(fi)
        if g is None:
            continue
        for _, spot in g.iterrows():
            cx, cy = float(spot['cx_px']), float(spot['cy_px'])
            c0   = max(0, int(np.floor(cx - half_px)))
            c1   = min(w, int(np.ceil(cx + half_px)) + 1)
            row0 = max(0, int(np.floor(cy - half_px)))
            row1 = min(h, int(np.ceil(cy + half_px)) + 1)

            yy, xx = np.ogrid[row0:row1, c0:c1]
            mask = (np.abs(xx - cx) <= half_px) & (np.abs(yy - cy) <= half_px)

            pixels = img[row0:row1, c0:c1][mask].astype(np.float64)
            n_px  = int(pixels.size)
            total = float(pixels.sum())
            records.append({
                'spot_id':                int(spot['ID']),
                'frame':                  fi,
                'x_um':                   round(float(spot['x_um']), 4),
                'y_um':                   round(float(spot['y_um']), 4),
                'cx_px':                  round(cx, 3),
                'cy_px':                  round(cy, 3),
                'roi_side_um':            ROI_SIZE_UM,
                'n_pixels':               n_px,
                'integrated_intensity':   total,
                'bg_per_px':              round(bg_ref, 4),
                'bg_corrected_intensity': round(total - bg_ref * n_px, 2),
            })
        if fi % 50 == 0:
            print(f'  frame {fi}/{n_frames}', flush=True)

res = pd.DataFrame(records)
res.to_csv(os.path.join(OUT, 'spot_intensity_square_roi.csv'), index=False)
pd.DataFrame(bg_frames).to_csv(os.path.join(OUT, 'bg_per_frame.csv'), index=False)

print(f'Saved: out/spot_intensity_square_roi.csv ({len(res)} rows)')
print(f'n_pixels range: {res["n_pixels"].min()} - {res["n_pixels"].max()}')
print(f'integrated  : mean={res["integrated_intensity"].mean():.1f}, '
      f'median={res["integrated_intensity"].median():.1f}')
print(f'bg-corrected: mean={res["bg_corrected_intensity"].mean():.1f}, '
      f'median={res["bg_corrected_intensity"].median():.1f}')
bgf = pd.DataFrame(bg_frames)
print(f'bg_local    : frame0={bgf.bg_local.iloc[0]:.3f}, '
      f'frame719={bgf.bg_local.iloc[-1]:.3f}, '
      f'min={bgf.bg_local.min():.3f}, max={bgf.bg_local.max():.3f}')
