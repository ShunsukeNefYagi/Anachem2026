"""
s1_parse_xml.py — laser1traj.xml (TrackMate v8.1.5, detection only)
                  → results/linking/spots_raw.csv

注意: <Model spatialunits=" ">, <ImageData pixelwidth="1.0"> なので
      POSITION_X/Y は「ピクセル単位」。µm 換算は 10/222 µm/px を後段で掛ける。
"""
import os
import xml.etree.ElementTree as ET
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
XML  = os.path.join(ROOT, 'data', 'linking', 'laser1traj.xml')
OUT  = os.path.join(ROOT, 'results', 'linking')
os.makedirs(OUT, exist_ok=True)

FIELDS_F = ['POSITION_X', 'POSITION_Y', 'QUALITY', 'RADIUS',
            'MEAN_INTENSITY_CH1', 'MEDIAN_INTENSITY_CH1', 'MAX_INTENSITY_CH1',
            'MIN_INTENSITY_CH1', 'TOTAL_INTENSITY_CH1', 'STD_INTENSITY_CH1',
            'CONTRAST_CH1', 'SNR_CH1']

rows = []
n_tracks = 0
for ev, el in ET.iterparse(XML, events=('end',)):
    if el.tag == 'Spot':
        r = {'ID': int(el.get('ID')), 'FRAME': int(el.get('FRAME'))}
        for k in FIELDS_F:
            v = el.get(k)
            r[k] = float(v) if v is not None else float('nan')
        rows.append(r)
        el.clear()
    elif el.tag == 'Track':
        n_tracks += 1
        el.clear()

df = pd.DataFrame(rows)
df = df.rename(columns={'POSITION_X': 'X_px', 'POSITION_Y': 'Y_px'})
df = df.sort_values(['FRAME', 'ID']).reset_index(drop=True)
df.to_csv(os.path.join(OUT, 'spots_raw.csv'), index=False)

print(f'spots            : {len(df)}')
print(f'<Track> elements : {n_tracks}')
print(f'frames           : {df.FRAME.min()}–{df.FRAME.max()} '
      f'({df.FRAME.nunique()} frames with spots)')
c = df.groupby("FRAME").size()
print(f'spots per frame  : mean {c.mean():.2f}, min {c.min()}, max {c.max()}')
print(f'X range          : {df.X_px.min():.1f} – {df.X_px.max():.1f} px')
print(f'Y range          : {df.Y_px.min():.1f} – {df.Y_px.max():.1f} px')
print(f'Saved: {os.path.join(OUT, "spots_raw.csv")}')
