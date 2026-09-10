"""Reproduce the T4 DC-MSS segmentation from the shipped, drift-corrected coordinates.

Self-verifying: asserts that the boundaries and classes match the reference output in
data/dcmss_T4/expected_T4_segments.csv and that |dS_MSS| < 1e-10, then prints PASS.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]                                  # repository root
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'analysis'/'dcmss'))       # mss_core.py
import mss_core as core
import dc_mss

def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.floating):return float(v) if np.isfinite(v) else None
 raise TypeError(type(v).__name__)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results'/'dcmss_T4');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 c=json.loads((HERE/'config.json').read_text());data=pd.read_csv(ROOT/c['input']).sort_values('FRAME')
 assert data.TRACK_ID.eq(c['track_id']).all() and len(data)==720
 assert np.array_equal(data.FRAME.to_numpy(),np.arange(720)),'Missing/duplicate frames: this package expects the documented complete T4 input.'
 pos=data[['X_px','Y_px']].to_numpy(float);assert np.isfinite(pos).all()
 thr=core.MSSThresholds(str(ROOT/c['thresholds']))
 assert thr.lag_frac==c['lag_frac'] and thr.lag_cap==c['lag_cap']
 cfg=dc_mss.Cfg(c['window'],c['sigma'],c['pctl'],c['min_seg'],c['lag_frac'],c['floor'],guard=c['guard'])
 assert cfg.ks_alpha==c['ks_alpha']
 result=dc_mss.dc_mss3(pos,thr,cfg)
 mpd=core.max_pairwise_distance_series(pos,c['window']);sm=gaussian_filter1d(mpd,c['sigma'],mode='nearest')
 den=(sm[1:]+sm[:-1])/2;ds=np.divide(abs(np.diff(sm)),den,out=np.zeros_like(den),where=den>0);cut=float(np.percentile(ds,c['pctl']))
 peak=np.zeros(len(ds),bool);peak[1:-1]=(ds[1:-1]>=ds[:-2])&(ds[1:-1]>=ds[2:]);cand=np.flatnonzero(peak&(ds>=cut))+1
 raw=sorted(set([0,*cand.tolist(),len(pos)]));extended=dc_mss.extend_short(raw,c['min_seg'],c['floor']);ks,nmerge=core._ks_merge(pos,extended,c['ks_alpha'])
 assert np.array_equal(ks,result['bounds'])
 pd.DataFrame(dict(frame=np.arange(720),x_px=pos[:,0],y_px=pos[:,1],MPD_px=mpd,SMPD_px=sm,delta_SMPD=np.r_[np.nan,ds],candidate=np.isin(np.arange(720),cand))).to_csv(args.out/'T4_diagnostics.csv',index=False)
 rows=[];mom=[]
 for i,s in enumerate(result['segs'],1):
  a,b=int(s['start']),int(s['end']);p=pos[a:b];v=np.diff(p,axis=0);n=np.linalg.norm(v,axis=1);u=np.divide(v,n[:,None],out=np.zeros_like(v),where=n[:,None]>0)
  md=core.mss_analysis(p,lag_frac=thr.lag_frac,lag_cap=thr.lag_cap);ts=thr.thresholds_at(len(p));cls=core.CLASS_NAMES[s['cls']]
  rows.append(dict(segment=i,start_frame=a,end_frame_exclusive=b,last_observed_frame=b-1,n_positions=b-a,elapsed_between_endpoints_min=(b-a-1)*c['dt_min'],class_label=cls,S_MSS=s['mss_slope'],max_lag_min=int(md['max_lag'])*c['dt_min'],threshold_imm_conf=ts[0],threshold_conf_free=ts[1],threshold_free_directed=ts[2],margin_free_directed=s['mss_slope']-ts[2],p_mis_model=thr.p_misclass(s['mss_slope'],len(p),s['cls']),alpha2_native=float(md['alphas'][0,2]),C1=float(np.mean(np.sum(v[:-1]*v[1:],axis=1))/np.mean(np.sum(v*v,axis=1))),Cu1=float(np.mean(np.sum(u[:-1]*u[1:],axis=1))),net_displacement_um=float(np.linalg.norm(p[-1]-p[0])*c['pixel_um'])))
  for q,g in enumerate(md['alphas'][0]):mom.append(dict(segment=i,q=q,gamma=g))
 out=pd.DataFrame(rows);out.to_csv(args.out/'T4_final_segments.csv',index=False,float_format='%.17g');pd.DataFrame(mom).to_csv(args.out/'T4_moment_spectrum.csv',index=False,float_format='%.17g')
 ini=pd.DataFrame(result['segs_initial']);ini['class_label']=ini.cls.map(core.CLASS_NAMES);ini.to_csv(args.out/'T4_initial_classification.csv',index=False)
 pd.DataFrame(np.column_stack([thr.durations,thr.thr]),columns=['n_positions','imm_conf','conf_free','free_directed']).to_csv(args.out/'threshold_grid.csv',index=False,float_format='%.17g')
 trace=dict(config=c,DSMPD_95th_percentile=cut,candidate_boundaries=raw,after_extension=extended,after_KS=ks,n_KS_merges=nmerge,final_boundaries=[int(s['start']) for s in result['segs']]+[720],threshold_metadata=thr.meta,n_threshold_durations=len(thr.durations),n_threshold_samples_per_class=thr.nsim)
 (args.out/'stages.json').write_text(json.dumps(trace,default=serial,indent=2),encoding='utf-8')
 expected=pd.read_csv(ROOT/'data'/'dcmss_T4'/'expected_T4_segments.csv')
 assert out.start_frame.tolist()==expected.start_min.tolist()
 assert out.end_frame_exclusive.tolist()==expected.end_min.tolist()
 assert out.class_label.tolist()==expected.cls.tolist()
 diff=float(np.max(abs(out.S_MSS-expected.S_MSS)));assert diff<1e-10,diff
 rawdata=pd.read_csv(ROOT/'data'/'dcmss_T4'/'T4_tracks_raw.csv');delta=pd.read_csv(ROOT/'data'/'dcmss_T4'/'T4_coordinate_correction.csv')
 reconstructed=rawdata[['X_px','Y_px']].to_numpy()-delta[['dx_raw_minus_vib_px','dy_raw_minus_vib_px']].to_numpy();assert np.max(abs(reconstructed-pos))<1e-10
 verification=dict(n_positions=len(pos),no_interpolation_needed=True,exact_boundary_and_class_match=True,max_abs_S_difference=diff,loaded_core=str(Path(core.__file__).resolve()),loaded_algorithm=str(Path(dc_mss.__file__).resolve()),input_sha256=hashlib.sha256((ROOT/c['input']).read_bytes()).hexdigest())
 (args.out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
 colors={'free':'#217fa8','directed':'#cb4f8d','confined':'#7252a1','immobile':'#a67c42'}
 fig,ax=plt.subplots(4,1,figsize=(10,9),sharex=True,gridspec_kw=dict(height_ratios=[2,2,1,1]))
 ax[0].plot(np.arange(720),mpd,color='#bcc9cf',label='MPD');ax[0].plot(np.arange(720),sm,color='#216c8c',label='smoothed MPD');ax[0].set_ylabel('Distance (px)');ax[0].legend(fontsize=8)
 ax[1].plot(np.arange(1,720),ds,lw=.8);ax[1].axhline(cut,color='crimson',ls='--',label='95th percentile');ax[1].scatter(cand,ds[cand-1],s=13,color='crimson');ax[1].set_ylabel('Normalized change');ax[1].legend(fontsize=8)
 for a,b in zip(result['bounds'][:-1],result['bounds'][1:]):ax[2].axvspan(a,b,facecolor='#c3d2d9',edgecolor='white')
 ax[2].set_ylabel('After step 1');ax[2].set_yticks([])
 for r in rows:
  a,b=r['start_frame'],r['end_frame_exclusive'];ax[3].axvspan(a,b,facecolor=colors[r['class_label']],edgecolor='white');ax[3].text((a+b)/2,.5,str(r['segment']),ha='center',va='center',color='white',fontsize=9)
 ax[3].set_ylabel('Final class');ax[3].set_yticks([]);ax[3].set_xlabel('Frame / min (right boundary is exclusive)');fig.tight_layout();fig.savefig(args.out/'T4_segmentation_steps.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4.5))
 for r in rows:
  a,b=r['start_frame'],r['end_frame_exclusive'];p=pos[a:b]*c['pixel_um'];axs[0].plot(p[:,0],p[:,1],color=colors[r['class_label']],lw=1);axs[0].text(p[0,0],p[0,1],str(r['segment']),fontsize=8)
  g=pd.DataFrame(mom).query('segment==@r["segment"]') if False else [t['gamma'] for t in mom if t['segment']==r['segment']]
  axs[1].plot(range(7),g,'o-',ms=3,label=f'{r["segment"]}: {a}-{b}, S={r["S_MSS"]:.3f}')
 axs[0].set_aspect('equal');axs[0].set_xlabel('x (um)');axs[0].set_ylabel('y (um)');axs[1].set_xlabel('Moment order q');axs[1].set_ylabel('gamma(q)');axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(args.out/'T4_trajectory_and_MSS.png',dpi=180);plt.close(fig)
 print(out[['segment','start_frame','end_frame_exclusive','class_label','S_MSS','max_lag_min']].to_string(index=False));print('PASS: reference boundaries/classes reproduced; max slope error',diff)
if __name__=='__main__':main()
