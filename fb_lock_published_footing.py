"""Lock the per-ion free-bound inversion impact on the PUBLISHED footing:
uses kappa_dem_pipeline.load_demregpy_response + compute_aia_noise + run_dem_inversion
(the exact functions behind the manuscript's chi2/dof = 1.00 continuum result).
Gates: lines-only -> chi2 ~ 7.588 ; draft proxy (fb=ff) -> chi2 ~ 5.020."""
import os, sys, json
import numpy as np
from pathlib import Path
if 'HOME' not in os.environ: os.environ['HOME']=os.path.expanduser('~')
REPO=Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT=Path(os.environ.get('FB_OUT', str(REPO / 'Results')))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(REPO))
import kappa_dem_pipeline as kp

CH=[94,131,171,193,211,335]; I131=1
FF_MXW=np.array([0.2478,3.350,3.203,5.653,3.864,0.1379])
FB_MXW=np.array([0.1240,1.432,1.234,2.100,1.388,0.02628])
FF_RATIO=np.array([0.785,1.005,1.235,1.329,1.405,1.802])
j=json.load(open(OUT/'freebound_perion_kappa.json'))
FB_SPEC=np.array([j[str(c)]['ratio_new_spec'] for c in CH])
recs=json.load(open(OUT/'freebound_perion_records.json'))
mxw_tot=np.array([sum(r['chan'][str(c)][0] for r in recs) for c in CH])
FB_FLOOR=np.array([sum(r['chan'][str(c)][1] for r in recs if r['f_mxw']>=1e-3) for c in CH])/mxw_tot

def fwhm(x,y):
    h=y.max()/2; ab=y>=h
    if not ab.any(): return float('nan')
    f=int(np.argmax(ab)); l=len(ab)-1-int(np.argmax(ab[::-1]))
    le=x[f] if f==0 else x[f-1]+(h-y[f-1])/(y[f]-y[f-1])*(x[f]-x[f-1])
    ri=x[l] if l+1>=len(x) else x[l]+(h-y[l])/(y[l+1]-y[l])*(x[l+1]-x[l])
    return ri-le

tlT,tm,names=kp.load_demregpy_response()
dn_lines=np.asarray(np.load(REPO/'Results'/'kappa_dem_inversion_results.npz',allow_pickle=True)['dn_kappa']).astype(float)
def corr(fb): return dn_lines + FF_MXW*FF_RATIO + FB_MXW*fb
cases=[('NOMINAL lines-only',dn_lines),
       ('DRAFT proxy (fb=ff)',corr(FF_RATIO)),
       ('SPECTRAL fb',corr(FB_SPEC)),
       ('FLOORED-FULL fb',corr(FB_FLOOR))]
log=open(OUT/'fb_lock_published_footing.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n'); log.flush()
P('FB ratios: spec',np.round(FB_SPEC,3).tolist(),'floor',np.round(FB_FLOOR,3).tolist())
P('\n%22s %8s %9s %9s %7s %8s %10s'%('case','chi2','chi2/dof','peaklogT','FWHM','131DN','sec>6.5'))
res={}
for name,dn in cases:
    edn=kp.compute_aia_noise(dn)
    dem,edem,elogt,chisq,dn_reg,mlogt,temps=kp.run_dem_inversion(dn,edn,tm,tlT)
    dem=np.asarray(dem).squeeze(); chisq=float(np.asarray(chisq).squeeze())
    c=np.asarray(mlogt); pk=c[int(np.argmax(dem))]; fw=fwhm(c,dem)
    tmask=c>6.5; sec='none'
    if tmask.any():
        td=dem.copy(); td[~tmask]=0
        if td.max()>dem.max()*0.05: sec='%.3f@rel%.3f'%(c[int(np.argmax(td))],td.max()/dem.max())
    P('%22s %8.3f %9.3f %9.3f %7.3f %8.3f %10s'%(name,chisq,chisq/5,pk,fw,dn[I131],sec))
    res[name]=dict(chi2=chisq,chi2dof=chisq/5,peak=float(pk),fwhm=float(fw),dn131=float(dn[I131]),secondary=sec)
json.dump(res,open(OUT/'fb_lock_published_footing.json','w'),indent=2)
P('\nGATES: lines-only expect chi2~7.588 ; draft proxy expect chi2~5.020 (published 1.00=5.020/5)')
log.close()
