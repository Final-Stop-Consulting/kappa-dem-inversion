import sys, os, json
from pathlib import Path
os.environ.setdefault('HOME', os.path.expanduser('~'))
REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT_DIR  = Path(os.environ.get('FB_OUT', str(REPO_DIR / 'Results')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_DIR))
import numpy as np
import kappa_dem_pipeline as kp
from demregpy import dn2dem

RES = str(REPO_DIR / 'Results') + os.sep
OUT = str(OUT_DIR) + os.sep

tlT, tm, names = kp.load_demregpy_response()
temps = 10**np.arange(5.7, 7.2 + 0.05, 0.05)
mlogt = np.array([np.mean([np.log10(temps[i]), np.log10(temps[i+1])]) for i in range(len(temps)-1)])
DOF = 5  # 6 channels - 1

def fwhm(x,y):
    h=y.max()/2; ab=y>=h
    if not ab.any(): return float('nan')
    f=int(np.argmax(ab)); l=len(ab)-1-int(np.argmax(ab[::-1]))
    le=x[f] if f==0 else x[f-1]+(h-y[f-1])/(y[f]-y[f-1])*(x[f]-x[f-1])
    ri=x[l] if l+1>=len(x) else x[l]+(h-y[l])/(y[l+1]-y[l])*(x[l+1]-x[l])
    return ri-le

families = [
    ('single-T kappa=2.5', 'kappa_dem_inversion_results.npz', 'dn_kappa'),
    ('multi-T kappa=2.5',  'multi_thermal_kappa_results.npz', 'dn_kappa_for_inversion'),
    ('Brooks-Mxw forward', 'brooks_mxw_forward_results.npz', 'dn_for_inversion'),
    ('isothermal Mxw',     'isothermal_mxw_baseline_results.npz', 'dn_mxw_iso'),
]
gates = {'single-T kappa=2.5':0.222, 'multi-T kappa=2.5':0.305,
         'Brooks-Mxw forward':0.319, 'isothermal Mxw':0.174}
reg_tweaks = [0.5, 1.0, 2.0]

dn_cache = {}
for label, fn, key in families:
    d = np.load(RES+fn, allow_pickle=True)
    dn_cache[label] = np.asarray(d[key], dtype=float)

results = {}   # label -> rt -> dict
gate_lines = []
for label, fn, key in families:
    dn = dn_cache[label]
    edn = kp.compute_aia_noise(dn)
    results[label] = {}
    for rt in reg_tweaks:
        dem, edem, elogt, chisq, dn_reg = dn2dem(dn, edn, tm, tlT, temps, reg_tweak=rt)
        dem = np.asarray(dem).ravel()
        chi2 = float(np.asarray(chisq).ravel()[0])
        pk = float(mlogt[int(np.argmax(dem))])
        fw = float(fwhm(mlogt, dem))
        results[label][rt] = dict(chi2=chi2, chi2_dof=chi2/DOF, peak_logT=pk, fwhm=fw)
    # gate
    fw1 = results[label][1.0]['fwhm']
    exp = gates[label]
    ok = bool(abs(fw1 - exp) <= 0.005)
    gate_lines.append((label, fw1, exp, fw1-exp, ok))

# stability per family
stability = {}
for label, _, _ in families:
    fws = [results[label][rt]['fwhm'] for rt in reg_tweaks]
    stability[label] = max(fws) - min(fws)

# ---- write txt ----
with open(OUT+'lambda_sweep_results.txt','w') as f:
    f.write("DEM-inversion regularization (reg_tweak) sensitivity sweep\n")
    f.write("demregpy 0.6.2 | dn2dem reg_tweak = target reduced chi^2 (Morozov discrepancy)\n")
    f.write("temps: 10**arange(5.7,7.25,0.05) -> 31 edges, 30 DEM bins; DOF=5 (6 chan -1)\n")
    f.write("noise: kp.compute_aia_noise (shot+read, dn2ph corrected, exp=2.9s)\n\n")
    f.write("GATE CHECK at reg_tweak=1.0 (must match published FWHM within +/-0.005):\n")
    for label, fw1, exp, diff, ok in gate_lines:
        f.write(f"  {label:22s} FWHM={fw1:.4f}  published={exp:.3f}  diff={diff:+.4f}  {'PASS' if ok else 'FAIL'}\n")
    f.write("\n")
    f.write(f"{'family':22s} {'reg_tweak':>9s} {'chi2':>10s} {'chi2/dof':>10s} {'peak_logT':>10s} {'FWHM':>8s}\n")
    f.write("-"*74+"\n")
    for label, _, _ in families:
        for rt in reg_tweaks:
            r = results[label][rt]
            f.write(f"{label:22s} {rt:9.1f} {r['chi2']:10.4f} {r['chi2_dof']:10.4f} {r['peak_logT']:10.4f} {r['fwhm']:8.4f}\n")
        f.write("-"*74+"\n")
    f.write("\nFWHM stability across 0.5x-2x reg_tweak (max|dFWHM| per family):\n")
    for label, _, _ in families:
        f.write(f"  {label:22s} max|dFWHM| = {stability[label]:.4f}\n")

# ---- write json ----
out = dict(
    demregpy_version="0.6.2",
    dn2dem_signature="dn2dem(dn_in, edn_in, tresp, tresp_logt, temps, reg_tweak=1.0, max_iter=10, gloci=0, rgt_fact=1.5, dem_norm0=None, nmu=40, warn=False, emd_int=False, emd_ret=False, l_emd=False, non_pos=False)",
    reg_tweak_meaning="The target normalised (reduced) chi-squared used in the Morozov discrepancy principle to select the regularization parameter lambda (discr = sum(arg) - sum(err**2)*reg_tweak).",
    temps_grid="10**arange(5.7, 7.25, 0.05)",
    dof=DOF,
    gate_check={lab:dict(fwhm=round(fw1,5),published=exp,diff=round(diff,5),pass_=ok) for lab,fw1,exp,diff,ok in gate_lines},
    sweep={lab:{str(rt):{k:round(v,5) for k,v in results[lab][rt].items()} for rt in reg_tweaks} for lab,_,_ in families},
    fwhm_stability_max_delta={lab:round(stability[lab],5) for lab,_,_ in families},
)
json.dump(out, open(OUT+'lambda_sweep_results.json','w'), indent=2)

# console summary
print("GATES:")
for label, fw1, exp, diff, ok in gate_lines:
    print(f"  {label:22s} FWHM={fw1:.4f} pub={exp:.3f} diff={diff:+.4f} {'PASS' if ok else 'FAIL'}")
print("\nSWEEP:")
print(f"{'family':22s} {'rt':>4s} {'chi2':>9s} {'chi2/dof':>9s} {'peakLogT':>9s} {'FWHM':>7s}")
for label, _, _ in families:
    for rt in reg_tweaks:
        r = results[label][rt]
        print(f"{label:22s} {rt:4.1f} {r['chi2']:9.4f} {r['chi2_dof']:9.4f} {r['peak_logT']:9.4f} {r['fwhm']:7.4f}")
print("\nSTABILITY max|dFWHM|:")
for label, _, _ in families:
    print(f"  {label:22s} {stability[label]:.4f}")
