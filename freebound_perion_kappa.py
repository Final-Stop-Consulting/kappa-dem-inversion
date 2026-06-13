#!/usr/bin/env python3
"""Per-ion / per-level kappa-consistent free-bound for AIA channels (Sec 2.3; R3.2=R4.5).
Milne/detailed balance: free-bound photon E from recombination into level of binding I_edge
is emitted by capture of an electron of energy E_e=E-I_edge; cross-section is distribution-
independent, so kappa/Mxw ratio per level i is R_i=f_kappa(E_e)/f_Mxw(E_e). The draft scaled
by the free-free ratio at the PHOTON energy E, missing the edge shift; this fixes it."""
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.special import gamma

if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser('~')
REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT_DIR = Path(os.environ.get('FB_OUT', str(REPO_DIR / 'Results')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ['XUVTOP'] = str(REPO_DIR / 'Data' / 'CHIANTI_11.0.2_database')

K_B = 1.380649e-16; EV_PER_ERG = 6.242e11; EV2ANG = 12398.41875
T_EFF = 1.5e6; LOG_T_EFF = 6.176; EM_SCALE = 1.0e27; KAPPA = 2.5
AIA_CHANNELS = [94, 131, 171, 193, 211, 335]
KT_EFF_EV = K_B * T_EFF * EV_PER_ERG
A_KAPPA = gamma(KAPPA + 1.0) / (gamma(KAPPA - 0.5) * (KAPPA - 1.5) ** 1.5)


def kappa_over_mxw(E_eV, kappa=KAPPA, kT_eV=KT_EFF_EV):
    E = np.asarray(E_eV, dtype=float)
    x = E / ((kappa - 1.5) * kT_eV)
    jk = A_KAPPA * np.power(1.0 + x, -(kappa + 1.0))
    jm = np.exp(-E / kT_eV)
    r = np.where(jm > 1e-300, jk / jm, 0.0)
    return np.where(E >= 0.0, r, 0.0)


def get_ion_fraction_ratios():
    sys.path.insert(0, str(REPO_DIR))
    from kappa_dem_pipeline import load_ion_fractions, compute_ion_fraction_ratios
    kl, ki, ml, mi = load_ion_fractions()
    return compute_ion_fraction_ratios(kl, ki, ml, mi)


def main():
    import ChiantiPy.core as ch
    import ChiantiPy.tools.io as chio
    import ChiantiPy.tools.util as chutil
    import aiapy.response as aresp
    import astropy.units as u

    log = open(OUT_DIR / 'freebound_perion_kappa_results.txt', 'w')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()

    P('=' * 78)
    P('PER-ION / PER-LEVEL KAPPA-CONSISTENT FREE-BOUND  (R3.2 = R4.5)')
    P('T_eff=%.2f MK (logT=%s) kappa=%s kT_eff=%.2f eV A_kappa=%.4f' % (T_EFF/1e6, LOG_T_EFF, KAPPA, KT_EFF_EV, A_KAPPA))
    P('=' * 78)

    aia_wvl, aia_ea = {}, {}
    for c in AIA_CHANNELS:
        r = aresp.Channel(c * u.angstrom)
        aia_wvl[c] = r.wavelength.to(u.angstrom).value
        aia_ea[c] = r.effective_area.to(u.cm ** 2).value
    wvl = np.arange(10.0, 400.0, 0.25)
    E_phot = EV2ANG / wvl
    ea = {c: np.interp(wvl, aia_wvl[c], aia_ea[c]) for c in AIA_CHANNELS}
    inband = np.zeros_like(wvl, dtype=bool)
    for c in AIA_CHANNELS:
        inband |= ea[c] > 0.01 * ea[c].max()

    ab = chio.abundanceRead(abundancename='sun_coronal_2021_chianti')['abundance']
    ioneq = chio.ioneqRead(ioneqName='chianti')
    iT = ioneq['ioneqTemperature']; iTl = np.log10(iT) if iT[0] > 100 else iT
    tix = int(np.argmin(np.abs(iTl - LOG_T_EFF))); ioneq_all = ioneq['ioneqAll']
    ionfrac = get_ion_fraction_ratios()
    sys.path.insert(0, str(REPO_DIR))
    from kappa_dem_pipeline import load_ion_fractions, interp_logT
    _kl,_ki,_ml,_mi = load_ion_fractions()
    fkap={}; fmxw={}
    for _z in _ki:
        fkap[_z]={}; fmxw[_z]={}
        for _st in _ki[_z]:
            fkap[_z][_st]=interp_logT(_kl,_ki[_z][_st],LOG_T_EFF)
            fmxw[_z][_st]=interp_logT(_ml,_mi[_z][_st],LOG_T_EFF) if (_z in _mi and _st in _mi[_z]) else 0.0
    recs=[]

    elements = {1:'H',2:'He',6:'C',7:'N',8:'O',10:'Ne',12:'Mg',14:'Si',16:'S',20:'Ca',26:'Fe',28:'Ni'}
    fb_mxw = {c:0.0 for c in AIA_CHANNELS}; fb_spec = {c:0.0 for c in AIA_CHANNELS}
    fb_full = {c:0.0 for c in AIA_CHANNELS}; fb_old = {c:0.0 for c in AIA_CHANNELS}
    spread_max = 0.0; ratio_max = 0.0; per131 = []

    for z, el in elements.items():
        if z-1 >= len(ab) or ab[z-1] < 1e-12:
            continue
        for stage in range(1, min(z+1, ioneq_all.shape[1]) + 1):
            try:
                if ioneq_all[z-1, stage-1, tix] < 1e-6:
                    continue
            except IndexError:
                continue
            ion = chutil.zion2name(z, stage)
            try:
                cont = ch.continuum(ion, [T_EFF], abundance='sun_coronal_2021_chianti', em=EM_SCALE)
                cont.freeBound(wvl)
                FB = getattr(cont, 'FreeBound', None)
                if not FB or 'errorMessage' in FB:
                    continue
            except Exception:
                continue
            I = np.asarray(FB['intensity'], float).squeeze()
            if I.shape != wvl.shape:
                continue
            edges = np.asarray(FB['edgeLvlAng'], float); nlv = edges.size
            fbn = np.asarray(FB['fbn'], float)
            fbn = fbn.reshape(nlv, wvl.size) if fbn.size == nlv*wvl.size else fbn.reshape(1, wvl.size)
            s = fbn.sum(axis=0); good = s > 0
            chk = good & inband & (I > 0)
            if chk.sum() > 3:
                rs = s[chk] / I[chk]; spread_max = max(spread_max, float(np.nanmax(rs)/np.nanmin(rs)))
            wgt = np.zeros_like(wvl)
            for i in range(nlv):
                wgt += fbn[i] * kappa_over_mxw(E_phot - EV2ANG/edges[i])
            rspec = np.ones_like(wvl); rspec[good] = wgt[good]/s[good]
            Ispec = I * rspec; Iold = I * kappa_over_mxw(E_phot)
            rr = 1.0
            try:
                rr = float(ionfrac[z][stage]); rr = rr if np.isfinite(rr) else 1.0
            except Exception:
                rr = 1.0
            Ifull = Ispec * rr
            mb = good & inband
            if mb.any():
                ratio_max = max(ratio_max, float(np.nanmax(rspec[mb])))
            ion_chan={}
            for c in AIA_CHANNELS:
                w = ea[c]; dmx = np.trapz(I*w, wvl); dkf = np.trapz(Ifull*w, wvl)
                dsp = np.trapz(Ispec*w, wvl)
                ion_chan[c]=(float(dmx),float(dkf),float(dsp))
                if dmx <= 0 and dkf <= 0:
                    continue
                fb_mxw[c] += dmx; fb_spec[c] += dsp
                fb_full[c] += dkf; fb_old[c] += np.trapz(Iold*w, wvl)
                if c == 131 and dmx > 0:
                    per131.append((ion, dmx, dkf, dsp/dmx, rr))
            recs.append(dict(ion=ion, z=z, stage=stage,
                             f_mxw=float(fmxw.get(z,{}).get(stage,0.0)),
                             f_kappa=float(fkap.get(z,{}).get(stage,0.0)),
                             ratio=float(rr), chan=ion_chan))
        P('  %s (Z=%d): done' % (el, z))

    P('\n--- VALIDATION GATES ---')
    P('(1) max in-band spread of sum_fbn/intensity per ion = %.4f  (~1.00 => shape OK)' % spread_max)
    P('(2) max in-band per-ion spectral kappa/Mxw ratio    = %.3f  (bound A_kappa=%.3f)' % (ratio_max, A_KAPPA))
    P('\n--- CHANNEL FREE-BOUND kappa/Maxwellian RATIOS ---')
    P('%5s %12s %13s %10s %14s' % ('chan','FB_Mxw DN','old(ffproxy)','new spec','new spec*ionf'))
    out = {}
    for c in AIA_CHANNELS:
        o = fb_old[c]/fb_mxw[c] if fb_mxw[c] > 0 else float('nan')
        sp = fb_spec[c]/fb_mxw[c] if fb_mxw[c] > 0 else float('nan')
        fl = fb_full[c]/fb_mxw[c] if fb_mxw[c] > 0 else float('nan')
        out[c] = dict(fb_mxw_dn=fb_mxw[c], fb_kappa_spec_dn=fb_spec[c], fb_kappa_full_dn=fb_full[c],
                      ratio_old_ffproxy=o, ratio_new_spec=sp, ratio_new_full=fl)
        P('%5d %12.4e %13.3f %10.3f %14.3f' % (c, fb_mxw[c], o, sp, fl))
    P('\n--- 131 A per-ion free-bound (top by Mxw DN) ---')
    per131.sort(key=lambda t: t[1], reverse=True)
    P('%8s %12s %12s %10s %8s' % ('ion','FB_Mxw DN','FB_kap DN','spec ratio','ionfrac'))
    for ion, dmx, dkf, sr, rr in per131[:15]:
        P('%8s %12.4e %12.4e %10.3f %8.3f' % (ion, dmx, dkf, sr, rr))
    json.dump({str(k): v for k, v in out.items()}, open(OUT_DIR/'freebound_perion_kappa.json','w'), indent=2)
    np.savez(OUT_DIR/'freebound_perion_kappa.npz', channels=np.array(AIA_CHANNELS),
             fb_mxw=np.array([fb_mxw[c] for c in AIA_CHANNELS]),
             fb_kappa_spec=np.array([fb_spec[c] for c in AIA_CHANNELS]),
             fb_kappa_full=np.array([fb_full[c] for c in AIA_CHANNELS]),
             fb_old_proxy=np.array([fb_old[c] for c in AIA_CHANNELS]),
             A_kappa=A_KAPPA, kT_eff_eV=KT_EFF_EV)
    json.dump(recs, open(OUT_DIR/'freebound_perion_records.json','w'))
    P('\nSaved JSON+NPZ+records to ' + str(OUT_DIR))
    log.close()


if __name__ == '__main__':
    main()
