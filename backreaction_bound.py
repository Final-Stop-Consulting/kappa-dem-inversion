"""R3.1 back-reaction bound -- closed-form static-loop (RTV) result.

Static loop, conduction-radiation balance, Lambda = L0 T^-0.5 (coronal regime),
pressure p constant. First integral (apex F=0 -> base):
   |F_base| = (p/2kB) sqrt(2 k_eff L0 (Tmax-Tbase)) ~ k_eff^{1/2} p Tmax^{1/2}
   L/2 = sqrt(k_eff/2L0)(2kB/p) * Integral[T^{5/2}/sqrt(Tmax-T)] = C k_eff^{1/2} Tmax^3 / p
=> at fixed (p,L):  Tmax  prop k_eff^{-1/6}   (RTV (pL)^{1/3} law, conductivity scaling)
   F_base prop k_eff^{1/2} Tmax^{1/2} = k_eff^{1/2}(k_eff^{-1/6})^{1/2} prop k_eff^{5/12}

Bulk conduction Tc = r*Teff makes the operator k0 Tc^{5/2}dTc = k0 r^{7/2} Teff^{5/2}dTeff,
i.e. the self-consistent loop = standard loop with k_eff = k0 * r^{7/2}, r=(kappa-3/2)/kappa.

Two readings:
 (1) FIXED OBSERVED structure (Teff profile + density scale height are measured):
     F_c is exactly F_c^std * r^{7/2}  (back-reaction = 0; correction multiplicative).
 (2) FLOATING apex (heating fixed, profile re-solves) -- the referee's self-consistency:
     Tmax rises by (r^{7/2})^{-1/6} = r^{-7/12}; F_c = F_c^std * r^{7/2} * (Tmax ratio)^{7/2}.
     This requires apex Teff = r^{-7/12} * 1.5 MK, which the EUV diagnostic EXCLUDES.
"""
import os
from pathlib import Path
import numpy as np
REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT_DIR  = Path(os.environ.get('FB_OUT', str(REPO_DIR / 'Results')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
KAPPA=2.5; r=(KAPPA-1.5)/KAPPA            # 0.4
Tobs=1.5e6
naive=r**3.5                              # fixed-structure multiplicative factor
Tmax_ratio=(r**3.5)**(-1/6)              # = r^{-7/12}, floating-apex temperature rise
Fc_float_over_Fc=r**3.5*Tmax_ratio**3.5  # floating-apex F_c / F_c^std
# (check via F prop k_eff^{5/12}: (r^3.5)^{5/12})
Fc_float_check=(r**3.5)**(5/12)

Fc0,Fr,Fw,tot0=2e5,1e5,5e4,3e5
def budget(fac): 
    Fc=Fc0*fac; tot=Fc+Fr+Fw; return Fc,tot,(1-tot/tot0)*100

print("r = T_core/T_eff = %.3f   (kappa=2.5)"%r)
print("naive factor r^(7/2) = %.4f  (F_c reduced x%.1f)"%(naive,1/naive))
print("floating-apex Tmax rise = r^(-7/12) = %.3f  => apex Teff = %.2f MK (obs pins 1.5 MK)"%(Tmax_ratio,Tmax_ratio*Tobs/1e6))
print("floating-apex F_c/F_c^std = %.4f  (check via k^(5/12)=%.4f)"%(Fc_float_over_Fc,Fc_float_check))
print()
print("%-38s %-12s %-12s %-10s"%("reading","F_c","fluid total","reduction"))
for lab,fac in [("(1) fixed observed structure [naive]",naive),
                ("(2) floating-apex self-consistent*",Fc_float_over_Fc)]:
    Fc,tot,red=budget(fac); print("%-38s %.2e   %.2e   %5.0f%%"%(lab,Fc,tot,red))
print("\n* reading (2) requires apex Teff=%.2f MK, excluded by the EUV diagnostic;"%(Tmax_ratio*Tobs/1e6))
print("  even so the reduction floors at %.0f%% -- the correction cannot be erased"%budget(Fc_float_over_Fc)[2])
print("  because F_r+F_w = 1.5e5 (kappa-invariant) already exceeds the corrected F_c.")
print("\nback-reaction compensation (float/naive) = %.1fx in F_c, but observationally excluded."%(Fc_float_over_Fc/naive))
import json
json.dump(dict(r=r,naive_factor=naive,Tmax_rise_float=Tmax_ratio,
   apex_Teff_float_MK=Tmax_ratio*Tobs/1e6,Fc_float_over_std=Fc_float_over_Fc,
   reduction_fixed_pct=budget(naive)[2],reduction_float_pct=budget(Fc_float_over_Fc)[2]),
   open(OUT_DIR / 'backreaction_bound.json','w'),indent=2)
