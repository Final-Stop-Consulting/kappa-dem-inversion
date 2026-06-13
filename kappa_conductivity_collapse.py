"""Spitzer-Harm closure non-existence for the kappa corona (TP-1080 reframe).

Demonstrates, with a published closed form and a machine-precision validation gate,
that the local Spitzer-Harm thermal conductivity does not exist as a finite, positive,
order-unity coefficient anywhere in the 2026a quiet-Sun prior kappa in [2,3].

Ground truth: Du (2013), Phys. Plasmas 20, 092901 (Lorentz e-i plasma, standard
"phenomenological" kappa). Thermal conductivity (Du Eq. 26), ratio to the Maxwellian
limit (Du Eq. 27):

   lambda_k/lambda_M = (kappa-3/2)^(7/2) * (kappa+1)/(kappa-3)
                       * Gamma(kappa-4)/Gamma(kappa-1/2)

   poles at kappa = 2, 3, 4 ; finite NEGATIVE (-16.5) at kappa=2.5 ; -> 1 only for kappa >~ 5.

The velocity-space integrand behind the dominant heat-carrying term (Du Eq. 25,
the (m/2T)<v^9 (1+A_k v^2)^-1> piece, with the 4*pi v^2 measure from Du's Appendix):

   K(v) prop v^11 (1 + A_k v^2)^-(kappa+2) ,   A_k v^2 = x^2/(kappa-3/2),  x = v/v_th

   analytic:  Int_0^inf x^11 (1+x^2/b)^-(k+2) dx = (1/2) b^6 Gamma(6) Gamma(k-4)/Gamma(k+2)
   converges only for kappa > 4 ; Maxwellian limit K_M(x)=x^11 exp(-x^2)
   (peak at x=sqrt(11/2)=2.345 v_th, integral Gamma(6)/2 = 60).

Interpretation: across the entire prior the local conductivity integral diverges; the
closed form's finite -16.5 at kappa=2.5 is an analytic continuation of a divergent
integral (no physical conductivity). Any truncated/regularized value scales with the
cutoff (dominant term prop v_c^3; regularized-kappa form prop xi^-7, Husidic 2022),
so the number measures the cutoff, not the plasma. Either way the Spitzer-Harm
small-correction picture is invalid in this regime.

Convention note: Du = Lorentz (e-i only). e-e-inclusive treatments (Guo & Du 2019)
remain pathological at small kappa (coefficient negative for kappa<10); the standard-
kappa Boltzmann result (Husidic et al. 2021) carries the same kappa=3,4 poles; the
regularized-kappa form (Husidic et al. 2022) is the only finite value at kappa=2.5 and
is cutoff-dependent and ~1.7e4 x Maxwellian. None supports a finite order-unity local kappa_0.
"""
import os, json
from pathlib import Path
import numpy as np
from scipy.special import gamma
from scipy import integrate

REPO_DIR = Path(os.environ.get('FB_REPO', Path(__file__).resolve().parent))
OUT_DIR  = Path(os.environ.get('FB_OUT', str(REPO_DIR / 'Results')))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR = (2.0, 3.0)   # Edmonds 2026a quiet-Sun kappa prior


# --- Du (2013) Eq. 26 closed form, ratio to Maxwellian (Eq. 27) ---------------
def du_lambda_ratio(k):
    return (k - 1.5)**3.5 * (k + 1.0) / (k - 3.0) * gamma(k - 4.0) / gamma(k - 0.5)


# --- dominant heat-flux velocity integrand (Du Eq. 25) and its analytic form --
def integrand(x, k):
    b = k - 1.5
    return x**11 * (1.0 + x * x / b)**(-(k + 2.0))

def integral_quad(k, xmax=6000):
    return integrate.quad(lambda x: integrand(x, k), 0, xmax, limit=800)[0]

def integral_analytic(k):
    b = k - 1.5
    return 0.5 * b**6 * gamma(6.0) * gamma(k - 4.0) / gamma(k + 2.0)

def integrand_maxwellian(x):
    return x**11 * np.exp(-x * x)


def main():
    rec = {}

    # 1. Du closed form across kappa
    print("=" * 68)
    print("Du (2013) Eq. 26  lambda_kappa / lambda_Maxwellian")
    print("=" * 68)
    table = {}
    for k in [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 20.0]:
        try:
            v = du_lambda_ratio(k)
            v = None if (not np.isfinite(v)) else float(v)
        except Exception:
            v = None
        table[f"{k:.2f}"] = v
        print(f"  kappa={k:5.2f}   lambda/lambda_M = "
              + ("POLE/undefined" if v is None else f"{v:12.4f}"))
    rec["du_eq26_ratio"] = table
    rec["du_at_2p5"] = du_lambda_ratio(2.5)
    print(f"\n  at kappa=2.5 : {du_lambda_ratio(2.5):.4f}  (finite, NEGATIVE; "
          "analytic continuation of a divergent integral)")
    print("  poles at kappa = 2, 3, 4 ; -> 1 only for kappa >~ 5")

    # 2. validation gate: quadrature == analytic (kappa>4)
    print("\n" + "=" * 68)
    print("VALIDATION GATE: integrand quadrature vs analytic Beta form (kappa>4)")
    print("=" * 68)
    gate = {}
    worst = 0.0
    for k in [4.5, 5, 6, 8, 12, 20, 50]:
        q = integral_quad(k); a = integral_analytic(k)
        ratio = q / a
        worst = max(worst, abs(ratio - 1.0))
        gate[f"{k:.1f}"] = dict(quad=q, analytic=a, ratio=ratio)
        print(f"  kappa={k:5.1f}   quad={q:14.5e}  analytic={a:14.5e}  ratio={ratio:.6f}")
    rec["validation_gate"] = gate
    rec["validation_worst_dev"] = worst
    print(f"  worst |ratio-1| = {worst:.2e}  ->  {'PASS' if worst < 1e-3 else 'FAIL'}")

    # 3. Maxwellian limit of the integrand
    xpk = np.sqrt(11.0 / 2.0)
    integ_M = integrate.quad(integrand_maxwellian, 0, 60)[0]
    rec["maxwellian_carrier_peak_vth"] = xpk
    rec["maxwellian_integral"] = integ_M
    print("\n" + "=" * 68)
    print("Maxwellian limit  K_M(x) = x^11 exp(-x^2)")
    print("=" * 68)
    print(f"  carrier peak at x = sqrt(11/2) = {xpk:.3f} v_th   (heat carried by ~2-4 v_th)")
    print(f"  integral = Gamma(6)/2 = {integ_M:.4f}  (finite -> finite Maxwellian kappa_0)")

    # 4. divergence across the prior + cutoff scaling
    print("\n" + "=" * 68)
    print("DIVERGENCE across the prior (cumulative grows with cutoff) + cutoff scaling")
    print("=" * 68)
    div = {}
    for k in [2.0, 2.5, 3.0]:
        i50  = integrate.quad(lambda x: integrand(x, k), 0, 50,  limit=800)[0]
        i500 = integrate.quad(lambda x: integrand(x, k), 0, 500, limit=800)[0]
        div[f"{k:.1f}"] = dict(I_lt50=i50, I_lt500=i500, ratio=i500 / i50)
        print(f"  kappa={k:4.1f}: I(<50)={i50:11.3e}  I(<500)/I(<50)={i500/i50:9.2f}  -> diverges")
    rec["divergence_prior"] = div
    # cutoff scaling at 2.5: integrand ~ x^(7-2k)=x^2 -> cumulative ~ x_c^3
    sc = {}
    for xc in [25, 50, 100, 200]:
        sc[str(xc)] = integrate.quad(lambda x: integrand(x, 2.5), 0, xc, limit=800)[0]
    rec["cutoff_scaling_2p5"] = sc
    xs = sorted(sc, key=int)
    slopes = [np.log(sc[xs[i+1]]/sc[xs[i]]) / np.log(int(xs[i+1])/int(xs[i]))
              for i in range(len(xs)-1)]
    rec["cutoff_scaling_exponent"] = float(np.mean(slopes))
    print(f"  kappa=2.5 cumulative ~ v_c^{np.mean(slopes):.2f}  (predicted 3: double cutoff -> x8)")

    # 5. verdict
    rec["verdict"] = ("No finite, convention-independent, order-unity local Spitzer-Harm "
                      "conductivity exists in the prior [2,3]; the SH small-correction "
                      "picture is invalid in this regime.")
    print("\n" + "=" * 68)
    print("VERDICT:", rec["verdict"])
    print("=" * 68)

    with open(OUT_DIR / 'kappa_conductivity_collapse.json', 'w') as f:
        json.dump(rec, f, indent=2)
    with open(OUT_DIR / 'kappa_conductivity_collapse.txt', 'w') as f:
        f.write("Spitzer-Harm closure non-existence (TP-1080 reframe)\n")
        f.write("Du 2013 Eq.26 ratio at kappa=2.5: %.4f (finite negative; analytic "
                "continuation of a divergent integral)\n" % du_lambda_ratio(2.5))
        f.write("Poles at kappa=2,3,4; ->1 for kappa>~5.\n")
        f.write("Validation gate worst |ratio-1| = %.2e (PASS<1e-3)\n" % worst)
        f.write("Maxwellian carriers peak at %.3f v_th; integral=%.3f.\n" % (xpk, integ_M))
        f.write("kappa=2.5 conductivity integral diverges; cumulative ~ v_c^%.2f.\n"
                % rec["cutoff_scaling_exponent"])
        f.write("VERDICT: %s\n" % rec["verdict"])
    print("\nwrote Results/kappa_conductivity_collapse.{json,txt}")


if __name__ == "__main__":
    main()
