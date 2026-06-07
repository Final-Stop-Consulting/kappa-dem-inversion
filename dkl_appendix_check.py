#!/usr/bin/env python3
"""dkl_appendix_check.py -- verifies manuscript Appendix A (closure-validity measure).

Closed form (Appendix A, Eq. for D_KL(kappa)): relative entropy of the 3D Olbert
kappa VDF from its energy-matched Maxwellian projection,
  D_KL(k) = (3/2) ln[2e/(2k-3)] + ln[Gamma(k+1)/Gamma(k-1/2)] - (k+1)[psi(k+1)-psi(k-1/2)]
Scale-invariant (the kappa scale w cancels); nats are convention-free.
Checks the closed form against direct log-space quadrature of the defining integral.
Expected: 0.695 / 0.320 / 0.187 / 0.033 nats at kappa = 2 / 2.5 / 3 / 6,
agreement with quadrature to ~1e-10 or better.
"""
import numpy as np
from scipy.special import gammaln, digamma
from scipy.integrate import quad

def dkl_closed(k):
    return 1.5*np.log(2*np.e/(2*k-3)) + gammaln(k+1)-gammaln(k-0.5) \
           - (k+1)*(digamma(k+1)-digamma(k-0.5))

def dkl_quad(k, w=1.0):
    lnNk = -1.5*np.log(np.pi*k*w*w) + gammaln(k+1)-gammaln(k-0.5)
    s2 = k*w*w/(2*k-3)
    lnNM = -1.5*np.log(2*np.pi*s2)
    def integrand(v):
        ln_fk = lnNk - (k+1)*np.log1p(v*v/(k*w*w))
        ln_fM = lnNM - v*v/(2*s2)
        return 4*np.pi*v*v*np.exp(ln_fk)*(ln_fk-ln_fM)
    return quad(integrand, 0, np.inf, limit=400)[0]

if __name__ == '__main__':
    print(f"{'kappa':>6} {'closed':>10} {'quadrature':>12} {'diff':>10}")
    ok = True
    for k in (2.0, 2.5, 3.0, 6.0):
        c, q = dkl_closed(k), dkl_quad(k)
        ok &= abs(c-q) < 1e-8
        print(f"{k:>6} {c:>10.5f} {q:>12.5f} {abs(c-q):>10.1e}")
    assert abs(dkl_closed(2.5)-0.31967) < 5e-5, "kappa=2.5 gate failed"
    assert abs(dkl_closed(6.0)-0.03348) < 5e-5, "kappa=6 gate failed"
    print("ALL GATES PASS" if ok else "QUADRATURE MISMATCH")
