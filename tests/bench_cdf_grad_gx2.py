"""Benchmark cdf_grad_gx2 (gradient and Hessian of the gx2 cdf wrt its native
parameters) against central finite differences of cdf, for Table 2.1.2 of
derivatives/gx2_derivatives.md.

Ground truth: cdf_grad_gx2 at tight tolerances (AbsTol=1e-13, RelTol=1e-11).
FD: central differences of cdf at default tolerances, swept over the step h;
its error is the best over the sweep, and its time is that of one full
gradient / Hessian at a single h. Errors are max-abs over entries, reported
with and without the entries involving a degree of freedom k.

Usage: python tests/bench_cdf_grad_gx2.py [case ...]   (default: 1 2 3)
"""
import sys
import time

import numpy as np

from gx2 import cdf, cdf_grad_gx2

CASES = {
    1: dict(w=[1, -2], k=[2, 3], l=[1, 0.5], s=1.5, m=1, x=2),
    2: dict(w=[1, -2, 3], k=[1, 2, 1], l=[0.5, 1, 0.5], s=1, m=2, x=5),
    3: dict(w=[1, -1.5], k=[1, 1], l=[0.5, 0.3], s=0, m=0, x=0.5),
}
TIGHT = dict(AbsTol=1e-13, RelTol=1e-11)
STEPS = np.logspace(-6, -1, 11)


def unpack(p, n):
    return p[:n], p[n:2 * n], p[2 * n:3 * n], p[3 * n], p[3 * n + 1]


def F(p, n, x):
    w, k, l, s, m = unpack(p, n)
    return float(np.squeeze(cdf(x, w, k, l, s, m)))


def fd_grad(p, n, x, h):
    P = p.size
    g = np.empty(P)
    for i in range(P):
        e = np.zeros(P); e[i] = h
        g[i] = (F(p + e, n, x) - F(p - e, n, x)) / (2 * h)
    return g


def fd_hess(p, n, x, h):
    P = p.size
    H = np.empty((P, P))
    f0 = F(p, n, x)
    for i in range(P):
        ei = np.zeros(P); ei[i] = h
        H[i, i] = (F(p + ei, n, x) - 2 * f0 + F(p - ei, n, x)) / h**2
        for j in range(i + 1, P):
            ej = np.zeros(P); ej[j] = h
            H[i, j] = H[j, i] = (F(p + ei + ej, n, x) - F(p + ei - ej, n, x)
                                 - F(p - ei + ej, n, x) + F(p - ei - ej, n, x)) / (4 * h**2)
    return H


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def errs(est, truth, kmask):
    d = np.abs(est - truth)
    return d.max(), d[kmask].max() if kmask.any() else 0.0


def run(case):
    c = CASES[case]
    n = len(c["w"])
    p = np.concatenate([c["w"], c["k"], c["l"], [c["s"], c["m"]]]).astype(float)
    x = c["x"]
    args = (x, c["w"], c["k"], c["l"], c["s"], c["m"])

    noK = np.ones(p.size, bool); noK[n:2 * n] = False
    noK2 = np.outer(noK, noK)

    (g_true, H_true), _ = timed(cdf_grad_gx2, *args, hess=True, **TIGHT)
    g_true = g_true.ravel(); H_true = np.squeeze(H_true)

    g_an, t_g = timed(cdf_grad_gx2, *args)
    (_, H_an), t_H = timed(cdf_grad_gx2, *args, hess=True)
    g_an = g_an.ravel(); H_an = np.squeeze(H_an)

    best_g = (np.inf, np.inf, None, None)
    best_H = (np.inf, np.inf, None, None)
    for h in STEPS:
        g, tg = timed(fd_grad, p, n, x, h)
        e_all, e_noK = errs(g, g_true, noK)
        if e_all < best_g[0]:
            best_g = (e_all, e_noK, h, tg)
        H, tH = timed(fd_hess, p, n, x, h)
        e_all, e_noK = errs(H, H_true, noK2)
        if e_all < best_H[0]:
            best_H = (e_all, e_noK, h, tH)

    print(f"\n=== case {case} ===")
    for name, t_an, an, truth, mask, best in (
            ("gradient", t_g, g_an, g_true, noK, best_g),
            ("Hessian", t_H, H_an, H_true, noK2, best_H)):
        a_all, a_noK = errs(an, truth, mask)
        f_all, f_noK, h, t_fd = best
        print(f"{name}: analytic {t_an:.3g} s, FD {t_fd:.3g} s (h={h:.1e}), "
              f"speed ratio FD/an {t_fd / t_an:.2g}")
        print(f"  all entries : an err {a_all:.2g}, FD err {f_all:.2g}, "
              f"acc ratio {f_all / max(a_all, 1e-300):.2g}")
        print(f"  without k   : an err {a_noK:.2g}, FD err {f_noK:.2g}, "
              f"acc ratio {f_noK / max(a_noK, 1e-300):.2g}")


if __name__ == "__main__":
    for cs in (sys.argv[1:] or ["1", "2", "3"]):
        run(int(cs))
