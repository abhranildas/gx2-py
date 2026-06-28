"""Internal numerical helpers for the gx2 package.

These mirror the small utility ``.m`` files of the MATLAB toolbox
(``log_sum_exp``, ``signed_log_sum_exp``, ``phi_ray``, ``Phibar_ray_split``,
``Phibar_sym``, ``prob_ray_sym``, ``standard_quad``), plus a couple of
helpers that stand in for MATLAB built-ins (``uniquetol``, ``fzero``).
"""

import numpy as np
from scipy.stats import chi2
from scipy.special import gammainc, gammaincc, gamma as _gamma
from scipy.optimize import brentq
from scipy.linalg import sqrtm

import mpmath as mp


def asrow(x):
    """Return ``x`` as a 1-D float array (MATLAB row vector)."""
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


# ---------------------------------------------------------------------------
# log-sum-exp utilities (base 10), matching log_sum_exp.m / signed_log_sum_exp.m
# ---------------------------------------------------------------------------

def log_sum_exp(logs, axis=None):
    """log10(sum(10**logs)) computed stably. Mirrors ``log_sum_exp.m``."""
    logs = np.asarray(logs, dtype=float)
    if logs.size == 0:
        return -np.inf
    m = np.max(logs, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        s = m + np.log10(np.sum(np.power(10.0, logs - m), axis=axis, keepdims=True))
    s = np.where(np.isneginf(m), -np.inf, s)
    return _squeeze_axis(s, axis)


def signed_log_sum_exp(logs, axis=None):
    """Signed log-space summation, matching ``signed_log_sum_exp.m``.

    A positive value ``x`` is represented by ``-log10(x)`` and a negative
    value ``x`` by ``+log10(|x|)``. The output ``s`` uses the same convention:
    ``s < 0`` means ``S = 10**(-|s|) > 0`` and ``s > 0`` means
    ``S = -10**(-|s|) < 0``.
    """
    logs = np.array(logs, dtype=float)
    logs = np.where(np.isnan(logs), -np.inf, logs)
    if logs.size == 0:
        return -np.inf

    mag = np.abs(logs)
    term_sign = -np.sign(logs)
    m = np.min(mag, axis=axis, keepdims=True)

    with np.errstate(over="ignore", invalid="ignore"):
        terms = term_sign * np.power(10.0, -(mag - m))
    terms = np.where(np.isnan(terms), 0.0, terms)
    r = np.sum(terms, axis=axis, keepdims=True)

    eps = np.finfo(float).eps
    r = np.where(r == 0, eps, r)

    s = np.zeros_like(r)
    pos = r > 0
    neg = r < 0
    with np.errstate(invalid="ignore"):
        s = np.where(pos, -(m - np.log10(np.where(pos, r, 1.0))), s)
        s = np.where(neg, m - np.log10(np.abs(np.where(neg, r, 1.0))), s)
    s = np.where(np.isposinf(m), -np.inf, s)
    return _squeeze_axis(s, axis)


def _squeeze_axis(s, axis):
    if axis is None:
        return float(np.squeeze(s))
    if isinstance(axis, (tuple, list)):
        axis = tuple(axis)
    out = np.squeeze(s, axis=axis)
    if out.ndim == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# ray-method geometry, matching phi_ray.m / Phibar_ray_split.m / Phibar_sym.m
# ---------------------------------------------------------------------------

def phi_ray(z, dim):
    """Standard multinormal density along a ray. Mirrors ``phi_ray.m``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = np.asarray(z, dtype=float) ** 2
        f = (np.abs(z) * (z2 ** (dim / 2 - 1)) * np.exp(-z2 / 2)
             / (2 ** (dim / 2) * _gamma(dim / 2)))
    return f


def Phibar_ray_split(z, dim):
    """Complementary multinormal cdf on a ray, split into a big and small part.

    Mirrors ``Phibar_ray_split.m``.
    """
    z = np.asarray(z, dtype=float)
    z_c = np.sqrt(chi2.ppf(0.5, dim))

    with np.errstate(invalid="ignore"):
        Phibar_big = (z <= -z_c).astype(float) + (z < z_c).astype(float)

        Phibar_small = gammainc(dim / 2, z ** 2 / 2)
        Phibar_small_upper = gammaincc(dim / 2, z ** 2 / 2)
        big_mask = np.abs(z) >= z_c
        Phibar_small = np.where(big_mask, Phibar_small_upper, Phibar_small)

        invert = (z <= -z_c) | ((z > 0) & (z < z_c))
        Phibar_small = Phibar_small * (1 - 2 * invert.astype(float))

    Phibar_big = np.where(np.isnan(Phibar_big), 0.0, Phibar_big)
    Phibar_small = np.where(np.isnan(Phibar_small), 0.0, Phibar_small)
    return Phibar_big, Phibar_small


def Phibar_sym(z, dim):
    """Symbol-friendly complementary cdf on a ray (mpmath). Mirrors ``Phibar_sym.m``."""
    z = mp.mpf(z)
    a = mp.mpf(dim) / 2
    chi_cdf = 1 - mp.gammainc(a, z ** 2 / 2, mp.inf) / mp.gamma(a)
    return (1 - mp.sign(z) * chi_cdf) / 2


def prob_ray_sym(init_sign_ray, z_ray, dim, side="upper"):
    """Symbolic probability slice along a ray. Mirrors ``prob_ray_sym.m``."""
    sgn = mp.mpf(int(init_sign_ray))
    z_ray = [zz for zz in np.atleast_1d(z_ray) if np.isfinite(zz)]
    phibars = [Phibar_sym(zz, dim) for zz in z_ray]
    acc = mp.mpf(0)
    for i, pb in enumerate(phibars, start=1):
        acc += ((-1) ** i) * pb
    p_ray = sgn + 1 + 2 * sgn * acc
    if str(side).lower() == "lower":
        p_ray = 2 - p_ray
    return p_ray


# ---------------------------------------------------------------------------
# quadratic-form standardisation, matching standard_quad.m
# ---------------------------------------------------------------------------

def standard_quad(quad, mu, v):
    """Standardise quadratic-form coefficients. Mirrors ``standard_quad.m``."""
    mu = np.asarray(mu, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    sv = np.real(sqrtm(v))
    q2 = quad["q2"]
    q1 = quad["q1"]
    q0 = quad["q0"]
    qs2 = sv @ q2 @ sv
    qs1 = sv @ (2 * q2 @ mu + q1)
    qs0 = float(mu @ q2 @ mu + q1 @ mu + q0)
    return {"q2": qs2, "q1": qs1, "q0": qs0}


# ---------------------------------------------------------------------------
# stand-ins for MATLAB built-ins
# ---------------------------------------------------------------------------

def uniquetol(a, tol=1e-6):
    """Approximate MATLAB ``uniquetol``: returns (u, ic) with u sorted ascending.

    Elements within ``tol * max(abs(a))`` of a cluster representative are
    merged. ``ic`` maps each input element to its index in ``u``.
    """
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return a.copy(), np.array([], dtype=int)
    scale = np.max(np.abs(a))
    thr = tol * (scale if scale > 0 else 1.0)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    u = [sa[0]]
    ic_sorted = np.zeros(sa.size, dtype=int)
    for i in range(1, sa.size):
        if sa[i] - u[-1] > thr:
            u.append(sa[i])
        ic_sorted[i] = len(u) - 1
    u = np.array(u, dtype=float)
    ic = np.empty(a.size, dtype=int)
    ic[order] = ic_sorted
    return u, ic


def fzero(f, x0, maxiter=2000):
    """Find a root of scalar ``f`` near ``x0``, mimicking MATLAB ``fzero`` with a
    scalar start: expand a search interval until a sign change, then bisect."""
    fx0 = f(x0)
    if fx0 == 0 or not np.isfinite(fx0):
        if fx0 == 0:
            return x0
    dx = (abs(x0) / 50.0) if x0 != 0 else 1.0 / 50.0
    sqrt2 = np.sqrt(2.0)
    for _ in range(maxiter):
        dx *= sqrt2
        a = x0 - dx
        fa = f(a)
        if np.isfinite(fa) and np.sign(fa) != np.sign(fx0) and fx0 != 0:
            return brentq(f, a, x0)
        b = x0 + dx
        fb = f(b)
        if np.isfinite(fb) and np.sign(fb) != np.sign(fx0) and fx0 != 0:
            return brentq(f, x0, b)
    raise RuntimeError("fzero: no sign change found near x0=%g" % x0)
