"""Deterministic computation methods for the generalized chi-square cdf/pdf:
Imhof, Ruben, IFFT, Pearson, infinite-tail and ellipse approximations.

Mirrors gx2_imhof.m, gx2_imhof_integrand.m, gx2_ruben.m, gx2_ifft.m,
gx2_pearson.m, gx2cdf_pearson.m, gx2_tail.m and gx2_ellipse.m.
"""

import warnings
import numpy as np
from scipy.stats import ncx2, chi2
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma as _gamma

import mpmath as mp

from ._helpers import asrow, uniquetol
from ._basic import stat, char

REALMIN = np.finfo(float).tiny  # ~2.2e-308


# --- noncentral chi-square wrappers that tolerate nc == 0 -------------------

def _ncx2cdf(x, df, nc, upper=False):
    if nc == 0:
        return chi2.sf(x, df) if upper else chi2.cdf(x, df)
    return ncx2.sf(x, df, nc) if upper else ncx2.cdf(x, df, nc)


def _ncx2pdf(x, df, nc):
    if nc == 0:
        return chi2.pdf(x, df)
    return ncx2.pdf(x, df, nc)


# ===========================================================================
# Imhof-Davies method
# ===========================================================================

def imhof_integrand(u, x, w, k, lambda_, s, m, output):
    """Imhof integrand; ``w,k,lambda_`` are 1-D."""
    w2u2 = w ** 2 * u ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        theta = np.sum(k * np.arctan(w * u) + (lambda_ * (w * u)) / (1 + w2u2)) / 2 + u * (m - x) / 2
        rho = np.prod(((1 + w2u2) ** (k / 4)) * np.exp((w2u2 * lambda_) / (2 * (1 + w2u2)))) * np.exp(u ** 2 * s ** 2 / 8)
        if output == "cdf":
            return np.sin(theta) / (u * rho)
        return np.cos(theta) / rho


def _imhof_integrand_mp(u, x, w, k, lambda_, s, m, output):
    u = mp.mpf(u)
    theta = mp.mpf(0)
    rho = mp.mpf(1)
    for wi, ki, li in zip(w, k, lambda_):
        wi, ki, li = mp.mpf(wi), mp.mpf(ki), mp.mpf(li)
        w2u2 = wi ** 2 * u ** 2
        theta += ki * mp.atan(wi * u) + (li * (wi * u)) / (1 + w2u2)
        rho *= ((1 + w2u2) ** (ki / 4)) * mp.e ** ((w2u2 * li) / (2 * (1 + w2u2)))
    theta = theta / 2 + u * (mp.mpf(m) - mp.mpf(x)) / 2
    rho *= mp.e ** (u ** 2 * mp.mpf(s) ** 2 / 8)
    if output == "cdf":
        return mp.sin(theta) / (u * rho)
    return mp.cos(theta) / rho


def imhof(x, w, k, lambda_, s, m, side="lower", output="cdf",
          precision="basic", AbsTol=1e-10, RelTol=1e-2):
    """Imhof-Davies method for the cdf (or pdf) of a generalized chi-square."""
    w = asrow(w); k = asrow(k); lambda_ = asrow(lambda_)
    x = np.asarray(x, dtype=float)
    xf = x.ravel()

    integral = np.empty(xf.size)
    if precision == "basic":
        for i, xi in enumerate(xf):
            integral[i] = quad(imhof_integrand, 0, np.inf,
                               args=(xi, w, k, lambda_, s, m, output),
                               epsabs=AbsTol, epsrel=RelTol, limit=200)[0]
    elif precision == "vpa":
        for i, xi in enumerate(xf):
            val = mp.quad(lambda u: _imhof_integrand_mp(u, xi, w, k, lambda_, s, m, output),
                          [0, mp.inf])
            integral[i] = float(val)
    else:
        raise ValueError("precision must be 'basic' or 'vpa'")

    integral = integral.reshape(x.shape)

    if output == "cdf":
        if side == "lower":
            p = 0.5 - integral / np.pi
        else:
            p = 0.5 + integral / np.pi
        errflag = (p < 0) | (p > 1)
        p = np.minimum(p, 1)
    else:
        p = integral / (2 * np.pi)
        errflag = p < 0

    if np.any(errflag):
        warnings.warn("Imhof method output(s) too close to limit to compute "
                      "exactly, so clipping. Check the errflag output, and try "
                      "stricter tolerances.")
        p = np.maximum(p, 0)
    return p, errflag


# ===========================================================================
# Ruben's series method
# ===========================================================================

def ruben(x, w, k, lambda_, m, side="lower", output="cdf", n_ruben=1000):
    """Ruben's series. Requires all ``w`` the same sign and ``s == 0``."""
    w = asrow(w); k = asrow(k); lambda_ = asrow(lambda_)
    x = np.asarray(x, dtype=float)
    xf = x.ravel().astype(float).copy()

    if not (np.all(w > 0) or np.all(w < 0)):
        raise ValueError("Ruben's method needs all w the same sign.")

    w_pos = True
    if np.all(w < 0):
        w = -w; xf = -xf; m = -m; w_pos = False

    beta = 0.90625 * np.min(w)
    M = np.sum(k)
    n = np.arange(1, n_ruben).reshape(-1, 1)  # (n_ruben-1, 1)

    g = (np.sum(k * (1 - beta / w) ** n, axis=1)
         + (beta * n.ravel() * ((1 - beta / w) ** (n - 1) @ (lambda_ / w))))

    a = np.full(n_ruben, np.nan)
    a[0] = np.sqrt(np.exp(-np.sum(lambda_)) * beta ** M * np.prod(w ** (-k)))
    if a[0] < REALMIN:
        raise FloatingPointError("Underflow: some series coefficients are "
                                 "smaller than machine precision.")
    for j in range(1, n_ruben):
        a[j] = np.dot(np.flip(g[:j]), a[:j]) / (2 * j)

    kgrid = (M + 2 * np.arange(n_ruben)).reshape(-1, 1)   # (n_ruben, 1)
    xgrid = ((xf - m) / beta).reshape(1, -1)              # (1, n_x)

    upper = (w_pos and side == "upper") or ((not w_pos) and side == "lower")
    if output == "cdf":
        if upper:
            F = chi2.sf(xgrid, kgrid)
        else:
            F = chi2.cdf(xgrid, kgrid)
    else:
        F = chi2.pdf(xgrid, kgrid)

    p = a @ F  # (n_x,)
    if output == "pdf":
        p = p / beta

    p_err = (1 - np.sum(a)) * chi2.cdf((xf - m) / beta, M + 2 * n_ruben)

    return p.reshape(x.shape), p_err.reshape(x.shape)


# ===========================================================================
# IFFT method
# ===========================================================================

def ifft(x, w, k, lambda_, s, m, side="lower", output="cdf",
         span=None, n_grid=int(1e6) + 1, ft_type="cft"):
    """IFFT method. ``x='full'`` returns the cdf/pdf over a spanning grid."""
    w = asrow(w); k = asrow(k); lambda_ = asrow(lambda_)
    full = isinstance(x, str) and x.lower() == "full"
    if not full:
        x = np.asarray(x, dtype=float)

    if span is None:
        if full:
            mu, v = stat(w, k, lambda_, s, m)
            span = np.max(np.abs(mu + np.array([-1, 1]) * 100 * np.sqrt(v)))
        else:
            span = 1e5

    n_grid = int(round(n_grid))
    if n_grid % 2 == 0:
        n_grid += 1
    n = (n_grid - 1) // 2
    idx = np.arange(-n, n + 1)
    dx = span / n

    if full:
        x_mid = 0.0
    else:
        x_mid = (np.min(x) + np.max(x)) / 2
    xgrid = x_mid + idx * dx

    if ft_type == "dft":
        from ._distribution import pdf as _pdf
        w_idx = np.nonzero(w)[0]
        ncpdfs = np.empty((w_idx.size, xgrid.size))
        for i, wi in enumerate(w_idx):
            off = m if i == 0 else 0.0
            pdfv = _pdf(xgrid, w[wi], k[wi], lambda_[wi], 0, off)
            pdfv = np.asarray(pdfv, dtype=float)
            finite = pdfv[~np.isinf(pdfv)]
            pdfv[np.isinf(pdfv)] = finite.max() if finite.size else 0.0
            ncpdfs[i, :] = pdfv
        if s:
            from scipy.stats import norm
            ncpdfs = np.vstack([ncpdfs, norm.pdf(xgrid, 0, abs(s))])
        phi = np.prod(np.fft.fft(np.fft.ifftshift(ncpdfs, axes=1), axis=1), axis=0)
        p = np.fft.fftshift(np.fft.ifft(phi))
        p = np.real(p)
        p = p / (np.sum(p) * dx)
    else:  # cft
        dt = 2 * np.pi / (n_grid * dx)
        t = idx * dt
        phi = char(-t, w, k, lambda_, s, m)
        if output == "pdf":
            phi = phi * np.exp(1j * x_mid * dt * idx)
            p = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(phi))) / dx)
        else:  # cdf
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = phi / (1j * t) * np.exp(1j * x_mid * dt * idx)
            phi[~np.isfinite(phi)] = 0
            p = 0.5 + np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(phi))) / dx)

    if output == "cdf" and side == "upper":
        p = 1 - p

    if not full:
        F = interp1d(xgrid, p, bounds_error=False, fill_value=(p[0], p[-1]))
        p = F(x)

    p = np.maximum(p, 0)
    return p, xgrid


# ===========================================================================
# Pearson's 3-moment approximation (Imhof's extension, including s and m)
# ===========================================================================

def pearson(x, w, k, lambda_, s, m, side="lower", output="cdf"):
    """Pearson's 3-moment approximation."""
    w = asrow(w); k = asrow(k); lambda_ = asrow(lambda_)
    x = np.asarray(x, dtype=float)

    mu1 = np.sum(w * (k + lambda_)) + m
    mu2 = 2 * np.sum(w ** 2 * (k + 2 * lambda_)) + s ** 2
    mu3 = 8 * np.sum(w ** 3 * (k + 3 * lambda_))
    h = 8 * mu2 ** 3 / mu3 ** 2

    if mu3 > 0:
        y = (x - mu1) * np.sqrt(2 * h / mu2) + h
        if output == "cdf":
            p = chi2.sf(y, h) if side == "upper" else chi2.cdf(y, h)
        else:
            p = np.sqrt(2 * h / mu2) * chi2.pdf(y, h)
    else:
        mu1 = -mu1
        x = -x
        y = (x - mu1) * np.sqrt(2 * h / mu2) + h
        if output == "cdf":
            p = chi2.cdf(y, h) if side == "upper" else chi2.sf(y, h)
        else:
            p = np.sqrt(2 * h / mu2) * chi2.pdf(y, h)
    return p


def cdf_pearson(x, w, k, lambda_, m, side="lower", output="cdf"):
    """Pearson's 3-moment approximation without a normal term."""
    w = asrow(w); k = asrow(k); lambda_ = asrow(lambda_)
    x = np.asarray(x, dtype=float)
    j = np.arange(1, 4).reshape(-1, 1)
    c = np.sum((w ** j) * (j * lambda_ + k), axis=1)
    h = c[1] ** 3 / c[2] ** 2
    if c[2] > 0:
        y = (x - m - c[0]) * np.sqrt(h / c[1]) + h
        if output == "cdf":
            p = chi2.sf(y, h) if side == "upper" else chi2.cdf(y, h)
        else:
            p = np.sqrt(h / c[1]) * chi2.pdf(y, h)
    else:
        c = np.sum(((-w) ** j) * (j * lambda_ + k), axis=1)
        y = (-(x - m) - c[0]) * np.sqrt(h / c[1]) + h
        if output == "cdf":
            p = chi2.cdf(y, h) if side == "upper" else chi2.sf(y, h)
        else:
            p = np.sqrt(h / c[1]) * chi2.pdf(y, h)
    flag = (p < 0) | (p > 1)
    p = np.clip(p, 0, 1)
    return p, flag


# ===========================================================================
# Das's infinite-tail approximation
# ===========================================================================

def tail(x, w, k, lambda_, s, m, side="lower", output="cdf"):
    """Infinite-tail approximation. Returns log10 values where the result
    underflows double precision (those entries are negative)."""
    w = asrow(w); k = asrow(k); lambda_ = asrow(lambda_)
    x = np.asarray(x, dtype=float)

    # merge into unique w's
    w, ic = uniquetol(w)
    k = np.array([np.sum(asrow(k)[ic == i]) for i in range(w.size)])
    lambda_full = asrow(lambda_)
    lambda_ = np.array([np.sum(lambda_full[ic == i]) for i in range(w.size)])

    if side == "upper":
        masked = w * (w > 0)
        max_idx = int(np.argmax(masked))
        w_max = masked[max_idx]
    else:
        masked = w * (w < 0)
        max_idx = int(np.argmin(masked))
        w_max = masked[max_idx]

    k_max = k[max_idx]
    lambda_max = lambda_[max_idx]

    keep = np.ones(w.size, dtype=bool)
    keep[max_idx] = False
    w_rest = w[keep]
    k_rest = k[keep]
    lambda_rest = lambda_[keep]

    a = (np.exp(m / (2 * w_max) + s ** 2 / (8 * w_max ** 2))
         * np.prod(np.exp((lambda_rest * w_rest) / (2 * (w_max - w_rest)))
                   / (1 - w_rest / w_max) ** (k_rest / 2)))

    xf = x.ravel().astype(float)
    if output == "pdf":
        with np.errstate(divide="ignore", invalid="ignore"):
            p = a / abs(w_max) * _ncx2pdf(xf / w_max, k_max, lambda_max)
        x_tiny = xf[p == 0]
        if lambda_max:
            p_tiny = (np.log10(a) - np.log10(abs(w_max)) - np.log10(2 * np.sqrt(2 * np.pi))
                      + (k_max - 3) / 4 * np.log10(x_tiny / w_max)
                      - (k_max - 1) / 4 * np.log10(lambda_max)
                      + (np.sqrt(lambda_max * x_tiny / w_max) - (lambda_max + x_tiny / w_max) / 2) / np.log(10))
        else:
            p_tiny = (np.log10(a) - np.log10(abs(w_max)) - k_max / 2 * np.log10(2)
                      - np.log10(_gamma(k_max / 2)) + (k_max / 2 - 1) * np.log10(x_tiny / w_max)
                      - x_tiny / (2 * w_max * np.log(10)))
    else:  # cdf
        with np.errstate(divide="ignore", invalid="ignore"):
            p = a * _ncx2cdf(xf / w_max, k_max, lambda_max, upper=True)
        x_tiny = xf[p == 0]
        if lambda_max:
            p_tiny = (np.log10(a) - np.log10(lambda_max ** ((k_max - 1) / 4) * np.sqrt(2 * np.pi))
                      - (np.sqrt(x_tiny / w_max) - np.sqrt(lambda_max)) ** 2 / (2 * np.log(10))
                      + (k_max - 3) / 4 * np.log10(x_tiny / w_max))
        else:
            p_tiny = (np.log10(a) + ((k_max - 2) / 2) * np.log10(x_tiny / (2 * w_max))
                      - x_tiny / (2 * w_max * np.log(10)) - np.log10(_gamma(k_max / 2)))

    p_tiny = np.asarray(p_tiny, dtype=float)
    p_tiny[np.isneginf(p_tiny)] = 0
    p = p.astype(float)
    p[p == 0] = p_tiny
    if np.any(p < 0):
        warnings.warn("Some output values are too small for double precision. "
                      "Returning their log10 values, which are negative.")
    return p.reshape(x.shape)


# ===========================================================================
# Ellipse approximation
# ===========================================================================

def ellipse(x, w, r, lambda_, m, side="lower", output="cdf", x_scale="linear"):
    """Ellipse approximation near the finite tail. Requires all ``w`` the same
    sign and ``s == 0``. With ``x_scale='log'`` the inputs and outputs are
    log10 values."""
    w = asrow(w); r = asrow(r); lambda_ = asrow(lambda_)
    x = np.asarray(x, dtype=float)
    xf = x.ravel().astype(float).copy()

    if not (np.all(w > 0) or np.all(w < 0)):
        raise ValueError("The ellipse approximation needs all w the same sign.")

    w_pos = True
    if np.all(w < 0):
        w = -w; m = -m; w_pos = False
        if x_scale == "linear":
            xf = -xf

    ellipse_center = np.concatenate(
        [np.concatenate(([np.sqrt(li)], np.zeros(int(round(ki)) - 1)))
         for li, ki in zip(lambda_, r)])
    ellipse_weights = np.concatenate(
        [np.full(int(round(ki)), wi) for wi, ki in zip(w, r)])

    dim = int(np.sum(r))
    cen_norm2 = np.sum(ellipse_center ** 2)
    a = np.exp(-cen_norm2 / 2) / (2 ** (dim / 2) * _gamma(dim / 2 + 1) * np.sqrt(np.prod(ellipse_weights)))

    if x_scale == "linear":
        x_eff = np.maximum(xf - m, 0)
        if output == "cdf":
            p = a * x_eff ** (dim / 2)
            if (w_pos and side == "upper") or ((not w_pos) and side == "lower"):
                p = 1 - p
        else:
            p = (a * dim / 2) * (xf - m) ** (dim / 2 - 1)
    else:
        log10_x = xf
        if output == "cdf":
            p = (dim / 2 * (log10_x - np.log10(2)) - cen_norm2 / np.log(100)
                 - np.log10(_gamma(dim / 2 + 1)) - np.sum(np.log10(ellipse_weights)) / 2)
        else:
            p = ((dim / 2 - 1) * log10_x - (dim / 2 + 1) * np.log10(2) + np.log10(dim)
                 - cen_norm2 / np.log(100) - np.log10(_gamma(dim / 2 + 1))
                 - np.sum(np.log10(ellipse_weights)) / 2)

    # relative error bound
    if cen_norm2 > 0:
        if x_scale == "linear":
            rr = np.sqrt(x_eff) / np.sqrt(np.sum(ellipse_center ** 2 * ellipse_weights))
            p_rel_err = cen_norm2 * rr
        else:
            p_rel_err = (2 * np.log10(np.sqrt(cen_norm2)) + log10_x / 2
                         - np.log10(np.sum(ellipse_center ** 2 * ellipse_weights)) / 2)
    else:
        if x_scale == "linear":
            p_rel_err = x_eff / 2 * np.min(ellipse_weights)
        else:
            p_rel_err = log10_x - np.log10(2 * np.min(ellipse_weights))

    return np.asarray(p).reshape(x.shape), np.asarray(p_rel_err).reshape(x.shape)
