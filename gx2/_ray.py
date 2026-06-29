"""Ray-trace method for the generalized chi-square cdf/pdf.

Mirrors gx2cdf_ray.m, gx2pdf_ray.m, ray_integrand.m, int_norm_ray.m,
norm_prob_across_rays.m and norm_prob_across_angles.m.

A generalized chi-square is the quadratic form of a standard normal vector;
the ray method integrates the standard multinormal over the region where the
quadratic form is below (or above) a level, by tracing rays from the origin.

Notes vs. MATLAB
----------------
* GPU batching is not ported (CPU/NumPy only); the ``gpu_batch`` argument is
  accepted and ignored.
* ``precision='vpa'`` is reimplemented best-effort with mpmath: sub-``realmin``
  values are returned as ``mpmath.mpf`` objects inside an object array.
"""

import warnings
import numpy as np
from scipy.stats import chi2
from scipy.integrate import quad_vec, dblquad, tplquad
from scipy.special import gamma as _gamma

import mpmath as mp

from ._helpers import (phi_ray, Phibar_ray_split, prob_ray_sym,
                       standard_quad, signed_log_sum_exp)
from ._convert import gx2_to_norm_quad_params

LOG10 = np.log(10)


def _phi_ray_sym(z, dim):
    z = mp.mpf(float(z))
    z2 = z ** 2
    d = mp.mpf(dim)
    return mp.fabs(z) * (z2 ** (d / 2 - 1)) * mp.e ** (-z2 / 2) / (2 ** (d / 2) * mp.gamma(d / 2))


# ===========================================================================
# differential probability / density on each ray (ray_integrand.m)
# ===========================================================================

def ray_integrand(x, n_z, quad, side="upper", output="prob", precision="log"):
    """Probability or density contribution on each ray, to be integrated over
    rays. Returns ``(p_rays, p_tiny_sum, sym_idx)``.

    ``x`` is the array of function levels (n_levels,), ``n_z`` the (dim, n_rays)
    matrix of ray directions (assumed already normalised), ``quad`` the
    standardised quadratic-form dict.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    Q2 = np.asarray(quad["q2"], dtype=float)
    q1v = np.asarray(quad["q1"], dtype=float).ravel()
    q0 = float(quad["q0"])
    dim = q1v.size
    n_levels = x.size
    n_rays = n_z.shape[1]

    q2 = np.sum(n_z * (Q2 @ n_z), axis=0)   # (n_rays,)
    q1 = q1v @ n_z                          # (n_rays,)
    xx = x.reshape(n_levels, 1)
    q0mx = q0 - xx                          # (n_levels, 1)

    delta2 = (q1 ** 2)[None, :] - 4 * q2[None, :] * q0mx     # (n_levels, n_rays)
    root_exists = delta2 > 0
    quad_root_exists = root_exists & (q2[None, :] != 0)
    delta = np.full_like(delta2, np.nan)
    delta[quad_root_exists] = np.sqrt(delta2[quad_root_exists])

    absq2 = np.abs(q2)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = (-q1[None, :] - delta) / (2 * absq2)
        z1 = (-q1[None, :] + delta) / (2 * absq2)
    z = np.stack([z0, z1], axis=2)          # (n_levels, n_rays, 2)

    lin = q2 == 0
    if np.any(lin):
        with np.errstate(divide="ignore", invalid="ignore"):
            z[:, lin, 0] = (-q0mx) / q1[None, lin]
        z[:, lin, 1] = np.nan

    sym_idx = None
    p_tiny_sum = None

    if output == "prob":
        init_sign_rays = np.sign(4 * np.sign(q2)[None, :] - 2 * np.sign(q1)[None, :] + np.sign(q0mx))
        Pbig, Psmall = Phibar_ray_split(z, dim)
        p_rays_big = init_sign_rays + 1 + init_sign_rays * (Pbig[:, :, 1] - Pbig[:, :, 0])
        p_rays_small = init_sign_rays * (Psmall[:, :, 1] - Psmall[:, :, 0])
        if side == "upper":
            p_rays = p_rays_big + p_rays_small
        else:
            p_rays = 2 - p_rays_big - p_rays_small
    else:  # prob_dens
        sum_phi = np.nansum(phi_ray(z, dim), axis=2)
        quad_slope = np.full_like(delta2, np.nan)
        quad_slope[root_exists] = np.sqrt(delta2[root_exists])
        with np.errstate(divide="ignore", invalid="ignore"):
            p_rays = sum_phi / quad_slope
        p_rays[np.isnan(p_rays)] = 0

    tiny_probs = root_exists & (p_rays == 0)

    if precision == "basic":
        if np.any(tiny_probs):
            kind = "probabilities" if output == "prob" else "probability densities"
            warnings.warn("%.1f%% of rays contain %s less than realmin=1e-308, "
                          "returning 0. Set precision to 'log' or 'vpa' to compute "
                          "these." % (100 * np.mean(tiny_probs), kind))
        return p_rays, p_tiny_sum, sym_idx

    if precision == "log":
        if not np.any(tiny_probs):
            # no sub-realmin values; nothing to compute in the log domain
            return p_rays, np.full(n_levels, -np.inf), sym_idx
        tiny3 = np.repeat(tiny_probs[:, :, None], 2, axis=2)
        z_tiny = np.full_like(z, np.nan)
        z_tiny[tiny3] = z[tiny3]
        z_med = np.sqrt(chi2.ppf(0.5, dim))

        if output == "prob":
            signs2 = np.array([-1.0, 1.0]).reshape(1, 1, 2)
            z_tiny_signs = init_sign_rays[:, :, None] * np.sign(z_tiny) * signs2
            if side == "lower":
                z_tiny_signs = -z_tiny_signs
            p_tiny = np.full_like(z_tiny, np.nan)
            lo = np.abs(z_tiny) < z_med
            hi = np.abs(z_tiny) > z_med
            with np.errstate(divide="ignore", invalid="ignore"):
                p_tiny[lo] = dim * np.log10(np.abs(z_tiny[lo])) - np.log10(_gamma(dim / 2) * 2 ** (dim / 2 - 1) * dim)
                p_tiny[hi] = ((dim - 2) * np.log10(np.abs(z_tiny[hi]))
                              - z_tiny[hi] ** 2 / (2 * LOG10)
                              - np.log10(_gamma(dim / 2) * 2 ** (dim / 2 - 1)))
            signed_p_tiny = p_tiny * z_tiny_signs
            p_tiny_sum = signed_log_sum_exp(signed_p_tiny, axis=(1, 2))
        else:  # prob_dens
            with np.errstate(divide="ignore", invalid="ignore"):
                p_tiny = ((dim - 1) * np.log10(np.abs(z_tiny)) - z_tiny ** 2 / (2 * LOG10)
                          - np.log10(_gamma(dim / 2) * 2 ** (dim / 2 - 1)))
            p_tiny[np.isnan(p_tiny)] = -np.inf
            tmp = signed_log_sum_exp(p_tiny, axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                tmp = tmp - np.log10(2 * quad_slope)
            p_tiny_sum = signed_log_sum_exp(tmp, axis=1)
        p_tiny_sum = np.atleast_1d(p_tiny_sum)
        return p_rays, p_tiny_sum, sym_idx

    if precision == "vpa":
        p_tiny_sum = np.array([mp.mpf(0) for _ in range(n_levels)], dtype=object)
        sym_idx = np.zeros(n_levels, dtype=bool)
        if np.any(tiny_probs):
            sym_idx = np.any(tiny_probs, axis=1)
            for lvl in range(n_levels):
                if not sym_idx[lvl]:
                    continue
                acc = mp.mpf(0)
                for ray in range(n_rays):
                    z_ray = z[lvl, ray, :]
                    z_ray = z_ray[np.isfinite(z_ray)]
                    if z_ray.size == 0:
                        continue
                    if output == "prob":
                        acc += prob_ray_sym(init_sign_rays[lvl, ray], z_ray, dim, side)
                    else:
                        qs = quad_slope[lvl, ray]
                        if np.isfinite(qs) and qs != 0:
                            acc += sum(_phi_ray_sym(zz, dim) for zz in z_ray) / mp.mpf(float(qs))
                p_tiny_sum[lvl] = acc
        return p_rays, p_tiny_sum, sym_idx

    raise ValueError("precision must be 'basic', 'log' or 'vpa'")


# ===========================================================================
# probability across rays (norm_prob_across_rays.m), quadratic domain only
# ===========================================================================

def norm_prob_across_rays(mu, v, dom, n_z, side="upper", output="prob",
                          precision="log", fun_level=0.0):
    n_z = np.asarray(n_z, dtype=float)
    n_z = n_z / np.linalg.norm(n_z, axis=0, keepdims=True)
    quad_s = standard_quad(dom, mu, v)
    p_rays, p_tiny_sum, sym_idx = ray_integrand(
        fun_level, n_z, quad_s, side=side, output=output, precision=precision)
    return p_rays, None, p_tiny_sum, sym_idx


# ===========================================================================
# probability across angles (norm_prob_across_angles.m) for grid integration
# ===========================================================================

def norm_prob_across_angles(mu, v, dom, side="upper", output="prob",
                            precision="basic", fun_level=0.0,
                            theta=None, phi=None, psi=None):
    dim = len(mu)
    fun_level = np.atleast_1d(np.asarray(fun_level, dtype=float))

    if dim == 1:
        n_z = np.array([[1.0]])
    elif dim == 2:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        n_z = np.vstack([np.cos(theta), np.sin(theta)])
    elif dim == 3:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        el = np.pi / 2 - theta
        az = phi
        n_z = np.vstack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    elif dim == 4:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        psi = np.atleast_1d(np.asarray(psi, dtype=float))
        xx = np.cos(psi)
        yy = np.sin(psi) * np.cos(theta)
        zz = np.sin(psi) * np.sin(theta) * np.cos(phi)
        ww = np.sin(psi) * np.sin(theta) * np.sin(phi)
        n_z = np.vstack([xx, yy, zz, ww])
    else:
        raise NotImplementedError("grid integration only implemented for dim<=4")

    p_angles, _, _, _ = norm_prob_across_rays(
        mu, v, dom, n_z, side=side, output=output, precision=precision, fun_level=fun_level)
    p_angles = np.asarray(p_angles, dtype=float)

    if dim == 2:
        p_angles = p_angles / np.pi
    elif dim == 3:
        p_angles = p_angles * np.sin(theta).ravel()[None, :] / (2 * np.pi)
    elif dim == 4:
        p_angles = (p_angles * np.sin(theta).ravel()[None, :]
                    * (np.sin(psi).ravel()[None, :] ** 2) / np.pi ** 2)

    if output == "prob":
        p_angles = p_angles / 2
    return p_angles


# ===========================================================================
# integrate the standard multinormal over the quadratic domain (int_norm_ray.m)
# ===========================================================================

def int_norm_ray(mu, v, dom, side="upper", output="prob", force_mc=False,
                 fun_level=0.0, AbsTol=1e-10, RelTol=1e-2, precision="log",
                 n_rays=500, gpu_batch=None, rng=None):
    mu = np.asarray(mu, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    dim = mu.size
    fun_level = np.atleast_1d(np.asarray(fun_level, dtype=float))

    if force_mc or dim > 4:
        if rng is None:
            rng = np.random.default_rng()

        n_levels = fun_level.size
        # Process rays in CPU batches to bound memory (analogous to the MATLAB
        # GPU batching). vpa is not batched (it is used with few rays).
        if precision == "vpa":
            batch = n_rays
        elif gpu_batch and gpu_batch > 0:
            batch = int(gpu_batch)
        else:
            batch = min(n_rays, 200000)
        n_batches = max(1, int(np.ceil(n_rays / batch)))
        sizes = [batch] * (n_batches - 1)
        sizes.append(n_rays - batch * (n_batches - 1))

        p_sum_acc = np.zeros(n_levels)
        p2_acc = np.zeros(n_levels)
        ptiny_batches = []
        p_tiny_sum = None
        sym_idx = None
        for b in sizes:
            n_z = rng.standard_normal((dim, b))
            if precision == "basic":
                pr, _, _, _ = norm_prob_across_rays(
                    mu, v, dom, n_z, side=side, output=output, precision="basic", fun_level=fun_level)
            elif precision == "log":
                pr, _, pts, _ = norm_prob_across_rays(
                    mu, v, dom, n_z, side=side, output=output, precision="log", fun_level=fun_level)
                ptiny_batches.append(np.atleast_1d(pts))
            elif precision == "vpa":
                pr, _, p_tiny_sum, sym_idx = norm_prob_across_rays(
                    mu, v, dom, n_z, side=side, output=output, precision="vpa", fun_level=fun_level)
            else:
                raise ValueError("precision must be 'basic', 'log' or 'vpa'")
            p_sum_acc += np.sum(pr, axis=1)
            p2_acc += np.sum(pr ** 2, axis=1)

        mean_p = p_sum_acc / n_rays
        with np.errstate(invalid="ignore"):
            p_err = np.sqrt(np.maximum(p2_acc / n_rays - mean_p ** 2, 0) / n_rays)
        if precision == "basic":
            p = mean_p
        else:
            p_sum = p_sum_acc
            if precision == "log":
                if n_batches > 1:
                    p_tiny_sum = signed_log_sum_exp(np.stack(ptiny_batches, axis=1), axis=1)
                else:
                    p_tiny_sum = ptiny_batches[0]
                p_tiny_sum = np.atleast_1d(p_tiny_sum)

        if precision == "log":
            p_tiny_sign = -np.sign(p_tiny_sum)
            with np.errstate(over="ignore"):
                p_sum = p_sum + p_tiny_sign * (10.0 ** (-np.abs(p_tiny_sum)))
            p = p_sum / n_rays
            if np.any(p_tiny_sign[p == 0] == -1):
                raise RuntimeError(
                    "p_tiny has wrong sign: the ray method's log-precision "
                    "extension does not support this configuration (typically a "
                    "finite tail). Use method='ellipse' or method='tail' for "
                    "sub-realmin probabilities here.")
            p_tiny_sum = p_tiny_sum - np.log10(n_rays)
            p_tiny_sum = np.where(np.isneginf(p_tiny_sum), 0.0, p_tiny_sum)
            p = np.where(p == 0, p_tiny_sum, p)
            if np.any(p < 0):
                warnings.warn("Some output values are too small for double "
                              "precision. Returning their log10 values, which "
                              "are negative.")
        elif precision == "vpa":
            p = (p_sum / n_rays).astype(object)
            for i in range(p.size):
                if sym_idx is not None and sym_idx[i] and p_sum[i] == 0:
                    p[i] = p_tiny_sum[i] / n_rays

        if output == "prob":
            if p.dtype == object:
                p = np.array([pp / 2 for pp in p], dtype=object)
            else:
                pos = p > 0
                neg = p < 0
                p = np.where(pos, p / 2, p)
                p = np.where(neg, p - np.log10(2), p)

    else:
        # grid integration (dim 1..4)
        if dim == 1:
            p = norm_prob_across_angles(mu, v, dom, side=side, output=output,
                                        precision=precision, fun_level=fun_level)
            p = np.asarray(p, dtype=float).ravel()
        elif dim == 2:
            def integrand(th):
                return norm_prob_across_angles(mu, v, dom, side=side, output=output,
                                               precision="basic", fun_level=fun_level,
                                               theta=th).ravel()
            p, _ = quad_vec(integrand, 0, np.pi, epsabs=AbsTol, epsrel=RelTol)
        elif dim == 3:
            p = np.empty(fun_level.size)
            for i, f in enumerate(fun_level):
                val, _ = dblquad(
                    lambda ph, th: float(norm_prob_across_angles(
                        mu, v, dom, side=side, output=output, precision="basic",
                        fun_level=np.array([f]), theta=th, phi=ph)),
                    0, np.pi / 2, 0, 2 * np.pi, epsabs=AbsTol, epsrel=RelTol)
                p[i] = val
        elif dim == 4:
            p = np.empty(fun_level.size)
            for i, f in enumerate(fun_level):
                val, _ = tplquad(
                    lambda ps, ph, th: float(norm_prob_across_angles(
                        mu, v, dom, side=side, output=output, precision="basic",
                        fun_level=np.array([f]), theta=th, phi=ph, psi=ps)),
                    0, np.pi / 2, 0, 2 * np.pi, 0, np.pi, epsabs=AbsTol, epsrel=RelTol)
                p[i] = val
        p_err = None

    return p, p_err


# ===========================================================================
# public ray wrappers (gx2cdf_ray.m / gx2pdf_ray.m)
# ===========================================================================

def cdf_ray(x, w, k, l, s, m, side="lower", **kwargs):
    x = np.asarray(x, dtype=float)
    y = x.ravel()
    quad = gx2_to_norm_quad_params(w, k, l, s, m)
    dim = quad["q1"].size
    mu = np.zeros(dim)
    v = np.eye(dim)
    p, p_err = int_norm_ray(mu, v, quad, side=side, output="prob", fun_level=y, **kwargs)
    p = np.asarray(p).reshape(x.shape) if x.ndim else np.asarray(p).reshape(())
    if p_err is not None:
        p_err = np.asarray(p_err).reshape(x.shape) if x.ndim else np.asarray(p_err).reshape(())
    return p, p_err


def pdf_ray(x, w, k, l, s, m, n_rays=1000, **kwargs):
    x = np.asarray(x, dtype=float)
    y = x.ravel()
    quad = gx2_to_norm_quad_params(w, k, l, s, m)
    dim = quad["q1"].size
    mu = np.zeros(dim)
    v = np.eye(dim)
    f, f_err = int_norm_ray(mu, v, quad, output="prob_dens", fun_level=y,
                            n_rays=n_rays, **kwargs)
    f = np.asarray(f).reshape(x.shape) if x.ndim else np.asarray(f).reshape(())
    if f_err is not None:
        f_err = np.asarray(f_err).reshape(x.shape) if x.ndim else np.asarray(f_err).reshape(())
    return f, f_err
