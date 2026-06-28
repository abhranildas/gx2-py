"""Conversions between generalized chi-square parameters and the quadratic form
of a normal vector. Mirrors ``gx2_to_norm_quad_params.m`` and
``norm_quad_to_gx2_params.m``.
"""

import numpy as np
from ._helpers import asrow, uniquetol


def gx2_to_norm_quad_params(w, k, lambda_, s, m):
    """Quadratic-form coefficients of the standard normal whose quadratic form
    is the given generalized chi-square.

    Parameters
    ----------
    w, k, lambda_ : array_like
        Weights, degrees of freedom and non-centralities of the non-central
        chi-square terms.
    s : float
        Scale of the normal term.
    m : float
        Offset.

    Returns
    -------
    quad : dict
        ``{'q2': matrix, 'q1': vector, 'q0': scalar}``. The dimension of the
        standard normal is ``len(q1)``.
    """
    w = asrow(w)
    k = asrow(k)
    lambda_ = asrow(lambda_)

    q2_parts = []
    q1_parts = []
    for wi, ki, li in zip(w, k, lambda_):
        ki = int(round(ki))
        q2_parts.append(np.full(ki, wi))
        q1_parts.append(np.concatenate(([wi * np.sqrt(li)], np.zeros(ki - 1))))
    q2 = np.concatenate(q2_parts) if q2_parts else np.array([])
    q1 = -2 * (np.concatenate(q1_parts) if q1_parts else np.array([]))

    if s:
        q2 = np.append(q2, 0.0)
        q1 = np.append(q1, s)

    return {"q2": np.diag(q2), "q1": q1.astype(float), "q0": float(np.dot(w, lambda_) + m)}


def norm_quad_to_gx2_params(mu, v, quad, merge=True):
    """Parameters of the generalized chi-square distribution of a quadratic
    form ``q(x) = x' q2 x + q1' x + q0`` of a normal vector ``x ~ N(mu, v)``.

    Parameters
    ----------
    mu : array_like
        Column vector of the normal mean.
    v : array_like
        Normal covariance matrix.
    quad : dict
        ``{'q2': matrix, 'q1': vector, 'q0': scalar}``.
    merge : bool, optional
        If True (default), merge non-central chi-square components with
        close-enough weights into single components. Set False to return all
        raw exact components.

    Returns
    -------
    w, k, lambda_, s, m
    """
    mu = np.asarray(mu, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    q2_in = np.asarray(quad["q2"], dtype=float)
    q1_in = np.asarray(quad["q1"], dtype=float).ravel()
    q0_in = float(quad["q0"])

    q2_sym = 0.5 * (q2_in + q2_in.T)

    # sqrtm(v) avoiding small negative eigenvalues
    d, R = np.linalg.eigh(v)
    d = np.where(d < 0, 0.0, d)
    sqrt_v = R @ np.diag(np.sqrt(d)) @ R.T

    q2 = sqrt_v @ q2_sym @ sqrt_v
    q2 = (q2 + q2.T) / 2
    q1 = sqrt_v @ (2 * q2_sym @ mu + q1_in)
    q0 = float(mu @ q2_sym @ mu + q1_in @ mu + q0_in)

    d2, R2 = np.linalg.eigh(q2)
    d = d2
    b = (R2.T @ q1)

    nz = d != 0
    if merge:
        w, ic = uniquetol(d[nz])
        k = np.bincount(ic, minlength=w.size).astype(float)
        b_sq_sum = np.bincount(ic, weights=b[nz] ** 2, minlength=w.size)
        lambda_ = b_sq_sum / (4 * w ** 2)
    else:
        w = d[nz].copy()
        k = np.ones(w.size)
        lambda_ = b[nz] ** 2 / (4 * w ** 2)

    m = q0 - np.dot(w, lambda_)
    s = np.linalg.norm(b[~nz])
    return w, k, lambda_, s, m
