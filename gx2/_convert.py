"""Conversions between generalized chi-square parameters and the quadratic form
of a normal vector. Mirrors ``gx2_to_norm_quad_params.m`` and
``norm_quad_to_gx2_params.m``.
"""

import numpy as np
from ._helpers import asrow, uniquetol


def gx2_to_norm_quad_params(w, k, l, s, m):
    """Quadratic-form coefficients of the standard normal whose quadratic form
    is the given generalized chi-square.

    Parameters
    ----------
    w, k, l : array_like
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
    l = asrow(l)

    k_int = np.round(k).astype(int)
    q2 = np.repeat(w, k_int).astype(float)   # each w_i, k_i times
    n = int(k_int.sum())
    q1 = np.zeros(n)
    if n:
        # put each w_i*sqrt(l_i) at the start of its block, 0 elsewhere
        starts = np.concatenate(([0], np.cumsum(k_int)[:-1]))
        q1[starts] = w * np.sqrt(l)
    q1 = -2 * q1

    if s:
        q2 = np.append(q2, 0.0)
        q1 = np.append(q1, s)

    return {"q2": np.diag(q2), "q1": q1.astype(float), "q0": float(np.dot(w, l) + m)}


def opt_norm_quad_bd(mu0, v0, mu1, v1, p0=0.5, p1=0.5):
    """Optimal (Bayes) quadratic boundary ``q(x) = x' q2 x + q1' x + q0 = 0``
    between two normal classes, with class 1 favored where ``q(x) > 0``.
    Matches ``IntClassNorm``'s ``opt_class_quad.m`` (class 1 <-> ``norm_1``,
    class 0 <-> ``norm_2``).

    Parameters
    ----------
    mu0, v0 : array_like
        Mean and covariance of class 0.
    mu1, v1 : array_like
        Mean and covariance of class 1.
    p0, p1 : float, optional
        Class priors. Default 0.5 each.

    Returns
    -------
    quad : dict
        ``{'q2': matrix, 'q1': vector, 'q0': scalar}``, for use with
        :func:`norm_err` and :func:`cdf_grad_norm_quad`.
    """
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
    v0 = np.atleast_2d(np.asarray(v0, dtype=float))
    v1 = np.atleast_2d(np.asarray(v1, dtype=float))
    v0inv = np.linalg.inv(v0)
    v1inv = np.linalg.inv(v1)
    q2 = 0.5 * (v0inv - v1inv)
    q1 = v1inv @ mu1 - v0inv @ mu0
    q0 = (0.5 * (mu0 @ v0inv @ mu0 - mu1 @ v1inv @ mu1)
          + 0.5 * (np.linalg.slogdet(v0)[1] - np.linalg.slogdet(v1)[1])
          + np.log(p1 / p0))
    return {"q2": q2, "q1": q1, "q0": float(q0)}


def norm_quad_to_gx2_params(mu, v, quad, merge=True, return_aux=False):
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
    return_aux : bool, optional
        If True, also return the eigen-structure of the standardized
        quadratic (see Returns), reused by ``cdf_grad_norm_quad`` so it need not
        redo the eigendecomposition.

    Returns
    -------
    w, k, l, s, m
    aux : dict, optional
        Only returned when ``return_aux`` is True. The full (pre-merge)
        eigen-structure of the standardized quadratic ``S @ q2_sym @ S``:
        ``S`` (``= Sigma^{1/2}``), ``V`` (eigenvectors), ``d`` (eigenvalues,
        1-D array), and ``b`` (``= V.T @ S @ (2 * q2_sym @ mu + q1)``).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    q2_in = np.asarray(quad["q2"], dtype=float)
    q1_in = np.asarray(quad["q1"], dtype=float).ravel()
    q0_in = float(quad["q0"])

    q2_sym = 0.5 * (q2_in + q2_in.T)

    # sqrtm(v) avoiding small negative eigenvalues
    dv, R = np.linalg.eigh(v)
    dv = np.where(dv < 0, 0.0, dv)
    sqrt_v = R @ np.diag(np.sqrt(dv)) @ R.T

    q2 = sqrt_v @ q2_sym @ sqrt_v
    q2 = (q2 + q2.T) / 2
    q1 = sqrt_v @ (2 * q2_sym @ mu + q1_in)
    q0 = float(mu @ q2_sym @ mu + q1_in @ mu + q0_in)

    d, R2 = np.linalg.eigh(q2)
    b = (R2.T @ q1)

    # Split into nonzero (chi-square) and effectively-zero (linear, feeding the
    # normal term s) eigenvalues using a relative rank tolerance rather than an
    # exact d==0 test: a numerically tiny eigenvalue from a rank-deficient q2
    # would otherwise be kept as a chi-square with a near-zero weight and a
    # blown-up non-centrality b^2/(4w^2), which overflows the Ruben series
    # downstream.
    dtol = np.max(np.abs(d)) * d.size * np.finfo(float).eps
    nz = np.abs(d) > dtol

    if merge:
        w, ic = uniquetol(d[nz])
        k = np.bincount(ic, minlength=w.size).astype(float)
        b_sq_sum = np.bincount(ic, weights=b[nz] ** 2, minlength=w.size)
        l = b_sq_sum / (4 * w ** 2)
    else:
        w = d[nz].copy()
        k = np.ones(w.size)
        l = b[nz] ** 2 / (4 * w ** 2)

    m = q0 - np.dot(w, l)
    s = np.linalg.norm(b[~nz])

    if return_aux:
        aux = {"S": sqrt_v, "V": R2, "d": d, "b": b}
        return w, k, l, s, m, aux
    return w, k, l, s, m
