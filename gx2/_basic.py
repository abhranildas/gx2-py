"""Mean/variance, characteristic function, and random number generation.
Mirrors ``gx2stat.m``, ``gx2char.m`` and ``gx2rnd.m``.
"""

import numpy as np
from scipy.stats import ncx2, chi2, norm

from ._helpers import asrow
from ._convert import gx2_to_norm_quad_params


def stat(w, k, lambda_, s, m):
    """Mean and variance of a generalized chi-square distribution.

    Parameters
    ----------
    w : array_like
        Weights of the non-central chi-square terms.
    k : array_like
        Degrees of freedom of the non-central chi-square terms.
    lambda_ : array_like
        Non-centrality parameters of the non-central chi-square terms.
    s : float
        Scale (standard deviation) of the added normal term.
    m : float
        Constant offset added to the distribution.

    Returns
    -------
    mu : float
        Mean.
    v : float
        Variance.
    """
    w = asrow(w)
    k = asrow(k)
    lambda_ = asrow(lambda_)
    mu = float(np.dot(w, k + lambda_) + m)
    v = float(2 * np.dot(w ** 2, k + 2 * lambda_) + s ** 2)
    return mu, v


def char(t, w, k, lambda_, s, m):
    """Characteristic function of a generalized chi-square distribution.

    Parameters
    ----------
    t : array_like
        Point(s) at which to evaluate the characteristic function.
    w : array_like
        Weights of the non-central chi-square terms.
    k : array_like
        Degrees of freedom of the non-central chi-square terms.
    lambda_ : array_like
        Non-centrality parameters of the non-central chi-square terms.
    s : float
        Scale (standard deviation) of the added normal term.
    m : float
        Constant offset added to the distribution.

    Returns
    -------
    phi : ndarray of complex
        The characteristic function at each ``t``, shaped like ``t``.
    """
    w = asrow(w)
    k = asrow(k)
    lambda_ = asrow(lambda_)
    t = np.asarray(t, dtype=float)
    tf = t.ravel()
    tc = tf[:, None]  # column

    term = np.sum((w * lambda_) / (1 - 2j * tc * w), axis=1)
    denom = np.prod((1 - 2j * w * tc) ** (k / 2), axis=1)
    phi = np.exp(1j * m * tf + 1j * tf * term - s ** 2 * tf ** 2 / 2) / denom
    return phi.reshape(t.shape)


def rnd(w, k, lambda_, s, m, size=None, method="sum"):
    """Generate generalized chi-square random numbers.

    Parameters
    ----------
    w : array_like
        Weights of the non-central chi-square terms.
    k : array_like
        Degrees of freedom of the non-central chi-square terms.
    lambda_ : array_like
        Non-centrality parameters of the non-central chi-square terms.
    s : float
        Scale (standard deviation) of the added normal term.
    m : float
        Constant offset added to the distribution.
    size : int or tuple, optional
        Output shape. A scalar ``n`` gives an ``n x n`` array; a tuple gives
        that exact shape. If omitted, a single scalar is returned.
    method : {'sum', 'norm_quad'}
        ``'sum'`` (default) generates non-central chi-square and normal numbers
        and adds them. ``'norm_quad'`` generates standard normal vectors and
        computes their quadratic form.

    Returns
    -------
    r : float or ndarray
        Random sample(s), of shape ``size`` (or a scalar if ``size`` is None).
    """
    w = asrow(w)
    k = asrow(k)
    lambda_ = asrow(lambda_)

    if size is None:
        shape = ()
    elif np.isscalar(size):
        shape = (int(size), int(size))
    else:
        shape = tuple(int(x) for x in size)

    method = str(method).lower()
    if method == "sum":
        r = np.zeros(shape)
        for wi, ki, li in zip(w, k, lambda_):
            if li == 0:
                r = r + wi * chi2.rvs(df=ki, size=shape)
            else:
                r = r + wi * ncx2.rvs(df=ki, nc=li, size=shape)
        if s:
            r = r + norm.rvs(loc=m, scale=s, size=shape)
        else:
            r = r + m
        return r
    elif method == "norm_quad":
        quad = gx2_to_norm_quad_params(w, k, lambda_, s, m)
        q1 = np.asarray(quad["q1"]).ravel()
        q2 = np.asarray(quad["q2"])
        q0 = quad["q0"]
        dim = q1.size
        n = int(np.prod(shape)) if shape else 1
        z = norm.rvs(loc=0, scale=1, size=(dim, n))
        r = np.sum(z * (q2 @ z), axis=0) + q1 @ z + q0
        if shape:
            return r.reshape(shape)
        return float(r[0])
    else:
        raise ValueError("method must be 'sum' or 'norm_quad'")
