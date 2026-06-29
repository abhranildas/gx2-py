"""Top-level cdf, pdf and inverse-cdf with automatic method selection.
Mirrors gx2cdf.m, gx2pdf.m, gx2inv.m and log_gx2cdf.m.
"""

import inspect
import numpy as np
from scipy.stats import norm

from ._helpers import asrow, fzero
from ._basic import stat
from ._methods import (imhof, ruben, ifft, pearson, tail,
                       ellipse, _ncx2cdf)
from ._ray import cdf_ray, pdf_ray, int_norm_ray


def _filter(func, kwargs):
    """Keep only kwargs accepted by ``func`` (mimics MATLAB KeepUnmatched)."""
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _is_full(x):
    return isinstance(x, str) and x.lower() == "full"


# ===========================================================================
# cdf
# ===========================================================================

def cdf(x, w, k, l, s, m, side="lower", method="auto",
        full_output=False, **kwargs):
    """Cdf of a generalized chi-square distribution.

    Parameters
    ----------
    x : float, array_like, or 'full'
        Point(s) at which to evaluate the cdf. The output has the same shape as
        ``x``. Pass the string ``'full'`` to evaluate the cdf over a grid that
        automatically spans the whole distribution; the grid is then returned
        as ``x_grid`` (see Returns), so use ``'full'`` together with the
        returned tuple.
    w : array_like
        Weights of the non-central chi-square terms.
    k : array_like
        Degrees of freedom of the non-central chi-square terms.
    l : array_like
        Non-centrality parameters of the non-central chi-square terms (one per
        term, same length as ``w`` and ``k``).
    s : float
        Scale (standard deviation) of the added normal term.
    m : float
        Constant offset added to the distribution.
    side : {'lower', 'upper'}
        ``'lower'`` (default) returns the cdf P(X <= x). ``'upper'`` returns the
        complementary cdf P(X > x), computed in a way that stays accurate when
        it is very small.
    method : {'auto','imhof','ray','ifft','ruben','tail','pearson','ellipse'}
        Algorithm used. ``'auto'`` (default) picks a suitable one for the given
        parameters. The others let you choose explicitly; some have constraints
        (e.g. ``'ruben'`` and ``'ellipse'`` need all ``w`` the same sign and
        ``s == 0``).
    full_output : bool
        Controls how many values are returned (see Returns). Defaults to False,
        and is forced True when ``x='full'``.

    Returns
    -------
    p : float or ndarray
        The cdf (or complementary cdf if ``side='upper'``) at each ``x``, shaped
        like ``x``. This is the sole return value when ``full_output`` is False.
        With the ``'tail'``, ``'ellipse'`` and ``'ray'`` methods, any entry that
        is too small for double precision (below ~1e-308) is returned instead as
        its base-10 logarithm, which is negative; entries that fit normally stay
        positive. This lets the far tails be represented down to arbitrarily
        tiny probabilities.
    p_err : ndarray or None
        Returned only when ``full_output=True``. An estimate of the numerical
        error in ``p`` (its exact meaning depends on the method, e.g. the
        Monte-Carlo standard error for ``'ray'`` or an error bound for
        ``'ruben'``), or ``None`` for methods that do not provide one.
    x_grid : ndarray or None
        Returned only when ``full_output=True``. When ``x='full'``, the array of
        points at which ``p`` was evaluated; otherwise ``None``.

    Examples
    --------
    >>> import gx2
    >>> gx2.cdf(25, [1, -5, 2], [1, 2, 3], [2, 3, 7], 0, 5)          # cdf at x=25
    >>> gx2.cdf(25, [1, -5, 2], [1, 2, 3], [2, 3, 7], 0, 5, side='upper')
    >>> p, p_err, x_grid = gx2.cdf('full', [1, -5, 2], [1, 2, 3], [2, 3, 7], 0, 5,
    ...                            full_output=True)
    """
    w = asrow(w); k = asrow(k); l = asrow(l)
    full = _is_full(x)
    if not full:
        x = np.asarray(x, dtype=float)
    p_err = None
    x_grid = None

    if full:
        method = "ifft"

    if method == "auto":
        uw = np.unique(w)
        if (not s) and uw.size == 1:
            uw0 = uw[0]
            lower = ((np.sign(uw0) == 1 and side == "lower")
                     or (np.sign(uw0) == -1 and side == "upper"))
            p = _ncx2cdf((x - m) / uw0, np.sum(k), np.sum(l), upper=not lower)
        elif np.sum(np.abs(w)) == 0 and s:
            p = norm.sf(x, m, s) if side == "upper" else norm.cdf(x, m, s)
        elif not s:
            if (np.all(w > 0) and side == "lower") or (np.all(w < 0) and side == "upper"):
                try:
                    p, p_err = ruben(x, w, k, l, m, side=side,
                                     **_filter(ruben, kwargs))
                except Exception:
                    p, p_err = imhof(x, w, k, l, 0, m, side=side,
                                     **_filter(imhof, kwargs))
            else:
                p, p_err = imhof(x, w, k, l, s, m, side=side,
                                 **_filter(imhof, kwargs))
        else:
            p, p_err = imhof(x, w, k, l, s, m, side=side,
                             **_filter(imhof, kwargs))
    elif method == "ifft":
        p, x_grid = ifft(x, w, k, l, s, m, side=side, output="cdf",
                         **_filter(ifft, kwargs))
    elif method == "ray":
        p, p_err = cdf_ray(x, w, k, l, s, m, side=side,
                           **_filter(int_norm_ray, kwargs))
    elif method == "imhof":
        p, p_err = imhof(x, w, k, l, s, m, side=side,
                         **_filter(imhof, kwargs))
    elif method == "ruben":
        if s or not (np.all(w > 0) or np.all(w < 0)):
            raise ValueError("Ruben's method can only be used when all w are "
                             "the same sign and s=0.")
        p, p_err = ruben(x, w, k, l, m, side=side,
                         **_filter(ruben, kwargs))
    elif method == "tail":
        p = tail(x, w, k, l, s, m, side=side, **_filter(tail, kwargs))
    elif method == "pearson":
        p = pearson(x, w, k, l, s, m, side=side, **_filter(pearson, kwargs))
    elif method == "ellipse":
        if s or not (np.all(w > 0) or np.all(w < 0)):
            raise ValueError("The ellipse approximation can only be used when "
                             "all w are the same sign and s=0.")
        p, p_err = ellipse(x, w, k, l, m, side=side,
                           **_filter(ellipse, kwargs))
    else:
        raise ValueError("unknown method %r" % method)

    if full or full_output:
        return p, p_err, x_grid
    return p


# ===========================================================================
# pdf
# ===========================================================================

def pdf(x, w, k, l, s, m, side="lower", method="auto", diff=False,
        dx=None, full_output=False, **kwargs):
    """Pdf of a generalized chi-square distribution.

    Parameters
    ----------
    x : float, array_like, or 'full'
        Point(s) at which to evaluate the pdf; output is shaped like ``x``.
        ``'full'`` evaluates over an automatically chosen spanning grid, which
        is returned as ``x_grid`` (see Returns).
    w : array_like
        Weights of the non-central chi-square terms.
    k : array_like
        Degrees of freedom of the non-central chi-square terms.
    l : array_like
        Non-centrality parameters of the non-central chi-square terms.
    s : float
        Scale (standard deviation) of the added normal term.
    m : float
        Constant offset added to the distribution.
    side : {'lower', 'upper'}
        Only affects the ``'tail'`` method, selecting which infinite tail to
        approximate.
    method : {'auto','imhof','ray','ifft','ruben','tail','pearson','ellipse'}
        Algorithm used; ``'auto'`` (default) picks a suitable one.
    diff : bool
        If True, obtain the pdf by numerically differentiating :func:`cdf`
        instead of evaluating it directly.
    dx : float, optional
        Step size for the ``diff=True`` finite difference. Defaults to the
        distribution's standard deviation divided by 1e4.
    full_output : bool
        Controls how many values are returned (see Returns). Defaults to False,
        and is forced True when ``x='full'``.

    Returns
    -------
    f : float or ndarray
        The pdf at each ``x``, shaped like ``x``. This is the sole return value
        when ``full_output`` is False. As with :func:`cdf`, the ``'tail'``,
        ``'ellipse'`` and ``'ray'`` methods return the (negative) base-10
        logarithm for entries too small for double precision.
    f_err : ndarray or None
        Returned only when ``full_output=True``. A method-dependent estimate of
        the numerical error in ``f``, or ``None`` if unavailable.
    x_grid : ndarray or None
        Returned only when ``full_output=True``. The grid of points used when
        ``x='full'``; otherwise ``None``.
    """
    w = asrow(w); k = asrow(k); l = asrow(l)
    full = _is_full(x)
    if not full:
        x = np.asarray(x, dtype=float)
    f_err = None
    x_grid = None

    if full:
        method = "ifft"

    if not diff:
        if method == "auto":
            uw = np.unique(w)
            if (not s) and uw.size == 1 and not full:
                from ._methods import _ncx2pdf
                f = _ncx2pdf((x - m) / uw[0], np.sum(k), np.sum(l)) / abs(uw[0])
            elif np.sum(np.abs(w)) == 0 and s:
                f = norm.pdf(x, m, s)
            else:
                f, _ = imhof(x, w, k, l, s, m, output="pdf",
                             **_filter(imhof, kwargs))
        elif method == "imhof":
            f, _ = imhof(x, w, k, l, s, m, output="pdf",
                         **_filter(imhof, kwargs))
        elif method == "ruben":
            if s or not (np.all(w > 0) or np.all(w < 0)):
                raise ValueError("Ruben's method can only be used when all w are "
                                 "the same sign and s=0.")
            f, _ = ruben(x, w, k, l, m, output="pdf",
                         **_filter(ruben, kwargs))
        elif method == "tail":
            f = tail(x, w, k, l, s, m, side=side, output="pdf",
                     **_filter(tail, kwargs))
        elif method == "pearson":
            f = pearson(x, w, k, l, s, m, side=side, output="pdf",
                        **_filter(pearson, kwargs))
        elif method == "ellipse":
            if s or not (np.all(w > 0) or np.all(w < 0)):
                raise ValueError("The ellipse approximation can only be used when "
                                 "all w are the same sign and s=0.")
            f, f_err = ellipse(x, w, k, l, m, side=side, output="pdf",
                               **_filter(ellipse, kwargs))
        elif method == "ray":
            f, f_err = pdf_ray(x, w, k, l, s, m,
                               **_filter(pdf_ray, kwargs))
        elif method == "ifft":
            f, x_grid = ifft(x, w, k, l, s, m, side=side, output="pdf",
                             **_filter(ifft, kwargs))
        else:
            raise ValueError("unknown method %r" % method)
    else:
        if dx is None:
            _, v = stat(w, k, l, s, m)
            dx = np.sqrt(v) / 1e4
        p_left = cdf(x - dx, w, k, l, s, m, side=side, method=method, **kwargs)
        p_right = cdf(x + dx, w, k, l, s, m, side=side, method=method, **kwargs)
        f = (p_right - p_left) / (2 * dx)
        f = np.maximum(f, 0)

    if full or full_output:
        return f, f_err, x_grid
    return f


# ===========================================================================
# inverse cdf
# ===========================================================================

def inv(p, w, k, l, s, m, side="lower", method="auto", **kwargs):
    """Inverse cdf (quantile function) of a generalized chi-square distribution.

    Parameters
    ----------
    p : float or array_like
        Probability or probabilities at which to evaluate the quantile, in
        (0, 1]. A negative value is interpreted as the base-10 logarithm of the
        probability, which lets you request quantiles for probabilities below
        ~1e-308 (e.g. ``p=-1000`` means a cdf of 1e-1000); pair this with a
        forward method that reaches such tiny values, e.g. ``method='tail'``,
        ``'ellipse'`` or ``'ray'``.
    w : array_like
        Weights of the non-central chi-square terms.
    k : array_like
        Degrees of freedom of the non-central chi-square terms.
    l : array_like
        Non-centrality parameters of the non-central chi-square terms.
    s : float
        Scale (standard deviation) of the added normal term.
    m : float
        Constant offset added to the distribution.
    side : {'lower', 'upper'}
        ``'upper'`` inverts the complementary cdf, i.e. ``p`` is a tail
        probability.
    method : str
        Forward cdf method used while root-finding; see :func:`cdf`.

    Returns
    -------
    x : float or ndarray
        The quantile(s): the value(s) of x at which the cdf equals ``p``.
        Returns a scalar for scalar ``p``, otherwise an array shaped like ``p``.
    """
    w = asrow(w); k = asrow(k); l = asrow(l)
    p = np.atleast_1d(np.asarray(p, dtype=float))

    uw = np.unique(w)
    if (not s) and uw.size == 1 and np.all(p > 0):
        uw0 = uw[0]
        pp = 1 - p if side == "upper" else p
        from scipy.stats import ncx2, chi2
        df = np.sum(k); nc = np.sum(l)

        def _ncx2inv(pr):
            if nc == 0:
                return chi2.ppf(pr, df)
            return ncx2.ppf(pr, df, nc)

        if np.sign(uw0) == 1:
            x = _ncx2inv(pp) * uw0 + m
        elif np.sign(uw0) == -1:
            x = _ncx2inv(1 - pp) * uw0 + m
        else:
            x = np.zeros_like(pp)
    else:
        mu, _ = stat(w, k, l, s, m)

        def solve_one(pi):
            if pi > 0:
                f = lambda xx: cdf(xx, w, k, l, s, m, side=side,
                                   method=method, **kwargs) - pi
            else:
                f = lambda xx: log_cdf(xx, w, k, l, s, m, side=side,
                                       method=method, **kwargs) - pi
            return fzero(f, mu)

        x = np.array([solve_one(pi) for pi in p])

    x = np.asarray(x)
    if x.size == 1:
        return float(x.ravel()[0])
    return x


def log_cdf(x, w, k, l, s, m, **kwargs):
    """log10 of the cdf, returning the (negative) value itself when the cdf
    has already underflowed to a log10 value."""
    p = cdf(x, w, k, l, s, m, **kwargs)
    p = float(np.asarray(p).ravel()[0]) if np.size(p) == 1 else p
    if np.isscalar(p) or np.ndim(p) == 0:
        return p if p <= 0 else np.log10(p)
    p = np.asarray(p, dtype=float)
    return np.where(p <= 0, p, np.log10(np.where(p > 0, p, 1)))
