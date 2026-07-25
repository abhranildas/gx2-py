"""Robust nx-th derivative in x of the generalized chi-square pdf.
Mirrors ``gx2_dens_deriv.m``.
"""

import numpy as np
from scipy.integrate import quad

from ._helpers import asrow
from ._methods import imhof, ruben, _ruben_coeffs, _ruben_eval


def dens_deriv(x, w, k, l, s, m, nx, AbsTol=1e-10, RelTol=1e-6,
               precision="basic", n_ruben=1000):
    """Robust ``nx``-th derivative in ``x`` of the generalized chi-square pdf.

    Single entry point for the density x-derivatives f', f'', f''' used by
    the gradient/Hessian routines (:func:`gx2.cdf_grad_gx2`,
    :func:`gx2.cdf_grad_norm_quad`). It picks between two exact methods:

    - the Gil-Pelaez (Imhof) t-weighted inversion, used whenever it
      converges comfortably: ``s != 0`` (Gaussian damping), or ``s == 0``
      with total dof large relative to the derivative order;
    - the differentiated shifted-dof (Ruben) series, used at ``s == 0`` with
      small total dof, where the inversion integrand loses convergence. For
      mixed-sign weights the variable is split as ``q = q_+ - q_-``, each
      part a same-sign (elliptical) gx2 that Ruben's series handles, and the
      density is their cross-correlation with the derivatives falling on
      ``q_+``.

    Parameters mirror :func:`gx2.pdf`, plus ``nx`` (the derivative order, 0
    gives the pdf).
    """
    w = asrow(w); k = asrow(k); l = asrow(l)
    x = np.asarray(x, dtype=float)

    D = np.sum(k)   # total degrees of freedom

    # Method choice at s=0 (a nonzero s adds Gaussian damping that makes the
    # inversion converge fast for every order, so the inversion is used
    # then). The Imhof density-derivative integrand behaves like
    # u^{nx-D/2} as u->inf, so at s=0 it is only conditionally convergent
    # for D<=2*nx+2 and slowly convergent just above; the differentiated
    # series is needed there. For same-sign (elliptical) weights the series
    # applies at ANY dof and is 1-3 orders of magnitude faster than the
    # inversion (which converges but crawls for a slowly-decaying
    # integrand), so prefer it regardless of dof. For mixed signs the series
    # needs a convolution, so fall back to it only in the small-dof regime
    # where the inversion loses convergence.
    same_sign = bool(np.all(w > 0) or np.all(w < 0))
    use_series = (s == 0) and (same_sign or (D <= 2 * nx + 3))

    if not use_series:
        if nx == 0:
            # the plain pdf: use pdf's default dispatch (exact where possible)
            from ._distribution import pdf
            fd = pdf(x, w, k, l, s, m, method="auto", precision=precision,
                     AbsTol=AbsTol, RelTol=RelTol)
        else:
            fd, _ = imhof(x, w, k, l, s, m, output="dens", nx=nx,
                         precision=precision, AbsTol=AbsTol, RelTol=RelTol)
        return np.asarray(fd)

    # ---- s=0 series route ----
    pos = w > 0
    neg = w < 0
    if np.all(pos) or np.all(neg):
        # same-sign (elliptical): differentiate Ruben's series directly
        fd, _ = ruben(x, w, k, l, m, output="pdf", nx=nx, n_ruben=n_ruben)
        return np.asarray(fd)

    # mixed sign: q = q_+ - q_-, where q_+ = (positive-weight part) + m has
    # support [m,inf) and q_- = (negated negative-weight part) has support
    # [0,inf). The density is their cross-correlation, whose x-derivatives
    # may be carried on either part. A low-dof density has a non-integrable
    # edge singularity once differentiated, so we carry the derivatives on
    # whichever part is sampled in its smooth interior, away from its own
    # edge (the differentiated factor then never meets that singularity).
    # For a threshold x below the q_+ floor m we differentiate q_-,
    # otherwise q_+:
    #   x <  m:  f^(nx)(x) = (-1)^nx int f_{q+}(x+v) f_{q-}^(nx)(v) dv
    #   x >= m:  f^(nx)(x) =          int f_{q+}^(nx)(x+v) f_{q-}(v) dv
    # At x=m exactly, both floors align into a genuine cusp; it is
    # measure-zero and not hit in practice.
    wp, kp, lp = w[pos], k[pos], l[pos]
    wn, kn, ln = -w[neg], k[neg], l[neg]   # negate -> positive weights

    # Ruben's series coefficients depend only on (w, k, l), not on the
    # evaluation point -- compute them once per dens_deriv call and reuse
    # across every scalar quadrature callback below, rather than
    # rebuilding the series from scratch on each of the (potentially
    # hundreds of) points the inner quad() sweeps per integral.
    coeffs_p = _ruben_coeffs(wp, kp, lp, n_ruben=n_ruben)
    coeffs_n = _ruben_coeffs(wn, kn, ln, n_ruben=n_ruben)

    def fp(y, n):
        v, _ = _ruben_eval(coeffs_p, float(y), m, output="pdf", nx=n)
        return float(v)

    def fm(v, n):
        r, _ = _ruben_eval(coeffs_n, float(v), 0.0, output="pdf", nx=n)
        return float(r)

    xf = x.ravel()
    vals = np.array([_conv_dens_deriv(xx, m, nx, fp, fm, AbsTol, RelTol)
                     for xx in xf])
    return vals.reshape(x.shape)


def _conv_dens_deriv(xx, m, nx, fp, fm, AbsTol, RelTol):
    """One point of the mixed-sign cross-correlation, with the ``nx``
    x-derivatives carried on the part sampled in its interior (see the
    parent function)."""
    if xx < m:
        # Differentiate q_-. Since f_{q+}(xx+v)=0 for v<m-xx, start the
        # integral at that floor; the (integrable) q_+ edge is then the
        # lower endpoint, and the singular f_{q-}^(nx) is evaluated only for
        # v>=m-xx>0, in q_-'s interior.
        val, _ = quad(lambda v: fp(xx + v, 0) * fm(v, nx),
                      m - xx, np.inf, epsabs=AbsTol, epsrel=RelTol)
        return ((-1) ** nx) * val
    else:
        # Differentiate q_+, which is sampled strictly inside its support
        # (xx+v>=m).
        val, _ = quad(lambda v: fp(xx + v, nx) * fm(v, 0),
                      0, np.inf, epsabs=AbsTol, epsrel=RelTol)
        return val
