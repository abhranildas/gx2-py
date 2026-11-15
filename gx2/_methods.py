"""Deterministic computation methods for the generalized chi-square cdf/pdf:
Imhof, Ruben, IFFT, Pearson, infinite-tail and ellipse approximations.

Mirrors gx2_imhof.m, gx2_imhof_integrand.m, gx2_ruben.m, gx2_ifft.m,
gx2_pearson.m, gx2_tail.m and gx2_ellipse.m.
"""

import warnings
from math import comb
import numpy as np
from scipy.stats import ncx2, chi2
from scipy.integrate import quad, quad_vec, IntegrationWarning
from scipy.interpolate import interp1d
from scipy.special import gamma as _gamma

import mpmath as mp

from ._helpers import asrow, uniquetol, ImhofClipWarning
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

def imhof_integrand(u, x, w, k, l, s, m, output, idx=None, nx=0):
    """Imhof integrand for the generalized chi-square inversion; ``w,k,l`` are
    1-D. Beyond the plain cdf/pdf it also returns the exact integrands for
    parameter-gradient and -Hessian components (no finite differencing).

    output:
        ``'cdf'``       cdf integrand.
        ``'pdf'``       pdf integrand.
        ``'dens'``      ``nx``-th x-derivative of the pdf, f^(nx) (nx>=1
                        gives f', f'', ...).
        ``'k_deriv'``   d/dk_idx of the cdf, with ``nx`` extra x-derivatives
                        (``idx`` a single 0-based component index).
        ``'kk_deriv'``  d^2/(dk_idx[0] dk_idx[1]) of the cdf, with ``nx``
                        extra x-derivatives (``idx`` a length-2 sequence).

    The derivative modes use the complex integrand ``Z = exp(i*theta)/rho``
    together with the rule that each x-derivative multiplies ``Z`` by
    ``-(i*u/2)``, and that d/dk_j brings down the factor
    ``ell_j = -1/2*log(1 - i*w_j*u)``.

    Every mode is ``Re(H(u) * exp(-i*c*u))`` with ``c = (x-m)/2`` and the
    x-independent amplitude ``H`` from :func:`_imhof_amplitude`.
    """
    H = _imhof_amplitude(u, w, k, l, s, output, idx, nx)
    return np.real(H * np.exp(-0.5j * u * (np.asarray(x) - m)))


def _imhof_amplitude(u, w, k, l, s, output, idx=None, nx=0):
    """Complex amplitude ``H(u)`` of the Imhof integrand, which is
    ``Re(H(u) * exp(-i*c*u))`` with ``c = (x-m)/2``. ``H`` does not depend on
    ``x`` or ``m`` and varies slowly in ``u``; all the oscillation of the
    integrand is in the factor ``exp(-i*c*u)``. :func:`imhof` uses this split
    to integrate the oscillatory tail with a Fourier-weighted quadrature.

    Writing ``Z = exp(i*theta0)/rho``, with ``theta0`` the x-free part of the
    phase, ``H = q*Z``: ``q = -i/u`` for the cdf (so ``Re(H e^{-icu})`` is
    ``sin(theta)/(u*rho)``), ``1`` for the pdf, the x-derivative factor
    ``(-i*u/2)**nx`` for ``'dens'``, and ``i*ell*(-i*u/2)**nx/u`` for the
    k-derivatives (``Re(i*y) = -Im(y)``).

    ``rho`` is computed in log form, which is faster and cannot overflow
    (``w^2 u^2/(1 + w^2 u^2)`` is written ``1 - 1/q`` so it stays finite)."""
    u = np.float64(u)   # numpy, not Python, float: overflows to inf, not an error
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        wu = w * u
        q = 1 + wu * wu
        theta0 = (np.dot(k, np.arctan(wu)) + np.dot(l, wu / q)) / 2
        log_rho = np.dot(k, np.log(q)) / 4 + np.dot(l, 1 - 1 / q) / 2 + (u * s) ** 2 / 8
        Z = np.exp(1j * theta0 - log_rho)
        if output == "cdf":
            return -1j * Z / u
        elif output == "pdf":
            return Z
        dfac = (-(1j * u / 2)) ** nx               # nx x-derivatives (R2)
        if output == "dens":
            return dfac * Z
        elif output == "k_deriv":
            ell = -0.5 * np.log(1 - 1j * w[idx] * u)
        elif output == "kk_deriv":
            ell = ((-0.5 * np.log(1 - 1j * w[idx[0]] * u))
                   * (-0.5 * np.log(1 - 1j * w[idx[1]] * u)))
        else:
            raise ValueError("unknown output %r" % output)
        return 1j * ell * dfac * Z / u


def _imhof_integrand_mp(u, x, w, k, l, s, m, output, idx=None, nx=0):
    u = mp.mpf(u)
    theta = mp.mpf(0)
    rho = mp.mpf(1)
    for wi, ki, li in zip(w, k, l):
        wi, ki, li = mp.mpf(wi), mp.mpf(ki), mp.mpf(li)
        w2u2 = wi ** 2 * u ** 2
        theta += ki * mp.atan(wi * u) + (li * (wi * u)) / (1 + w2u2)
        rho *= ((1 + w2u2) ** (ki / 4)) * mp.e ** ((w2u2 * li) / (2 * (1 + w2u2)))
    theta = theta / 2 + u * (mp.mpf(m) - mp.mpf(x)) / 2
    rho *= mp.e ** (u ** 2 * mp.mpf(s) ** 2 / 8)
    if output == "cdf":
        return mp.sin(theta) / (u * rho)
    elif output == "pdf":
        return mp.cos(theta) / rho
    else:
        Z = mp.e ** (1j * theta) / rho
        dfac = (-(1j * u / 2)) ** nx
        if output == "dens":
            return mp.re(dfac * Z)
        elif output == "k_deriv":
            ell = -0.5 * mp.log(1 - 1j * mp.mpf(w[idx]) * u)
            return -mp.im(ell * dfac * Z) / u
        elif output == "kk_deriv":
            ell = ((-0.5 * mp.log(1 - 1j * mp.mpf(w[idx[0]]) * u))
                   * (-0.5 * mp.log(1 - 1j * mp.mpf(w[idx[1]]) * u)))
            return -mp.im(ell * dfac * Z) / u
        else:
            raise ValueError("unknown output %r" % output)


def imhof(x, w, k, l, s, m, side="lower", output="cdf",
          idx=None, nx=0, precision="basic", AbsTol=1e-10, RelTol=1e-6):
    """Imhof-Davies method for the cdf/pdf of a generalized chi-square, and
    for the exact (non-finite-differenced) parameter-derivative integrands
    used by ``cdf_grad_gx2``/``cdf_grad_bd`` -- see :func:`imhof_integrand`
    for the ``output`` modes.

    The result is an integral over ``u`` in ``(0, inf)`` of an integrand that
    oscillates with period ``4*pi/|x-m|`` under an amplitude that can decay
    as slowly as a power of ``u`` (e.g. when some weights are tiny). With
    ``precision='basic'``, several ``x`` points are first integrated together
    in one shared adaptive quadrature (``quad_vec``). If that does not
    converge, or there is only one point, each point is integrated
    separately: within ``2 sd`` of ``m`` with plain ``quad`` first, and
    otherwise, or if that fails, split into a head over the first period
    (capped at ``2*pi/sd`` in ``u``) integrated plainly, and a tail integrated
    with QUADPACK's Fourier-integral routine QAWF, which handles slowly
    decaying oscillatory tails. With ``precision='vpa'``, ``mpmath.quadosc``
    plays the role of QAWF, used unless ``x`` is within ``0.01 sd`` of ``m``
    (where plain ``mpmath.quad`` is used). All quadratures run in the
    dimensionless variable ``sd*u``, so results don't depend on the units of
    ``x``.

    Parameters
    ----------
    AbsTol, RelTol : float, optional
        Absolute and relative tolerances on the integral itself, not on the
        returned probability. For the cdf the integral is of order ``pi/2``,
        so the defaults allow an absolute error of up to about ``5e-7`` in
        ``p`` (typically far less); for small tail probabilities, tighten
        them (e.g. ``AbsTol=1e-14, RelTol=1e-12``) for tail-relative
        accuracy. Probabilities below
        about ``1e-14`` are lost to cancellation in ``0.5 -/+ integral/pi``
        at any tolerance; use the 'tail' or 'ray' methods there.

    Returns
    -------
    p : ndarray
        The cdf (``side`` tail), pdf, or derivative output at ``x``.
    errflag : ndarray of bool
        True where ``p`` is unreliable: either the quadrature did not reach
        the requested accuracy (also warned with ``IntegrationWarning``), or
        a probability fell outside [0, 1] and was clipped (warned with
        :class:`ImhofClipWarning`). Not set by ``precision='vpa'``'s
        quadrature, which reports no convergence status.
    """
    w = asrow(w); k = asrow(k); l = asrow(l)
    x = np.asarray(x, dtype=float)
    xf = x.ravel()

    integral = np.empty(xf.size)
    failed = np.zeros(xf.size, dtype=bool)
    # Integrate in the dimensionless variable v = sd*u, in which the
    # integrand's mass lies at v ~ 1 and it oscillates at rate (x-m)/(2*sd).
    # The quadrature routines below have built-in unit scales (QUADPACK's map
    # of (0, inf), QAWF's cycle length, mpmath's tanh-sinh nodes), so without
    # this their accuracy -- and QAWF's error estimate -- would depend on the
    # units of x.
    sd = np.sqrt(stat(w, k, l, s, m)[1]) or 1.0
    cs = (xf - m) / (2 * sd)
    if precision == "basic":
        H = lambda v: _imhof_amplitude(v / sd, w, k, l, s, output, idx, nx) / sd
        converged = False
        if xf.size > 1:
            # Fast path for many x: one adaptive quadrature over (0, inf)
            # shared by all x, so H is computed once per node. It handles
            # most cases, and when it doesn't, it reports so (checked against
            # the per-point route below, it was accurate whenever it
            # converged). Its budget is capped so a failed attempt is cheap.
            r = quad_vec(lambda v: np.real(H(v) * np.exp(-1j * cs * v)),
                         0, np.inf, epsabs=AbsTol, epsrel=RelTol, limit=200,
                         full_output=True)
            integral, converged = r[0], r[2].status == 0
        if not converged:
            for i, ci in enumerate(cs):
                integral[i], failed[i] = _imhof_quad(H, ci, AbsTol, RelTol)
        if np.any(failed):
            warnings.warn("Imhof integral did not reach the requested accuracy "
                          "at %d of %d point(s); these are flagged in errflag."
                          % (failed.sum(), failed.size), IntegrationWarning)
    elif precision == "vpa":
        # Where the integrand oscillates over its mass (|c| not small), plain
        # mp.quad can be far off yet report a small error, so quadosc (which
        # sums the integral period by period and extrapolates the sum) is
        # used. For |c| below ~0.005 (x within ~0.01 sd of m) the periods are
        # too long for quadosc, and plain mp.quad is the more accurate
        # (crossover found empirically).
        for i, (xi, ci) in enumerate(zip(xf, cs)):
            f = lambda v: _imhof_integrand_mp(v / sd, xi, w, k, l, s, m, output, idx, nx) / sd
            if abs(ci) >= 0.005:
                val = mp.quadosc(f, [0, mp.inf], omega=abs(ci))
            else:
                val = mp.quad(f, [0, mp.inf])
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
    elif output == "pdf":
        p = integral / (2 * np.pi)
        errflag = p < 0
    else:
        # signed derivative outputs (no probability clipping):
        #   'dens'      x-derivatives of the pdf, normalized by 1/(2*pi)
        #   'k_deriv'   d/dk of the cdf (and its x-derivatives), normalized by 1/pi
        #   'kk_deriv'  d^2/(dk dk) of the cdf, normalized by 1/pi
        p = integral / (2 * np.pi) if output == "dens" else integral / np.pi
        errflag = np.zeros_like(p, dtype=bool)

    if np.any(errflag):
        warnings.warn("Imhof method output(s) too close to limit to compute "
                      "exactly, so clipping. Check the errflag output, and try "
                      "stricter tolerances.", ImhofClipWarning)
        p = np.maximum(p, 0)
    return p, errflag | failed.reshape(x.shape)


def _imhof_quad(H, c, AbsTol, RelTol):
    """Integral over v in (0, inf) of ``Re(H(v) * exp(-i*c*v))``, in the
    dimensionless variable of :func:`imhof` (integrand mass at ``v ~ 1``),
    and whether QUADPACK failed to reach the requested accuracy.

    With ``c != 0`` the integrand oscillates with period ``2*pi/|c|`` while
    its amplitude ``H`` may decay as slowly as a power of ``v`` (e.g. when
    some weights are tiny, the integrand only stops oscillating at ``v`` of
    order ``sd/min|w|``). A generic quadrature over ``(0, inf)`` can then
    fail. In that case we integrate a head ``(0, v0)`` plainly and the rest
    with QUADPACK's Fourier-integral routine QAWF, which integrates cycle by
    cycle and extrapolates their sum. The tail splits as
    ``Re(H)*cos(|c|v) + sign(c)*Im(H)*sin(|c|v)``. QAWF takes only an
    absolute tolerance, so we give it ``RelTol`` relative to the head.

    The head is one period, but no longer than ``2*pi``, so that it covers
    the integrand's mass even when ``x`` is near ``m`` (tiny ``c``). When
    ``|c| <= 1`` (``x`` within ``2 sd`` of ``m``) the oscillation is slow over
    that mass, the plain quadrature over ``(0, inf)`` usually works and is
    more accurate, so we try it first and split only if it fails. Farther
    out, we split straight away."""
    full = lambda v: np.real(H(v) * np.exp(-1j * c * v))
    if abs(c) <= 1:
        r = quad(full, 0, np.inf, epsabs=AbsTol, epsrel=RelTol, full_output=1)
        if len(r) == 3 or c == 0:
            return r[0], len(r) > 3
    v0 = 2 * np.pi / max(abs(c), 1)
    head = quad(full, 0, v0, epsabs=AbsTol, epsrel=RelTol, full_output=1)
    tol = max(AbsTol, RelTol * abs(head[0]))
    tail_c = quad(lambda v: np.real(H(v)), v0, np.inf, weight="cos", wvar=abs(c),
                  epsabs=tol, full_output=1)
    tail_s = quad(lambda v: np.imag(H(v)), v0, np.inf, weight="sin", wvar=abs(c),
                  epsabs=tol, full_output=1)
    val = head[0] + tail_c[0] + np.sign(c) * tail_s[0]
    return val, any(len(r) > 3 for r in (head, tail_c, tail_s))


# ===========================================================================
# Ruben's series method
# ===========================================================================

def _ruben_coeffs(w, k, l, n_ruben=1000):
    """Ruben's series expansion coefficients. Depends only on ``w, k, l``
    (and the term-count cap ``n_ruben``) -- not on the evaluation point ``x``
    or offset ``m`` -- so callers evaluating many points against the same
    ``(w, k, l)`` (e.g. the mixed-sign convolution's inner quadrature, which
    samples one scalar point per callback) should compute this once and
    reuse it via :func:`_ruben_eval`, rather than recomputing the series for
    every point."""
    w = asrow(w); k = asrow(k); l = asrow(l)

    w_pos = True
    if np.all(w < 0):
        w = -w; w_pos = False

    beta = 0.90625 * np.min(w)
    M = np.sum(k)
    n = np.arange(1, n_ruben).reshape(-1, 1)  # (n_ruben-1, 1)

    g = (np.sum(k * (1 - beta / w) ** n, axis=1)
         + (beta * n.ravel() * ((1 - beta / w) ** (n - 1) @ (l / w))))

    # expansion coefficients, stopping once the leftover series mass is
    # negligible. The a_j are nonnegative and sum to 1, so the tail mass
    # 1-sum(a[:N]) both bounds the truncation error and decreases
    # monotonically. The stop uses only this cheap coefficient recurrence --
    # not the chi-square grid below -- so the term count N is fixed in a
    # single pass, and the expensive evaluation is then done once at that
    # reduced size. n_ruben is the safety cap; most cases converge in ~1e2
    # terms well under it.
    #
    # Either branch below can instead exhaust n_ruben without its stopping
    # criterion ever firing -- a real outcome, not a formal edge case: it
    # happens whenever the smallest |w_j| is small enough that the implied
    # scale beta=0.90625*min|w_j| makes (x-m)/beta huge, which is exactly
    # what a near-rank-deficient quadratic boundary produces just off its
    # exact zero-eigenvalue point. There the series provably still
    # converges, but needs many more terms than are safe to compute here (a
    # single extra order of magnitude in N costs two more in runtime, since
    # the recursion is O(N^2)); confirmed directly on such a case: it
    # doesn't converge or match Imhof until N~2e4, taking >0.5s, versus
    # Imhof's ~0.01s for the same point at any N. So this cap is a genuine
    # efficiency boundary, not just a safety net, and a fixed larger cap
    # would only move the failure to a slightly more extreme case, not
    # remove it. We signal this with NaN (via the python for/else, which
    # runs only when the loop never broke) rather than returning whatever
    # badly-truncated partial sum the loop stopped at -- both of this
    # function's callers already treat a non-finite Ruben output as "fall
    # back to Imhof" (cdf's "auto" method, and dens_deriv's same-sign and
    # mixed-sign routes), so this reuses that existing, already-tested path
    # instead of adding a new failure signal each caller would need to
    # learn about separately.
    masstol = 1e-14
    a = np.full(n_ruben, np.nan)
    a[0] = np.sqrt(np.exp(-np.sum(l)) * beta ** M * np.prod(w ** (-k)))
    if a[0] < REALMIN:
        # The true leading coefficient underflows (e.g. when some
        # non-centrality in l is large, as happens for a quadratic form
        # whose curvature is small relative to its linear part). The a_j
        # are nonnegative and sum to 1, but that overall scale is lost here
        # -- only their relative sizes survive, since the recursion below is
        # linear and homogeneous in a[:j]. Recover the coefficients up to
        # that lost scale (starting from b[0]=1 instead of the
        # unrepresentable true a[0]), then renormalize at the end so they
        # sum to 1, exactly as they must.
        b = np.full(n_ruben, np.nan)
        b[0] = 1.0
        cum = b[0]
        N = n_ruben
        for j in range(1, n_ruben):
            b[j] = np.dot(np.flip(g[:j]), b[:j]) / (2 * j)
            cum += b[j]
            if b[j] < masstol * cum:
                N = j + 1
                break
        else:
            return dict(a=np.full(1, np.nan), N=1, beta=beta, M=M, w_pos=w_pos)
        a = b[:N] / cum
    else:
        cum = a[0]
        N = n_ruben
        for j in range(1, n_ruben):
            a[j] = np.dot(np.flip(g[:j]), a[:j]) / (2 * j)
            cum += a[j]
            if 1 - cum < masstol:
                N = j + 1
                break
        else:
            return dict(a=np.full(1, np.nan), N=1, beta=beta, M=M, w_pos=w_pos)
        a = a[:N]

    return dict(a=a, N=N, beta=beta, M=M, w_pos=w_pos)


def _ruben_eval(coeffs, x, m, side="lower", output="cdf", nx=0):
    """Evaluate Ruben's series at ``x`` (offset ``m``) from coefficients
    already computed by :func:`_ruben_coeffs`."""
    a, N, beta, M, w_pos = (coeffs["a"], coeffs["N"], coeffs["beta"],
                             coeffs["M"], coeffs["w_pos"])
    x = np.asarray(x, dtype=float)
    xf = x.ravel().astype(float).copy()
    if not w_pos:
        xf = -xf; m = -m

    kgrid = (M + 2 * np.arange(N)).reshape(-1, 1)   # (N, 1)
    xgrid = ((xf - m) / beta).reshape(1, -1)              # (1, n_x)

    upper = (w_pos and side == "upper") or ((not w_pos) and side == "lower")
    if output == "cdf":
        if upper:
            F = chi2.sf(xgrid, kgrid)
        else:
            F = chi2.cdf(xgrid, kgrid)
    else:
        F = _chi2pdf_nderiv(xgrid, kgrid, nx)   # nx-th y-derivative of the chi2 density

    p = a @ F  # (n_x,)
    if output == "pdf":
        # each x-derivative brings a factor 1/beta from y=(x-m)/beta; the
        # flipped (all-negative-weight) frame contributes a factor (-1)^nx.
        p = p / beta ** (nx + 1)
        if not w_pos:
            p = p * (-1) ** nx

    # truncation-error indicator: the leftover series mass (now negligible
    # unless the n_ruben cap was hit) times the next central-chi-square factor
    p_err = (1 - np.sum(a)) * chi2.cdf((xf - m) / beta, M + 2 * N)

    return p.reshape(x.shape), p_err.reshape(x.shape)


def ruben(x, w, k, l, m, side="lower", output="cdf", nx=0, n_ruben=1000):
    """Ruben's series. Requires all ``w`` the same sign and ``s == 0``.

    Parameters
    ----------
    nx : int, optional
        x-derivative order of the pdf (0 gives the plain pdf, the default).
        Only defined for ``output='pdf'``.
    n_ruben : int, optional
        Term-count cap. If the series hasn't converged within this many
        terms (checked cheaply from the coefficients alone, before the
        expensive per-``x`` evaluation) -- which happens when the smallest
        ``|w|`` is small enough to need far more terms than are efficient to
        compute here -- this returns NaN rather than a value truncated at an
        arbitrary, unverified point. Callers that dispatch across methods
        (:func:`gx2.cdf`'s ``method='auto'``, and :func:`gx2.dens_deriv`)
        already treat a non-finite Ruben output as "fall back to Imhof".
    """
    if nx and output != "pdf":
        raise ValueError("The x-derivative order 'nx' is only defined for "
                         "the 'pdf' output.")

    w_check = asrow(w)
    if not (np.all(w_check > 0) or np.all(w_check < 0)):
        raise ValueError("Ruben's method needs all w the same sign.")

    coeffs = _ruben_coeffs(w, k, l, n_ruben=n_ruben)
    return _ruben_eval(coeffs, x, m, side=side, output=output, nx=nx)


def _chi2pdf_nderiv(y, nu, n):
    """``n``-th derivative in ``y`` of the central chi-square density
    ``g_nu(y)``. Uses the closed form
    ``g_nu^(n)(y) = g_nu(y) * sum_{j=0}^n C(n,j)(-1/2)^{n-j} (a)_j y^{-j}``,
    where ``a = nu/2-1`` and ``(a)_j`` is the falling factorial. Exact for
    any ``nu>0`` (no negative-dof chi-square ever appears). The derivative
    vanishes on the support edge ``y<=0``."""
    gd = chi2.pdf(y, nu)
    if n == 0:
        return gd
    a = nu / 2 - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        poly = np.zeros_like(gd)
        for j in range(n + 1):
            ff = np.ones_like(gd)
            for ell in range(j):
                ff = ff * (a - ell)
            poly = poly + comb(n, j) * (-0.5) ** (n - j) * ff * (y ** (-float(j)))
        gd = gd * poly
    gd = np.where(y <= 0, 0.0, gd)
    return gd


# ===========================================================================
# IFFT method
# ===========================================================================

def ifft(x, w, k, l, s, m, side="lower", output="cdf",
         span=None, n_grid=int(1e6) + 1, ft_type="cft"):
    """IFFT method. ``x='full'`` returns the cdf/pdf over a spanning grid."""
    w = asrow(w); k = asrow(k); l = asrow(l)
    full = isinstance(x, str) and x.lower() == "full"
    if not full:
        x = np.asarray(x, dtype=float)

    if span is None:
        if full:
            mu, v = stat(w, k, l, s, m)
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
            pdfv = _pdf(xgrid, w[wi], k[wi], l[wi], 0, off)
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
        phi = char(-t, w, k, l, s, m)
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

def pearson(x, w, k, l, s, m, side="lower", output="cdf"):
    """Pearson's 3-moment approximation."""
    w = asrow(w); k = asrow(k); l = asrow(l)
    x = np.asarray(x, dtype=float)

    mu1 = np.sum(w * (k + l)) + m
    mu2 = 2 * np.sum(w ** 2 * (k + 2 * l)) + s ** 2
    mu3 = 8 * np.sum(w ** 3 * (k + 3 * l))
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


# ===========================================================================
# Das's infinite-tail approximation
# ===========================================================================

def tail(x, w, k, l, s, m, side="lower", output="cdf"):
    """Infinite-tail approximation. Returns log10 values where the result
    underflows double precision (those entries are negative)."""
    w = asrow(w); k = asrow(k); l = asrow(l)
    x = np.asarray(x, dtype=float)

    # merge into unique w's
    w, ic = uniquetol(w)
    k = np.array([np.sum(asrow(k)[ic == i]) for i in range(w.size)])
    l_full = asrow(l)
    l = np.array([np.sum(l_full[ic == i]) for i in range(w.size)])

    if side == "upper":
        masked = w * (w > 0)
        max_idx = int(np.argmax(masked))
        w_max = masked[max_idx]
    else:
        masked = w * (w < 0)
        max_idx = int(np.argmin(masked))
        w_max = masked[max_idx]

    k_max = k[max_idx]
    l_max = l[max_idx]

    keep = np.ones(w.size, dtype=bool)
    keep[max_idx] = False
    w_rest = w[keep]
    k_rest = k[keep]
    l_rest = l[keep]

    a = (np.exp(m / (2 * w_max) + s ** 2 / (8 * w_max ** 2))
         * np.prod(np.exp((l_rest * w_rest) / (2 * (w_max - w_rest)))
                   / (1 - w_rest / w_max) ** (k_rest / 2)))

    xf = x.ravel().astype(float)
    if output == "pdf":
        with np.errstate(divide="ignore", invalid="ignore"):
            p = a / abs(w_max) * _ncx2pdf(xf / w_max, k_max, l_max)
        x_tiny = xf[p == 0]
        if l_max:
            p_tiny = (np.log10(a) - np.log10(abs(w_max)) - np.log10(2 * np.sqrt(2 * np.pi))
                      + (k_max - 3) / 4 * np.log10(x_tiny / w_max)
                      - (k_max - 1) / 4 * np.log10(l_max)
                      + (np.sqrt(l_max * x_tiny / w_max) - (l_max + x_tiny / w_max) / 2) / np.log(10))
        else:
            p_tiny = (np.log10(a) - np.log10(abs(w_max)) - k_max / 2 * np.log10(2)
                      - np.log10(_gamma(k_max / 2)) + (k_max / 2 - 1) * np.log10(x_tiny / w_max)
                      - x_tiny / (2 * w_max * np.log(10)))
    else:  # cdf
        with np.errstate(divide="ignore", invalid="ignore"):
            p = a * _ncx2cdf(xf / w_max, k_max, l_max, upper=True)
        x_tiny = xf[p == 0]
        if l_max:
            p_tiny = (np.log10(a) - np.log10(l_max ** ((k_max - 1) / 4) * np.sqrt(2 * np.pi))
                      - (np.sqrt(x_tiny / w_max) - np.sqrt(l_max)) ** 2 / (2 * np.log(10))
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

def ellipse(x, w, r, l, m, side="lower", output="cdf", x_scale="linear"):
    """Ellipse approximation near the finite tail. Requires all ``w`` the same
    sign and ``s == 0``. With ``x_scale='log'`` the inputs and outputs are
    log10 values."""
    w = asrow(w); r = asrow(r); l = asrow(l)
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
         for li, ki in zip(l, r)])
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
            p_rel_err = x_eff / (2 * np.min(ellipse_weights))
        else:
            p_rel_err = log10_x - np.log10(2 * np.min(ellipse_weights))

    return np.asarray(p).reshape(x.shape), np.asarray(p_rel_err).reshape(x.shape)
