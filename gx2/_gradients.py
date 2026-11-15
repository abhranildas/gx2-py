"""Exact (non-finite-differenced) gradient and Hessian of the generalized
chi-square cdf, with respect to its native parameters and to the quadratic
boundary coefficients of a normal classifier. Mirrors ``cdf_grad_gx2.m`` and
``cdf_grad_bd.m``.
"""

import numpy as np
from scipy.integrate import quad_vec

from ._helpers import asrow
from ._convert import norm_quad_to_gx2_params
from ._distribution import cdf, pdf
from ._methods import imhof
from ._basic import char, stat
from ._dens_deriv import dens_deriv

_NATIVE_GROUPS = ("w", "k", "l", "s", "m")


def cdf_grad_gx2(x, w, k, l, s, m, wrt=None, hess=False,
             AbsTol=1e-10, RelTol=1e-6, precision="basic"):
    """Gradient (and optionally the Hessian) of the cdf of a generalized
    chi-square distribution with respect to its parameters ``w, k, l, s, m``.
    These are computed exactly, with no finite differencing.

    Parameters
    ----------
    x : array_like
        Point(s) at which to evaluate the gradient/Hessian of the cdf.
    w, k, l : array_like
        Weights, degrees of freedom and non-centralities of the non-central
        chi-square terms.
    s : float
        Scale of the normal term.
    m : float
        Offset.
    wrt : sequence of str, optional
        Which parameter groups to differentiate with respect to, drawn from
        ``{'w', 'k', 'l', 's', 'm'}`` (``'l'`` names the non-centrality
        group, since ``lambda`` is a Python keyword). Default is all of
        them. Only the requested groups are returned, in the canonical
        order below with the unrequested groups omitted; the Hessian is the
        corresponding principal submatrix.
    hess : bool, optional
        If True, also return the Hessian (see Returns). Default False.
    AbsTol, RelTol : float, optional
        Error tolerances for the underlying integrals.
    precision : {'basic', 'vpa'}
        ``'basic'`` (default) uses double precision.

    Returns
    -------
    grad : ndarray
        Gradient of the cdf, shape ``[P, numel(x)]``, where ``P`` is the
        number of requested parameters. The rows are stacked in the
        canonical order::

            [ dF/dw_1 ... dF/dw_n,
              dF/dk_1 ... dF/dk_n,
              dF/dl_1 ... dF/dl_n,
              dF/ds,
              dF/dm ]

        where ``n = len(w)``, for a total length ``3n+2`` when all groups
        are requested. When ``wrt`` omits a group, its rows are dropped and
        the remaining rows keep this relative order.
    hessian : ndarray, optional
        Only returned when ``hess`` is True. The symmetric matrix of second
        derivatives ``d^2F/(da db)`` over the same parameters and in the
        same canonical order as ``grad``. Shape ``[P, P]`` for a scalar
        ``x``, else ``[P, P, numel(x)]``.
    """
    w = asrow(w); k = asrow(k); l = asrow(l)
    if wrt is None:
        wrt = _NATIVE_GROUPS
    wanted = lambda g: g in wrt

    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n_pts = x.size
    n = w.size

    opts = dict(AbsTol=AbsTol, RelTol=RelTol, precision=precision)
    imhof_opts = dict(AbsTol=AbsTol, RelTol=RelTol, precision=precision)

    # ---------------------------- gradient ----------------------------
    F = cdf(x, w, k, l, s, m, **opts) if wanted("l") else None
    f = dens_deriv(x, w, k, l, s, m, 0, **opts) if wanted("m") else None

    gw = None
    if wanted("w"):
        gw = np.zeros((n, n_pts))
        for j in range(n):
            kp2 = k.copy(); kp2[j] += 2
            kp4 = k.copy(); kp4[j] += 4
            f2 = dens_deriv(x, w, kp2, l, s, m, 0, **opts)
            f4 = dens_deriv(x, w, kp4, l, s, m, 0, **opts)
            gw[j, :] = -k[j] * f2 - l[j] * f4

    gk = None
    if wanted("k"):
        gk = np.zeros((n, n_pts))
        for j in range(n):
            gk[j, :] = imhof(x, w, k, l, s, m, output="k_deriv", idx=j, **imhof_opts)[0]

    gl = None
    if wanted("l"):
        gl = np.zeros((n, n_pts))
        for j in range(n):
            kp2 = k.copy(); kp2[j] += 2
            F2 = cdf(x, w, kp2, l, s, m, **opts)
            gl[j, :] = 0.5 * (F2 - F)

    gs = None
    if wanted("s"):
        if s == 0:
            gs = np.zeros(n_pts)
        else:
            fprime = dens_deriv(x, w, k, l, s, m, 1, **imhof_opts)
            gs = s * fprime

    gm = -f if wanted("m") else None

    rows = []; sel = []
    if wanted("w"): rows.append(gw); sel += list(range(0, n))
    if wanted("k"): rows.append(gk); sel += list(range(n, 2 * n))
    if wanted("l"): rows.append(gl); sel += list(range(2 * n, 3 * n))
    if wanted("s"): rows.append(gs.reshape(1, -1)); sel.append(3 * n)
    if wanted("m"): rows.append(gm.reshape(1, -1)); sel.append(3 * n + 1)
    grad = np.vstack(rows) if rows else np.zeros((0, n_pts))

    if not hess:
        return grad

    # ---------------------------- Hessian ----------------------------
    P0 = 3 * n + 2
    H = np.zeros((P0, P0, n_pts))
    IW = lambda j: j
    IK = lambda j: n + j
    IL = lambda j: 2 * n + j
    IS = 3 * n
    IM = 3 * n + 1

    def sh(kk, j, d):
        kk2 = kk.copy(); kk2[j] += d; return kk2

    def put(a, b, val):
        H[a, b, :] = val
        if a != b:
            H[b, a, :] = val

    Fh = lambda kk: cdf(x, w, kk, l, s, m, **opts)
    fh = lambda kk: dens_deriv(x, w, kk, l, s, m, 0, **opts)
    fp = lambda kk: dens_deriv(x, w, kk, l, s, m, 1, **imhof_opts)
    fpp = lambda kk: dens_deriv(x, w, kk, l, s, m, 2, **imhof_opts)
    fppp = lambda kk: dens_deriv(x, w, kk, l, s, m, 3, **imhof_opts)
    dkF = lambda kk, j: imhof(x, w, kk, l, s, m, output="k_deriv", idx=j, nx=0, **imhof_opts)[0]
    dkf = lambda kk, j: imhof(x, w, kk, l, s, m, output="k_deriv", idx=j, nx=1, **imhof_opts)[0]
    dkFxx = lambda kk, j: imhof(x, w, kk, l, s, m, output="k_deriv", idx=j, nx=2, **imhof_opts)[0]
    dkkF = lambda kk, i, j: imhof(x, w, kk, l, s, m, output="kk_deriv", idx=[i, j], nx=0, **imhof_opts)[0]

    F0 = Fh(k); f0 = fh(k); fp0 = fp(k)

    z = np.zeros(n_pts)
    put(IM, IM, fp0)                                     # H_mm = f'
    if s == 0:
        put(IS, IM, z)                                   # H_ms = -s f'' = 0
        put(IS, IS, fp0)                                 # H_ss = f' + s^2 f''' = f'
    else:
        put(IS, IM, -s * fpp(k))                          # H_ms = -s f''
        put(IS, IS, fp0 + s ** 2 * fppp(k))               # H_ss = f' + s^2 f'''

    for j in range(n):
        kj = k[j]; lj = l[j]
        kp2 = sh(k, j, 2); kp4 = sh(k, j, 4); kp6 = sh(k, j, 6); kp8 = sh(k, j, 8)
        put(IM, IL(j), -0.5 * (fh(kp2) - f0))                                            # H_m,l_j
        put(IM, IW(j), kj * fp(kp2) + lj * fp(kp4))                                      # H_m,w_j
        put(IL(j), IL(j), 0.25 * (Fh(kp4) - 2 * Fh(kp2) + F0))                           # H_l_j,l_j
        put(IL(j), IW(j), 0.5 * kj * fh(kp2) + 0.5 * (lj - kj - 2) * fh(kp4) - 0.5 * lj * fh(kp6))  # H_l_j,w_j
        put(IW(j), IW(j), kj * (kj + 2) * fp(kp4) + 2 * lj * (kj + 2) * fp(kp6) + lj ** 2 * fp(kp8))  # H_w_j,w_j
        put(IM, IK(j), -dkf(k, j))                                                       # H_m,k_j
        put(IL(j), IK(j), 0.5 * (dkF(kp2, j) - dkF(k, j)))                               # H_l_j,k_j
        put(IW(j), IK(j), -fh(kp2) - kj * dkf(kp2, j) - lj * dkf(kp4, j))                # H_w_j,k_j
        put(IK(j), IK(j), dkkF(k, j, j))                                                 # H_k_j,k_j
        if s == 0:
            put(IS, IL(j), z); put(IS, IW(j), z); put(IS, IK(j), z)
        else:
            put(IS, IL(j), 0.5 * s * (fp(kp2) - fp0))                                   # H_s,l_j
            put(IS, IW(j), -s * (kj * fpp(kp2) + lj * fpp(kp4)))                        # H_s,w_j
            put(IS, IK(j), s * dkFxx(k, j))                                             # H_s,k_j

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ki = k[i]; li = l[i]; kj = k[j]; lj = l[j]
            kip2 = sh(k, i, 2); kip4 = sh(k, i, 4); kjp2 = sh(k, j, 2); kjp4 = sh(k, j, 4)
            put(IL(i), IW(j), -0.5 * (kj * (fh(sh(kip2, j, 2)) - fh(kjp2))
                                      + lj * (fh(sh(kip2, j, 4)) - fh(kjp4))))            # H_l_i,w_j
            put(IL(i), IK(j), 0.5 * (dkF(kip2, j) - dkF(k, j)))                          # H_l_i,k_j
            put(IW(i), IK(j), -ki * dkf(kip2, j) - li * dkf(kip4, j))                    # H_w_i,k_j
            if i < j:
                put(IL(i), IL(j), 0.25 * (Fh(sh(kip2, j, 2)) - Fh(kip2) - Fh(kjp2) + F0))  # H_l_i,l_j
                put(IW(i), IW(j), ki * kj * fp(sh(kip2, j, 2)) + ki * lj * fp(sh(kip2, j, 4))
                                  + li * kj * fp(sh(kip4, j, 2)) + li * lj * fp(sh(kip4, j, 4)))  # H_w_i,w_j
                put(IK(i), IK(j), dkkF(k, i, j))                                          # H_k_i,k_j

    sel = np.array(sel, dtype=int)
    hessian = H[sel][:, sel, :]
    if n_pts == 1:
        hessian = hessian[:, :, 0]
    return grad, hessian


_BOUNDARY_GROUPS = ("q2", "q1", "q0")

# Relative tolerance for treating a threshold x0 as sitting exactly on the
# mixed-sign density cusp x0=m (see _resolve_cusp_terms below and
# gx2_derivatives.md open items 3.3/3.7(a)). Deliberately tiny: meant to
# catch a boundary landing on the cusp by floating-point-exact construction
# (e.g. an antipodal-mean, swapped-covariance classification boundary), not
# merely "close" boundaries, which the ordinary s=0 route already handles
# correctly.
_CUSP_REL_TOL = 1e-8

# Shrinking artificial normal-term probes used to test convergence at the
# cusp (see _resolve_cusp_terms). Chosen from direct testing on the term
# that first exposed this (problem 6 of the wider benchmark): a genuinely
# divergent term shows a clean, unmistakable, non-shrinking decade-over-
# decade change throughout this whole range (and well beyond it); a
# genuinely finite term converges within the first one or two decades of it.
# Below roughly 1e-8*scale the inversion route itself becomes ill-conditioned
# (checked directly) regardless of convergence, so this stays well clear of
# that.
_CUSP_PROBE_FACTORS = (1e-2, 1e-3, 1e-4)


# Absolute floor (relative to cusp_scale) below which a probe sequence is
# trusted as converged regardless of the relative decade-over-decade test
# below. That test only looks at *relative* change, which is ill-conditioned
# when a component's true value is exactly (or extremely close to) zero --
# which happens whenever the boundary passed in is itself already the
# Bayes-optimal one, since the classification error's gradient vanishes
# there identically (confirmed directly: gx2_derivatives.md's problem 9
# uses a boundary that reduces to the true optimum, and its Q2 off-diagonal
# and q0 gradient entries are pure floating-point noise at every probe
# scale, down to 1e-17 -- yet without this floor that noise occasionally
# mimics a genuine divergence's signature and gets stamped +-inf instead of
# ~0). A real divergence is already well clear of this floor at even the
# largest (least-refined) probe (see the comment on _CUSP_PROBE_FACTORS),
# so this floor can only rescue a false positive, never mask a real one.
_CUSP_ZERO_FLOOR = 1e-6


def _cusp_probe_merge(vals, cusp_scale):
    """vals: 3 arrays (same shape), evaluated at a shrinking sequence of
    tiny artificial normal-term probes s (largest first). Elementwise: if
    the value has converged (its most recent decade-over-decade change
    shrank to well under the one before, or all three probes already sit
    below the noise floor ``_CUSP_ZERO_FLOOR*cusp_scale``), trust the
    smallest-s value; otherwise the term is genuinely unbounded (confirmed
    directly for the term that exposed this: its damped value grew by an
    almost exactly constant amount every time the probe s shrank by 10x,
    over six full decades -- the signature of an unbounded logarithmic
    divergence, not a slow-converging finite limit) and the correctly-signed
    infinity is returned instead of an arbitrary finite number.
    """
    v0, v1, v2 = vals
    d1, d2 = v1 - v0, v2 - v1
    floor = _CUSP_ZERO_FLOOR * cusp_scale
    with np.errstate(invalid="ignore"):
        converging = (d1 == 0) | (np.abs(d2) < 0.5 * np.abs(d1))
        all_below_floor = (np.abs(v0) < floor) & (np.abs(v1) < floor) & (np.abs(v2) < floor)
        converging = converging | all_below_floor
    return np.where(converging, v2, np.where(d2 > 0, np.inf, -np.inf))


def _wynn_epsilon(S):
    """Wynn's epsilon algorithm on the partial sums ``S`` along axis 0,
    elementwise over the remaining axes: the extrapolated limit, taken as the
    last entry of the even column of the epsilon table that changed least
    from the previous even column (column 0 counting with its own last
    change). Same as ``wynn_epsilon`` in ``gx2_osc_integral.m``."""
    prev = np.zeros((S.shape[0] + 1,) + S.shape[1:])     # column -1
    cur = S                                              # column 0
    best = S[-1]
    best_change = np.abs(S[-1] - S[max(S.shape[0] - 2, 0)])
    last_even = S[-1]
    for col in range(1, S.shape[0]):
        with np.errstate(divide="ignore", invalid="ignore"):
            nxt = prev[1:cur.shape[0]] + 1.0 / np.diff(cur, axis=0)
        if not np.isfinite(nxt).any():
            break        # every sequence already converged exactly
        prev, cur = cur, nxt
        if col % 2 == 0:
            change = np.abs(cur[-1] - last_even)
            better = np.isfinite(change) & (change < best_change)
            best = np.where(better, cur[-1], best)
            best_change = np.where(better, change, best_change)
            last_even = np.where(np.isfinite(cur[-1]), cur[-1], last_even)
    return best


def _osc_integral(amp, xm, sd, AbsTol, RelTol, n_cycles=50):
    """Integrals over ``t`` in ``(0, inf)`` of ``Re(amp(t) * exp(-1j*t*xm))``,
    for every entry of the complex vector ``amp(t)`` and every offset
    ``xm = x - m``; shape ``(numel(amp), numel(xm))``.

    This is the Fourier-inversion integral of :func:`imhof` with a
    vector-valued amplitude, and it is handled the same way. The amplitude
    varies slowly, but at ``s = 0`` it decays only like a power of ``t``, so
    the integrand keeps oscillating far out. Several points are first
    integrated together in one plain quadrature over ``(0, inf)``. If that
    does not converge, or there is one point, each point is integrated on
    its own: within ``2 sd`` of ``m`` plainly first, and otherwise, or if
    that fails, as a head of one period (at most ``2*pi/sd`` in ``t``)
    integrated plainly, plus a tail summed cycle by cycle, with the partial
    sums of the alternating cycle integrals extrapolated by Wynn's epsilon
    algorithm, as QUADPACK's QAWF does (QAWF itself is scalar-only). A cycle
    is ``2*floor(|c|)+1`` half-periods, so about ``2*pi`` long for
    ``|c| >= 1``, and all cycles go into one vectorized quadrature.
    Everything runs in the dimensionless variable ``v = sd*t``, so results
    do not depend on the units of ``x``.
    """
    om = np.atleast_1d(xm) / sd                 # frequency in v, per point

    def f(v, o):
        return np.real(amp(v / sd)[:, None] * np.exp(-1j * v * o[None, :])) / sd

    def plain(o):
        r = quad_vec(lambda v: f(v, o), 0, np.inf, epsabs=AbsTol, epsrel=RelTol,
                     limit=200, full_output=True)
        return r[0], r[2].status == 0

    if om.size > 1:
        out, ok = plain(om)
        if ok:
            return out
    else:
        out = np.empty((amp(0.0).size, 1))
    for i, o in enumerate(om):
        oi = om[i:i + 1]
        if abs(o) <= 1:
            val, ok = plain(oi)
            if ok or o == 0:
                out[:, i] = val[:, 0]
                continue
        v0 = 2 * np.pi / max(abs(o), 1)
        L = (2 * np.floor(abs(o)) + 1) * np.pi / abs(o)
        head = quad_vec(lambda v: f(v, oi)[:, 0], 0, v0, epsabs=AbsTol, epsrel=RelTol)[0]
        cycles = quad_vec(
            lambda u: np.stack([f(v0 + (j + u) * L, oi)[:, 0] for j in range(n_cycles)]) * L,
            0, 1, epsabs=AbsTol / n_cycles, epsrel=RelTol)[0]
        out[:, i] = head + _wynn_epsilon(np.cumsum(cycles, axis=0))
    return out


def _boundary_raw(x, mu, v, quad, wrt, hess, AbsTol=1e-10, RelTol=1e-6,
                   precision="basic"):
    """Shared machinery behind :func:`cdf_grad_bd` and
    :func:`norm_err_grad_bd`: one class's raw (pre-cusp-resolution,
    pre-squeeze, trailing-``nx``) gradient/Hessian blocks for
    ``F(x0)=P(q(x)<=x0)``, plus everything needed to resolve a cusp there --
    the ``at_cusp`` mask, its scale, and a probe callable
    ``inversion_grad_hess(s_eff, want_hess)`` giving the grad/Hessian with
    the normal term replaced by an arbitrary ``s_eff``.
    :func:`cdf_grad_bd` resolves the cusp on this class alone;
    :func:`norm_err_grad_bd` instead combines two classes' raw output and
    inversion probes *before* resolving, since resolving each class's
    infinity independently and then subtracting can throw away a genuine
    ``inf - inf`` cancellation in the combined error.

    Returns
    -------
    raw_grad, raw_hess, at_cusp, cusp_scale, inversion_grad_hess, nx
    """
    if precision != "basic":
        raise NotImplementedError(
            "cdf_grad_bd only supports precision='basic'; the 'vpa' "
            "path from cdf_grad_bd.m has not been ported.")

    if wrt is None:
        wrt = _BOUNDARY_GROUPS
    wanted = lambda g: g in wrt

    mu = np.asarray(mu, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    q1c = np.asarray(quad["q1"], dtype=float).ravel()
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    nx = x.size
    dim = mu.size

    # convert to gx2 params, and reuse the standardized-quadratic
    # eigen-structure (S=Sigma^{1/2}, V, and the full eigenvalues of S*Q2*S)
    # for the per-node M^{-1}(t)=S*V*diag(1/(1-2i*t*d_j))*V'*S -- no d-by-d
    # inverse per node.
    w, k, l, s, m, aux = norm_quad_to_gx2_params(mu, v, quad, return_aux=True)
    S = aux["S"]; V = aux["V"]; dvals = aux["d"].ravel()
    SigInv_mu = np.linalg.solve(v, mu)
    opts = dict(AbsTol=AbsTol, RelTol=RelTol, precision=precision)

    # At s=0 with mixed-sign weights, the density has a cusp at x0=m (e.g. a
    # logarithmic divergence for two k=1 terms of opposite sign), and some
    # derivative blocks are genuinely infinite there. A boundary built from
    # exact symmetry (e.g. an antipodal-mean, swapped-covariance
    # classification boundary) lands exactly on it, not just in principle.
    # Detected once here; resolved via _resolve_cusp_terms, which compares
    # the inversion at a shrinking sequence of tiny artificial normal terms
    # rather than trusting the s=0 integral right at that point.
    mixed_sign = not (bool(np.all(w > 0)) or bool(np.all(w < 0)))
    cusp_scale = max(1.0, float(np.sum(np.abs(w) * k)))
    at_cusp = (s == 0) & mixed_sign & (np.abs(x - m) <= _CUSP_REL_TOL * cusp_scale)

    want_q12 = wanted("q1") or wanted("q2")

    def inversion_grad_hess(s_eff, want_hess):
        # Every block is one Fourier-inversion integral of the form
        # Re(amplitude * exp(-i t (x0-m))), the amplitude being built from
        # the weights M^{-1}(t) and mu_tilde(t) times the characteristic
        # function (with its offset m moved into the kernel). The gradient
        # blocks are density-type integrals (the (it) of the master formula
        # cancels the 1/t of the inversion); the Hessian blocks keep one
        # factor of t, giving t*Im(z) = Re(-i*t*z).
        def weights(t):
            phi = complex(np.asarray(char(t, w, k, l, s_eff, 0)))
            g = 1.0 / (1 - 2j * t * dvals)               # (dim,)
            Minv = S @ (V * g) @ V.T @ S
            mut = Minv @ (SigInv_mu + 1j * t * q1c)       # mu_tilde(t), (dim,)
            return Minv, mut, phi

        def grad_amp(t):
            Minv, mut, phi = weights(t)
            P = Minv + np.outer(mut, mut)
            return (np.concatenate([mut[:, None], P], axis=1) * phi).ravel()

        def hess_amp(t):
            Minv, mut, phi = weights(t)
            P = Minv + np.outer(mut, mut)
            Pmm = np.outer(mut, mut)
            Wq1q2 = 2 * np.einsum("ab,c->abc", Minv, mut) + np.einsum("a,bc->abc", mut, P)
            WQ2 = (2 * (np.einsum("qr,sp->pqrs", Minv, Minv)
                        + np.einsum("qr,sp->pqrs", Minv, Pmm)
                        + np.einsum("qr,sp->pqrs", Pmm, Minv))
                   + np.einsum("pq,rs->pqrs", P, P))
            Wflat = np.concatenate([np.array([1.0 + 0j]), mut.ravel(), P.ravel(), P.ravel(),
                                    Wq1q2.ravel(), WQ2.ravel()])
            return -1j * t * Wflat * phi

        sd = float(np.sqrt(stat(w, k, l, s_eff, m)[1])) or 1.0
        # dF/dq0 = -f(x0), since q0 shifts q rigidly
        g = dict(q0=-np.asarray(pdf(x, w, k, l, s_eff, m, **opts)).reshape(nx))
        if want_q12:
            A = -_osc_integral(grad_amp, x - m, sd, AbsTol, RelTol).reshape(dim, dim + 1, nx) / np.pi
            g["q1"] = A[:, 0, :]
            g["q2"] = 0.5 * (A[:, 1:, :] + np.swapaxes(A[:, 1:, :], 0, 1))
        if not want_hess:
            return g, None
        Hraw = _osc_integral(hess_amp, x - m, sd, AbsTol, RelTol) / np.pi
        off = 0
        q0q0 = Hraw[0, :]; off = 1
        q0q1 = Hraw[off:off + dim, :]; off += dim
        q0q2 = Hraw[off:off + dim ** 2, :].reshape(dim, dim, nx); off += dim ** 2
        q1q1 = Hraw[off:off + dim ** 2, :].reshape(dim, dim, nx); off += dim ** 2
        q1q2 = Hraw[off:off + dim ** 3, :].reshape(dim, dim, dim, nx); off += dim ** 3
        q2q2 = Hraw[off:off + dim ** 4, :].reshape(dim, dim, dim, dim, nx)
        q0q2 = 0.5 * (q0q2 + np.swapaxes(q0q2, 0, 1))
        q1q1 = 0.5 * (q1q1 + np.swapaxes(q1q1, 0, 1))
        return g, dict(q0q0=q0q0, q0q1=q0q1, q0q2=q0q2, q1q1=q1q1, q1q2=q1q2, q2q2=q2q2)

    g, raw_hess = inversion_grad_hess(s, hess)
    raw_grad = {key: g[key] for key in ("q0", "q1", "q2") if wanted(key)}
    return raw_grad, raw_hess, at_cusp, cusp_scale, inversion_grad_hess, nx


def _resolve_cusp_terms(raw_grad, raw_hess, at_cusp, inversion_grad_hess, cusp_scale):
    """Override only the points where ``at_cusp`` is True (broadcasting
    against each block's trailing ``nx`` axis) in ``raw_grad``/``raw_hess``,
    leaving every other point -- and every problem that never hits this
    measure-adjacent exact coincidence -- completely untouched.

    ``inversion_grad_hess(s_eff, want_hess)`` is the probe: the (always
    valid, cusp-independent) ``s!=0`` inversion route at a shrinking
    sequence of artificial normal terms, in the same raw shape as
    ``raw_grad``/``raw_hess``. :func:`cdf_grad_bd` passes a
    single class's own probe; :func:`norm_err_grad_bd` passes the *combined*
    ``p1*F1-p0*F0`` probe instead, so that a real cancellation between the
    two classes' divergences shows up as convergence here rather than being
    lost by resolving each class's infinity independently beforehand.
    """
    probes_g = []
    probes_h = []
    for factor in _CUSP_PROBE_FACTORS:
        g_p, h_p = inversion_grad_hess(factor * cusp_scale, raw_hess is not None)
        probes_g.append(g_p)
        if raw_hess is not None:
            probes_h.append(h_p)
    resolved_grad = dict(raw_grad)
    for key in raw_grad:
        merged = _cusp_probe_merge([p[key] for p in probes_g], cusp_scale)
        resolved_grad[key] = np.where(at_cusp, merged, raw_grad[key])
    if raw_hess is None:
        return resolved_grad, None
    resolved_hess = dict(raw_hess)
    for key in raw_hess:
        merged = _cusp_probe_merge([p[key] for p in probes_h], cusp_scale)
        resolved_hess[key] = np.where(at_cusp, merged, raw_hess[key])
    return resolved_grad, resolved_hess


def _finalize_grad_hess(raw_grad, raw_hess, hess, nx):
    """Squeeze raw (trailing-``nx``) grad/Hessian dicts into the public
    ``cdf_grad_bd``/``norm_err_grad_bd`` return shapes."""
    grad = {}
    if "q0" in raw_grad:
        g0 = raw_grad["q0"]
        grad["q0"] = g0 if nx > 1 else float(g0[0])
    if "q1" in raw_grad:
        Gq1 = raw_grad["q1"]
        grad["q1"] = Gq1 if nx > 1 else Gq1[:, 0]
    if "q2" in raw_grad:
        Gq2 = raw_grad["q2"]
        grad["q2"] = Gq2 if nx > 1 else Gq2[:, :, 0]

    if not hess:
        return grad

    hessian = raw_hess
    if nx == 1:
        hessian = {key: np.asarray(val)[..., 0] for key, val in hessian.items()}
    return grad, hessian


def cdf_grad_bd(x, mu, v, quad, wrt=None, hess=False,
                    AbsTol=1e-10, RelTol=1e-6, precision="basic"):
    """Gradient (and optionally the Hessian) of the cdf of a quadratic form
    ``q(x) = x' q2 x + q1' x + q0`` of a normal vector ``x ~ N(mu, v)``, with
    respect to the quadratic's coefficients ``q2, q1, q0`` (holding ``mu``
    and ``v`` fixed).

    ``F(x0) = P(q(x) <= x0)`` is the probability content of the normal in the
    quadratic region ``q(x) <= x0``. This returns its derivatives with
    respect to ``q2``, ``q1`` and ``q0``, computed exactly (no finite
    differencing).

    Parameters
    ----------
    x : array_like
        Threshold(s) ``x0`` at which to evaluate the gradient/Hessian.
    mu : array_like
        Normal mean.
    v : array_like
        Normal covariance matrix.
    quad : dict
        ``{'q2': matrix, 'q1': vector, 'q0': scalar}`` (symmetrized
        internally).
    wrt : sequence of str, optional
        Which coefficient groups to differentiate with respect to, drawn
        from ``{'q2', 'q1', 'q0'}``. Default is all.
    hess : bool, optional
        If True, also return the Hessian (see Returns). Default False.
    AbsTol, RelTol : float, optional
        Error tolerances for the underlying integrals.
    precision : {'basic'}
        Only ``'basic'`` (double precision) is supported; ``'vpa'`` is not
        yet ported (see the module notes).

    Returns
    -------
    grad : dict
        Dict mirroring ``quad``, holding the cdf gradient:
        ``q2`` a symmetric ``d x d`` matrix ``G = dF/dQ2``, in the sense
        ``dF ~ trace(G @ dQ2)`` for symmetric perturbations ``dQ2`` (so a
        lone off-diagonal ``(Q2)_ab`` sees ``2*G_ab``); ``q1`` a ``d``-vector
        ``dF/dq1``; ``q0`` a scalar ``dF/dq0 = -pdf`` of ``q(x)`` at the
        threshold. For an array ``x`` each field carries a trailing
        dimension over the points. Groups omitted by ``wrt`` are absent.
    hessian : dict, optional
        Only returned when ``hess`` is True: the six blocks ``q0q0``,
        ``q0q1``, ``q0q2``, ``q1q1``, ``q1q2`` (a 3-tensor), and ``q2q2``
        (a 4-tensor), each carrying a trailing dimension per threshold. The
        tensor blocks follow the same symmetric (vech) convention as the
        gradient: contracting with symmetric perturbations gives the
        directional second derivatives.

    See Also
    --------
    cdf_grad_gx2, norm_quad_to_gx2_params
    """
    raw_grad, raw_hess, at_cusp, cusp_scale, inversion_grad_hess, nx = _boundary_raw(
        x, mu, v, quad, wrt, hess, AbsTol, RelTol, precision)
    if np.any(at_cusp):
        raw_grad, raw_hess = _resolve_cusp_terms(raw_grad, raw_hess, at_cusp, inversion_grad_hess, cusp_scale)
    return _finalize_grad_hess(raw_grad, raw_hess, hess, nx)


def norm_err(mu0, v0, mu1, v1, quad, p0=0.5, p1=0.5, wrt=None, grad=False, hess=False, **kwargs):
    """Total two-class classification error for a quadratic boundary shared
    between two normal classes, with its gradient and Hessian with respect
    to the boundary coefficients available as optional additional outputs.

    The error is ``E = p0*P(q(x)<=0 | x~N(mu0,v0)) + p1*P(q(x)>0 | x~N(mu1,v1))``,
    i.e. class 0 mass on the class-1 side of the boundary plus class 1 mass
    on the class-0 side; since ``P(q(x)>0)=1-F(0)``, this is
    ``p0*F0(0) + p1*(1-F1(0))`` for the two classes' cdfs of ``q(x)``. The
    gradient/Hessian, when requested, are the analogous signed combination
    of the two classes' :func:`cdf_grad_bd` outputs at ``x0=0``:
    ``dE = p0*dF0 - p1*dF1``, and likewise block by block for the Hessian.
    (This is the mirror image of the convention used before
    :func:`norm_class_opt_bd` replaced ``opt_norm_quad_bd``: ``q(x)>0`` now
    favors class 0 rather than class 1, to match ``IntClassNorm``'s
    ``norm_class_opt_bd.m`` directly instead of with its two classes
    reversed -- see that function's docstring.)

    If the shared boundary lands exactly on one class's density cusp (see
    :func:`cdf_grad_bd`), it generically lands on both classes' cusps
    at once (the two ``F``'s share the same boundary coefficients), and
    resolving each class's divergence independently before subtracting can
    turn a genuine ``inf - inf`` cancellation into ``nan`` -- the two
    classes' divergences may be mirror images that cancel in the combined
    error even though neither class's own Hessian is finite there. So this
    cusp is resolved on the *combined* gradient/Hessian directly: the same
    shrinking-probe test as :func:`cdf_grad_bd`, applied to
    ``p0*F0 - p1*F1`` as a whole rather than to each class alone. (The error
    value ``E`` itself is a plain cdf evaluation and never hits this cusp --
    only its derivatives can diverge there.)

    Parameters
    ----------
    mu0, v0 : array_like
        Mean and covariance of class 0.
    mu1, v1 : array_like
        Mean and covariance of class 1.
    quad : dict
        Boundary coefficients ``{'q2': matrix, 'q1': vector, 'q0': scalar}``,
        shared between the two classes (e.g. the optimal quadratic
        discriminant between them).
    p0, p1 : float, optional
        Class priors (weights on each class's error term). Default 0.5 each.
    grad : bool, optional
        If True, also return the gradient of ``E`` (see Returns). Default
        False.
    hess : bool, optional
        If True, also return the gradient and the Hessian of ``E`` (the
        Hessian is never computed without the gradient, since the gradient
        falls out of the same computation for free). Default False.
    wrt, **kwargs
        Forwarded to :func:`cdf_grad_bd` for each class; only used
        when ``grad`` or ``hess`` is True.

    Returns
    -------
    err : float
        The classification error ``E``. Always returned.
    grad : dict, optional
        Only if ``grad=True`` or ``hess=True``. Gradient of ``E``, same keys
        as :func:`cdf_grad_bd`'s gradient.
    hessian : dict, optional
        Only if ``hess=True``. Same keys as :func:`cdf_grad_bd`'s Hessian.

    See Also
    --------
    cdf_grad_bd
    """
    w0, k0, l0, s0, m0 = norm_quad_to_gx2_params(mu0, v0, quad)
    w1, k1, l1, s1, m1 = norm_quad_to_gx2_params(mu1, v1, quad)
    F0 = float(cdf(0, w0, k0, l0, s0, m0))
    F1 = float(cdf(0, w1, k1, l1, s1, m1))
    err = p0 * F0 + p1 * (1 - F1)

    if not grad and not hess:
        return err

    g0, h0, at_cusp0, scale0, inv0, nx = _boundary_raw(0, mu0, v0, quad, wrt, hess, **kwargs)
    g1, h1, at_cusp1, scale1, inv1, nx = _boundary_raw(0, mu1, v1, quad, wrt, hess, **kwargs)

    raw_grad = {key: p0 * g0[key] - p1 * g1[key] for key in g0}
    raw_hess = None
    if hess:
        raw_hess = {key: p0 * h0[key] - p1 * h1[key] for key in h0}

    at_cusp = at_cusp0 | at_cusp1
    if np.any(at_cusp):
        cusp_scale = max(scale0, scale1)

        def inversion_grad_hess(s_eff, want_hess):
            cg0, ch0 = inv0(s_eff, want_hess)
            cg1, ch1 = inv1(s_eff, want_hess)
            cg = {key: p0 * cg0[key] - p1 * cg1[key] for key in cg0}
            ch = None if not want_hess else {key: p0 * ch0[key] - p1 * ch1[key] for key in ch0}
            return cg, ch

        raw_grad, raw_hess = _resolve_cusp_terms(raw_grad, raw_hess, at_cusp, inversion_grad_hess, cusp_scale)

    finalized = _finalize_grad_hess(raw_grad, raw_hess, hess, nx)
    if hess:
        g, h = finalized
        return err, g, h
    return err, finalized
