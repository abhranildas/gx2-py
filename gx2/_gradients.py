"""Exact (non-finite-differenced) gradient and Hessian of the generalized
chi-square cdf, with respect to its native parameters and to the quadratic
boundary coefficients of a normal classifier. Mirrors ``cdf_grad_gx2.m`` and
``cdf_grad_norm_quad.m``.
"""

import numpy as np
from scipy.integrate import quad_vec

from ._helpers import asrow
from ._convert import norm_quad_to_gx2_params
from ._distribution import cdf, pdf
from ._methods import imhof
from ._basic import char
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


def cdf_grad_norm_quad(x, mu, v, quad, wrt=None, hess=False,
                    AbsTol=1e-10, RelTol=1e-6, precision="basic", n_ruben=1000):
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
    n_ruben : int, optional
        Term-count cap passed through to the Ruben-series density
        derivatives used on the ``s == 0`` route (see :func:`gx2.dens_deriv`
        and :func:`gx2.ruben`). Only affects that route; ignored when
        ``s != 0``. Python-only addition, not yet in the MATLAB toolbox.

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
    if precision != "basic":
        raise NotImplementedError(
            "cdf_grad_norm_quad only supports precision='basic'; the 'vpa' "
            "path from cdf_grad_norm_quad.m has not been ported.")

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
    grad = {}

    # s=0 is the pure-quadratic-boundary (classification) regime, where the
    # inversion integrals lose convergence for total dof D<=4. There we take
    # the robust shifted-dof route: expand the weights in the eigenbasis,
    # M^{-1}=sum_j g_j u_j u_j' and mu_tilde=sum_j g_j c_j u_j (with
    # u_j=Sigma^{1/2} v_j, g_j=(1-2i t w_j)^{-1}, c_j=alpha_j+i t beta_j), so
    # that every block collapses to a finite sum of shifted-dof density
    # derivatives f^(n)_[..], evaluated robustly by dens_deriv. A single
    # engine (evalblock) serves the gradient (one factor it, p0=1) and the
    # Hessian (two, p0=2). When s!=0 the Gaussian damping makes the direct
    # inversion converge, and we keep it.
    s0 = (s == 0)
    if s0:
        U = S @ V                     # columns u_j = Sigma^{1/2} v_j
        alph = U.T @ SigInv_mu        # alpha_j = u_j' Sigma^{-1} mu
        bet = U.T @ q1c               # beta_j  = u_j' q1
        tol0 = 1e-9 * max(1.0, np.max(np.abs(dvals)))
        compj = np.full(dim, -1, dtype=int)     # mode -> merged (w,k) component; -1 for a zero mode
        for j in range(dim):
            if abs(dvals[j]) > tol0:
                compj[j] = int(np.argmin(np.abs(w - dvals[j])))
        memo = {}
        densopts = dict(AbsTol=AbsTol, RelTol=RelTol, n_ruben=n_ruben)

        def fder(bumpvec, n):
            # memoized robust n-th density derivative of the gx2 with
            # k+bumpvec dof
            key = (tuple(bumpvec.tolist()), n)
            if key in memo:
                return memo[key]
            val = np.asarray(dens_deriv(x, w, k + bumpvec, l, s, m, n, **densopts)).reshape(nx)
            memo[key] = val
            return val

        def Dterm(gmodes, p):
            # T[(it)^p prod_g(gmodes) phi] = (-1)^p f^{(p-1)}_[shift](x),
            # where each g_j advances its component's dof by 2 (rule R1) and
            # (it)^p is p argument-derivatives (rule R2). Zero modes add no
            # shift.
            bumpvec = np.zeros(k.size)
            for jj in gmodes:
                c = compj[jj]
                if c >= 0:
                    bumpvec[c] += 2
            return ((-1.0) ** p) * fder(bumpvec, p - 1)

        def cexp(cmodes, gmodes, p0):
            # T[(it)^p0 * prod_g(gmodes) * prod_{i in cmodes}(alpha_i+i t
            # beta_i) phi] = sum over subsets S of cmodes: (prod_S
            # beta)(prod_rest alpha) * Dterm(gmodes, p0+|S|). gmodes fixes
            # the dof shift; each beta pick raises the it-power (hence the
            # density-derivative order) by one.
            nc = cmodes.size
            val = np.zeros(nx)
            for mask in range(2 ** nc):
                coef = 1.0
                nb = 0
                for ii in range(nc):
                    if (mask >> ii) & 1:
                        coef *= bet[cmodes[ii]]
                        nb += 1
                    else:
                        coef *= alph[cmodes[ii]]
                if coef != 0:
                    val = val + coef * Dterm(gmodes, p0 + nb)
            return val

        def evalblock(monos, F, p0):
            # Assemble a rank-F tensor (shape (dim,)*F + (nx,)) from a list
            # of monomials in M^{-1} and mu_tilde. Each monomial mo records,
            # per free tensor index, which mode variable supplies its
            # u-column (mo['mv']), and, per mode variable, whether that
            # factor carries the linear term c=alpha+i t beta (mo['cv'] True
            # for a mu_tilde factor, False for a bare M^{-1}). Sums over all
            # mode assignments; for each, cexp expands the c-factors and
            # Dterm reads off the shifted-dof density derivative.
            T = np.zeros((dim,) * F + (nx,))
            for mo in monos:
                mv = mo["mv"]; cv = mo["cv"]; nmv = cv.size
                for lin in range(dim ** nmv):
                    assign = np.empty(nmv, dtype=int)
                    r = lin
                    for vv in range(nmv):
                        assign[vv] = r % dim
                        r //= dim
                    sc = cexp(assign[cv], assign, p0)      # (nx,)
                    if not np.any(sc):
                        continue
                    rank1 = U[:, assign[mv[0]]]
                    for f in range(1, F):
                        rank1 = np.multiply.outer(rank1, U[:, assign[mv[f]]])
                    T += mo["pref"] * np.multiply.outer(rank1, sc)
            return T

        def mk(pref, mv, cv):
            return dict(pref=pref, mv=np.asarray(mv, dtype=int), cv=np.asarray(cv, dtype=bool))
    else:
        def weights(t):
            phi = complex(np.asarray(char(t, w, k, l, s, m)))
            g = 1.0 / (1 - 2j * t * dvals)              # (dim,)
            Minv = S @ (V * g) @ V.T @ S
            pv = SigInv_mu + 1j * t * q1c                # p(t), (dim,)
            mut = Minv @ pv
            return Minv, mut, phi

        def grad_integrand(t):
            Minv, mut, phi = weights(t)
            P = Minv + np.outer(mut, mut)
            block = np.concatenate([mut[:, None], P], axis=1)   # (dim, dim+1)
            kern = np.exp(-1j * t * x)                          # (nx,)
            Bphi = block * phi
            return np.real(Bphi[:, :, None] * kern[None, None, :])

        def hess_integrand(t):
            Minv, mut, phi = weights(t)
            P = Minv + np.outer(mut, mut)
            Pmm = np.outer(mut, mut)
            Wq1q2 = 2 * np.einsum("ab,c->abc", Minv, mut) + np.einsum("a,bc->abc", mut, P)
            WQ2 = (2 * (np.einsum("qr,sp->pqrs", Minv, Minv)
                        + np.einsum("qr,sp->pqrs", Minv, Pmm)
                        + np.einsum("qr,sp->pqrs", Pmm, Minv))
                   + np.einsum("pq,rs->pqrs", P, P))
            Wflat = np.concatenate([
                np.array([1.0 + 0j]), mut.ravel(),
                P.ravel(), P.ravel(),
                Wq1q2.ravel(), WQ2.ravel(),
            ])
            kern = np.exp(-1j * t * x)
            return t * np.imag(Wflat[:, None] * phi * kern[None, :])

    # q0 block: dF/dq0 = -f(x0). Since q0 shifts q rigidly, this is just
    # -pdf.
    if wanted("q0"):
        if s0:
            g0 = Dterm(np.array([], dtype=int), 1)
        else:
            g0 = -np.asarray(pdf(x, w, k, l, s, m, **opts)).reshape(nx)
        grad["q0"] = g0 if nx > 1 else float(g0[0])

    # q1 and Q2 blocks. Inversion route (s!=0): one integration over t
    # returns both, since they share the weights M^{-1}(t) and mu_tilde(t);
    # the (it) of the master formula cancels the 1/t of the inversion,
    # leaving density-type integrals. Robust route (s=0): the same two
    # blocks from the eigenbasis engine at p0=1.
    if wanted("q1") or wanted("q2"):
        if s0:
            if wanted("q1"):
                Gq1 = evalblock([mk(1, [0], [True])], 1, 1)
                grad["q1"] = Gq1 if nx > 1 else Gq1[:, 0]
            if wanted("q2"):
                Pblk = [mk(1, [0, 0], [False]), mk(1, [0, 1], [True, True])]
                Gq2 = evalblock(Pblk, 2, 1)
                Gq2 = 0.5 * (Gq2 + np.swapaxes(Gq2, 0, 1))
                grad["q2"] = Gq2 if nx > 1 else Gq2[:, :, 0]
        else:
            A, _ = quad_vec(grad_integrand, 0, np.inf, epsabs=AbsTol, epsrel=RelTol)
            A = -A / np.pi                              # (dim, dim+1, nx)
            if wanted("q1"):
                Gq1 = A[:, 0, :]
                grad["q1"] = Gq1 if nx > 1 else Gq1[:, 0]
            if wanted("q2"):
                Gq2 = A[:, 1:, :]
                Gq2 = 0.5 * (Gq2 + np.swapaxes(Gq2, 0, 1))
                grad["q2"] = Gq2 if nx > 1 else Gq2[:, :, 0]

    if not hess:
        return grad

    # ---- Hessian (2nd output): the boundary blocks ----
    if s0:
        q0q0 = Dterm(np.array([], dtype=int), 2)
        q0q1 = evalblock([mk(1, [0], [True])], 1, 2)
        Pblk = [mk(1, [0, 0], [False]), mk(1, [0, 1], [True, True])]
        q0q2 = evalblock(Pblk, 2, 2)
        q1q1 = q0q2
        q1q2blk = [mk(2, [0, 0, 1], [False, True]),
                   mk(1, [0, 1, 1], [True, False]),
                   mk(1, [0, 1, 2], [True, True, True])]
        q1q2 = evalblock(q1q2blk, 3, 2)
        q2q2blk = [mk(2, [1, 0, 0, 1], [False, False]),
                   mk(2, [2, 0, 0, 1], [False, True, True]),
                   mk(2, [2, 0, 1, 2], [True, True, False]),
                   mk(1, [0, 0, 1, 1], [False, False]),
                   mk(1, [0, 0, 2, 1], [False, True, True]),
                   mk(1, [1, 0, 2, 2], [True, True, False]),
                   mk(1, [1, 0, 3, 2], [True, True, True, True])]
        q2q2 = evalblock(q2q2blk, 4, 2)
    else:
        Ntot = 1 + dim + dim ** 2 + dim ** 2 + dim ** 3 + dim ** 4
        Hraw, _ = quad_vec(hess_integrand, 0, np.inf, epsabs=AbsTol, epsrel=RelTol)
        Hraw = Hraw / np.pi
        off = 0
        q0q0 = Hraw[0, :]; off = 1
        q0q1 = Hraw[off:off + dim, :]; off += dim
        q0q2 = Hraw[off:off + dim ** 2, :].reshape(dim, dim, nx); off += dim ** 2
        q1q1 = Hraw[off:off + dim ** 2, :].reshape(dim, dim, nx); off += dim ** 2
        q1q2 = Hraw[off:off + dim ** 3, :].reshape(dim, dim, dim, nx); off += dim ** 3
        q2q2 = Hraw[off:off + dim ** 4, :].reshape(dim, dim, dim, dim, nx)

    q0q2 = 0.5 * (q0q2 + np.swapaxes(q0q2, 0, 1))
    q1q1 = 0.5 * (q1q1 + np.swapaxes(q1q1, 0, 1))

    hessian = dict(q0q0=q0q0, q0q1=q0q1, q0q2=q0q2, q1q1=q1q1, q1q2=q1q2, q2q2=q2q2)
    if nx == 1:
        hessian = {key: np.asarray(val)[..., 0] for key, val in hessian.items()}
    return grad, hessian
