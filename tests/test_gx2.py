"""Tests for the gx2 package.

Run with:  pytest -q
"""
import warnings
import numpy as np
import pytest
from scipy.stats import ncx2
from scipy.integrate import IntegrationWarning

import gx2

warnings.simplefilter("ignore")

W, K, LAM, S, M = [1, -5, 2], [1, 2, 3], [2, 3, 7], 3.0, 5.0


def _f(x):
    return float(np.asarray(x).ravel()[0])


def test_stat():
    mu, v = gx2.stat([1, -5, 2], [1, 2, 3], [2, 3, 7], 0, 5)
    assert mu == pytest.approx(3.0)
    assert v == pytest.approx(546.0)


def test_char_at_zero():
    phi = gx2.char(np.array([0.0]), W, K, LAM, S, M)
    assert phi[0] == pytest.approx(1.0 + 0j)


def test_char_complex_arg():
    # regression: char() used to force-cast t to float, crashing on complex
    # input needed for Cauchy-integral moments / contour-shifted Fourier
    # integrals. Expected value cross-checked against MATLAB's gx2char.
    phi = gx2.char(0.3 + 0.7j, W, K, LAM, 0, 5.0)
    assert complex(phi) == pytest.approx(2.359135028791082e-06 - 8.189234259495991e-07j)


def test_ncx2_fallback_cdf_pdf():
    w, k, lam, s, m = [3.0], [4], [2.0], 0, 1.5
    x = np.array([5.0, 10.0, 20.0])
    assert np.allclose(gx2.cdf(x, w, k, lam, s, m), ncx2.cdf((x - m) / 3, 4, 2.0), atol=1e-9)
    assert np.allclose(gx2.pdf(x, w, k, lam, s, m), ncx2.pdf((x - m) / 3, 4, 2.0) / 3, atol=1e-9)


def test_cdf_methods_agree():
    x = 25.0
    p_imhof = _f(gx2.cdf(x, W, K, LAM, S, M, method="imhof"))
    p_ifft = _f(gx2.cdf(x, W, K, LAM, S, M, method="ifft"))
    p_ray = _f(gx2.cdf(x, W, K, LAM, S, M, method="ray", n_rays=200000))
    assert abs(p_imhof - p_ifft) < 2e-3
    assert abs(p_imhof - p_ray) < 5e-3


def test_pdf_integrates_to_one():
    xg = np.linspace(-150, 120, 4000)
    for method in ("imhof", "ifft"):
        fg = np.asarray(gx2.pdf(xg, W, K, LAM, S, M, method=method), dtype=float)
        assert np.trapezoid(fg, xg) == pytest.approx(1.0, abs=5e-3)


def test_pdf_is_derivative_of_cdf():
    x, dx = 10.0, 1e-3
    num = (_f(gx2.cdf(x + dx, W, K, LAM, S, M, method="imhof"))
           - _f(gx2.cdf(x - dx, W, K, LAM, S, M, method="imhof"))) / (2 * dx)
    ana = _f(gx2.pdf(x, W, K, LAM, S, M, method="imhof"))
    assert num == pytest.approx(ana, abs=1e-4)


def test_inv_round_trip():
    for p in (0.1, 0.5, 0.9):
        x = gx2.inv(p, W, K, LAM, S, M, method="imhof")
        assert _f(gx2.cdf(x, W, K, LAM, S, M, method="imhof")) == pytest.approx(p, abs=1e-4)


def test_ruben_matches_imhof():
    w, k, lam, m = [1, 2, 3], [1, 2, 3], [2, 3, 7], 5
    p_rub = _f(gx2.ruben(30.0, w, k, lam, m)[0])
    p_imh = _f(gx2.imhof(30.0, w, k, lam, 0, m)[0])
    assert p_rub == pytest.approx(p_imh, abs=1e-4)


@pytest.mark.parametrize("a", [1e-3, 1e-8])
@pytest.mark.parametrize("precision", ["basic", "vpa"])
def test_imhof_tiny_weights_upper_tail(a, precision):
    # regression for GitHub issue #1: Q = a*(Z1^2+Z2^2) + (Z3^2+Z4^2) is a sum
    # of two exponentials, so P(Q>x) = (e^{-x/2} - a e^{-x/(2a)})/(1-a). Ruben
    # can't converge (tiny min weight) and the plain Imhof quadrature used to
    # return 8e-4 or 0 here instead of ~4.5e-5.
    # The default tolerances apply to the Imhof integral (~pi/2), so they
    # bound the absolute error in p (by ~5e-7; in practice ~1e-8); the tighter
    # ones give tail-relative accuracy. Both must hold, with nothing flagged or warned.
    w, k, lam = [a, a, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]
    x = np.array([5.0, 20.0, 30.0])
    exact = (np.exp(-x / 2) - a * np.exp(-x / (2 * a))) / (1 - a)
    tight = dict(AbsTol=1e-14, RelTol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # no integration or clip warnings
        for xi, ei in zip(x, exact):     # scalar route, as in the issue
            p, p_err, _ = gx2.cdf(xi, w, k, lam, 0, 0, side="upper", method="auto",
                                  precision=precision, full_output=True)
            assert _f(p) == pytest.approx(ei, abs=5e-8, rel=1e-6)
            assert not np.any(p_err)
        if precision == "basic":
            for opts, tols in (({}, dict(atol=5e-8, rtol=0)),
                               (tight, dict(atol=0, rtol=1e-6))):
                for xs, ex in ((x[1], exact[1]), (x, exact)):  # scalar, grid
                    p, p_err, _ = gx2.cdf(xs, w, k, lam, 0, 0, side="upper",
                                          method="imhof", full_output=True, **opts)
                    assert np.allclose(p, ex, **tols)
                    assert not np.any(p_err)


def test_imhof_nonconvergence_is_flagged():
    # an absurd tolerance can't be met: the result must be flagged in errflag
    # and warned about, rather than returned as if accurate
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, errflag = gx2.imhof(20.0, [1e-3, 1e-3, 1, 1], [1, 1, 1, 1],
                               [0, 0, 0, 0], 0, 0, side="upper",
                               AbsTol=1e-300, RelTol=1e-300)
    assert np.all(errflag)
    assert any(issubclass(c.category, IntegrationWarning) for c in caught)


def test_ellipse_matches_imhof_finite_tail():
    w, k, lam, m = [4, 5, 1], [1, 2, 3], [5, 6, 0], -50
    p_ell = _f(gx2.ellipse(-40.0, w, k, lam, m)[0])
    p_imh = _f(gx2.imhof(-40.0, w, k, lam, 0, m)[0])
    assert abs(np.log10(p_ell) - np.log10(p_imh)) < 0.3


def test_tail_matches_imhof_moderate():
    w, k, lam, s, m = [-2, -3, 4, 1], [4, 2, 6, 1], [10, 20, 5, 0], 20, 0
    p_tail = _f(gx2.cdf(300.0, w, k, lam, s, m, side="upper", method="tail"))
    p_imh = _f(gx2.cdf(300.0, w, k, lam, s, m, side="upper", method="imhof",
                       AbsTol=1e-18, RelTol=1e-13))
    assert p_tail == pytest.approx(p_imh, rel=0.2)


def test_conversion_round_trip_stats():
    quad = gx2.gx2_to_norm_quad_params(W, K, LAM, S, M)
    dim = quad["q1"].size
    w, k, lam, s, m = gx2.norm_quad_to_gx2_params(np.zeros(dim), np.eye(dim), quad)
    mu0, v0 = gx2.stat(W, K, LAM, S, M)
    mu1, v1 = gx2.stat(w, k, lam, s, m)
    assert mu0 == pytest.approx(mu1, abs=1e-6)
    assert v0 == pytest.approx(v1, abs=1e-6)


def test_rnd_matches_stats():
    mu, v = gx2.stat(W, K, LAM, S, M)
    for method in ("sum", "norm_quad"):
        r = gx2.rnd(W, K, LAM, S, M, size=(200000,), method=method)
        assert abs(r.mean() - mu) < 0.4
        assert abs(r.var() - v) / v < 0.06


def test_ray_grid_matches_imhof():
    w, k, lam, s, m = [1.0, -2.0], [1, 1], [0.0, 0.0], 0, 0
    x = np.array([-1.0, 0.5, 3.0])
    p_ray = np.asarray(gx2.cdf(x, w, k, lam, s, m, method="ray"), dtype=float)
    p_imh = np.asarray(gx2.cdf(x, w, k, lam, s, m, method="imhof"), dtype=float)
    assert np.allclose(p_ray, p_imh, atol=2e-3)


def test_ray_log_precision_subrealmin():
    w, k, lam, s, m = [-2, -3, 4, 1], [4, 2, 6, 1], [10, 20, 5, 0], 20, 0
    v = _f(gx2.cdf(12000.0, w, k, lam, s, m, side="upper", method="ray",
                   precision="log", n_rays=3000))
    assert v < -100


def test_ray_log_finite_tail_raises():
    w, k, lam, s, m = [2, 3, 4, 1], [4, 2, 6, 1], [0, 0, 0, 0], 0, 0
    with pytest.raises(RuntimeError):
        gx2.cdf_ray(np.array([1e-70]), w, k, lam, s, m, side="lower", n_rays=100)


# ---------------------------------------------------------------------------
# Boundary derivatives at s=0 and small total dof D. There the Hessian's
# inversion integrand behaves like t^(1-D/2) (gx2_derivatives.md, section
# 1.4), so for D<=2 it does not decay, and cdf_grad_bd's value rests on the
# extrapolated cycle sums of _osc_integral. The reference here uses no gx2
# code: for a normal with diagonal covariance, F=P(q(x)<=0) is a closed form
# in 1-D (from the roots of the quadratic) and a single 1-D integral in 2-D
# (given x1, q is a quadratic in x2), evaluated in mpmath, with directional
# derivatives by high-precision central differences.
# ---------------------------------------------------------------------------
def _interval_prob(A, B, C, mu, sd):
    """P(A y^2 + B y + C <= 0) for y ~ N(mu, sd^2), A != 0, in mpmath."""
    import mpmath as mp
    disc = B * B - 4 * A * C
    if disc <= 0:
        return mp.mpf(0) if A > 0 else mp.mpf(1)
    r = sorted([(-B - mp.sqrt(disc)) / (2 * A), (-B + mp.sqrt(disc)) / (2 * A)])
    inside = mp.ncdf((r[1] - mu) / sd) - mp.ncdf((r[0] - mu) / sd)
    return inside if A > 0 else 1 - inside


def _region_prob(mu, sd, Q2, q1, q0):
    """P(x'Q2 x + q1'x + q0 <= 0) for x ~ N(mu, diag(sd^2)), d = 1 or 2."""
    import mpmath as mp
    if len(mu) == 1:
        return _interval_prob(Q2[0][0], q1[0], q0, mu[0], sd[0])
    A = Q2[1][1]
    B = lambda x1: 2 * Q2[0][1] * x1 + q1[1]
    C = lambda x1: Q2[0][0] * x1 ** 2 + q1[0] * x1 + q0
    # kinks of the integrand: where the x2-quadratic's discriminant vanishes
    a = 4 * Q2[0][1] ** 2 - 4 * A * Q2[0][0]
    b = 4 * Q2[0][1] * q1[1] - 4 * A * q1[0]
    c = q1[1] ** 2 - 4 * A * q0
    kinks = [z.real for z in mp.polyroots([a, b, c]) if abs(z.imag) < 1e-20] if a != 0 else []
    pts = [-mp.inf] + sorted(kinks) + [mp.inf]
    f = lambda x1: mp.npdf(x1, mu[0], sd[0]) * _interval_prob(A, B(x1), C(x1), mu[1], sd[1])
    return mp.quad(f, pts)


@pytest.mark.parametrize("mu, v, means, covs", [
    ([0.0], [1.0], ([0.0], [3.0]), ([1.0], [4.0])),                 # D=1 (problem 1)
    ([2.0], [1.05], ([0.0], [2.0]), ([1.0], [1.05])),                # D=1, near-linear (problem 3)
    ([.3, .3], [3, .5], ([-.3, -.3], [.3, .3]), ([1, 1], [3, .5])),  # D=2, mixed sign (problem 7)
    ([.3, .3], [.4, .6], ([-.3, -.3], [.3, .3]), ([1, 1], [.4, .6])),  # D=2, same sign (problem 5)
])
def test_boundary_derivs_small_dof(mu, v, means, covs):
    import mpmath as mp
    mp.mp.dps = 30
    d = len(mu)
    quad = gx2.norm_class_opt_bd(np.array(means[0]), np.diag(covs[0]),
                                 np.array(means[1]), np.diag(covs[1]))
    g, H = gx2.cdf_grad_bd(0, mu, np.diag(v), quad, hess=True, AbsTol=1e-12, RelTol=1e-10)
    sd = [mp.sqrt(vi) for vi in v]
    mmu = [mp.mpf(x) for x in mu]
    Q2 = np.atleast_2d(quad["q2"]); q1 = np.atleast_1d(quad["q1"]); q0 = float(quad["q0"])
    rng = np.random.default_rng(0)
    for _ in range(2):
        dQ2 = rng.standard_normal((d, d)); dQ2 = (dQ2 + dQ2.T) / 2
        dq1 = rng.standard_normal(d); dq0 = rng.standard_normal()

        def F(e):
            Qe = [[mp.mpf(Q2[i, j]) + e * dQ2[i, j] for j in range(d)] for i in range(d)]
            return _region_prob(mmu, sd, Qe, [mp.mpf(q1[i]) + e * dq1[i] for i in range(d)],
                                mp.mpf(q0) + e * dq0)
        h1, h2 = mp.mpf("1e-10"), mp.mpf("1e-6")
        ref_g = float((F(h1) - F(-h1)) / (2 * h1))
        ref_h = float((F(h2) - 2 * F(0) + F(-h2)) / h2 ** 2)
        an_g = np.sum(g["q2"] * dQ2) + g["q1"] @ dq1 + g["q0"] * dq0
        an_h = (np.einsum("abcd,ab,cd", H["q2q2"], dQ2, dQ2) + 2 * np.einsum("abc,a,bc", H["q1q2"], dq1, dQ2)
                + dq1 @ H["q1q1"] @ dq1 + 2 * dq0 * (H["q0q1"] @ dq1) + 2 * dq0 * np.sum(H["q0q2"] * dQ2)
                + H["q0q0"] * dq0 ** 2)
        assert abs(an_g - ref_g) < 1e-8 * max(1, abs(ref_g))
        assert abs(an_h - ref_h) < 1e-8 * max(1, abs(ref_h))
