"""Tests for the gx2 package.

Run with:  pytest -q
"""
import warnings
import numpy as np
import pytest
from scipy.stats import ncx2

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
