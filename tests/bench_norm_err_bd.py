"""Wider benchmark of ``gx2.norm_err`` (analytic gradient/Hessian of
total two-class classification error w.r.t. the boundary coefficients) on
real binary-Gaussian classification problems spanning D=1..5.

For each problem this computes the optimal quadratic (Bayes) boundary, then
compares three ways of getting the *Hessian* of the classification error
w.r.t. that boundary's coefficients (q2,q1,q0) there:
  - a tight-tolerance analytic evaluation, treated as ground truth;
  - the default-tolerance analytic evaluation (production settings);
  - finite differences via ``numdifftools`` (Richardson-extrapolated).
(The gradient is also recorded at this boundary, but it's identically ~0
there -- it's the error minimizer -- so it's not a useful accuracy
benchmark on its own; see the naive-QDA-boundary gradient test below for one
that is.) Some problems are deliberately included *because* one or both
methods may be slow there (mixed-sign, higher-dimension boundaries) -- the
point of this benchmark is partly to demonstrate exactly that contrast, not
to avoid it.

Each problem also gets the same tight/default/FD comparison for the plain
*gradient* (no Hessian) at a second, deliberately non-optimal boundary: a
naive-QDA boundary, i.e. the same quadratic-discriminant formula as the
optimal boundary above, but with each class's covariance replaced by its own
diagonal (see ``naive_qda_boundary``). Unlike a Fisher/LDA boundary (which
collapses Q2 to exactly zero), this keeps a genuinely nonzero, generically
mixed-sign quadratic term, so it exercises the same non-degenerate gx2
machinery (Ruben/Imhof, cross-component derivatives) as the optimal-boundary
Hessian test -- while still being deliberately non-optimal for the real
(fully correlated) problem, so the error's gradient there is generically
nonzero, which is exactly what the optimal-boundary Hessian test's own
gradient can't offer.

Every one of a problem's stages (4 for the optimal-boundary Hessian test, 3
for the naive-QDA-boundary gradient test -- ground truth, default-tol, FD; no
FD Hessian is computed there) runs in its own subprocess, with no wall-clock cap:
a stage runs to completion however long it needs (some of these, by design,
take hours -- see below), rather than being killed on a timeout. The
subprocess boundary here is only for per-stage warning capture (see
``_mp_call``), not for enforcing a time limit.

Designed to run on a separate, faster, multi-core machine. Parallelism is at
the *problem* level only, one problem per ``ProcessPoolExecutor`` worker,
which runs that problem's stages one after another rather than fanning them
out further -- and every worker is pinned to a single thread (the
OMP/OpenBLAS/MKL/numexpr env vars set at the very top of this file, before
numpy is ever imported, in every process this script spawns). Together
these two choices mean a problem's reported time is what it actually costs
running alone on one core, not smeared across however many other problems
or BLAS threads happened to be sharing the machine at that moment. This
trades away throughput for correctness: a handful of genuinely slow problems
(mixed-sign, higher-D FD Hessians can run for hours) can now each occupy a
worker for its whole duration while fast, already-finished-in-milliseconds
problems sit queued behind them, so the *total* wall-clock for the whole
sweep is no longer close to optimal -- only the correctness of each
individual problem's own number is guaranteed. (An earlier version of this
script scheduled at the stage level instead, precisely to avoid that
idling; if throughput ever matters more than clean single-threaded timing
again, that design is the one to revert to.)

Results are written incrementally (one JSON file per problem, updated after
every stage) so a killed or crashed run still leaves usable diagnostics for
whatever finished. This is safe for a worker to do directly to disk itself
(unlike under the old stage-level design, where several workers could touch
the same problem's file concurrently): here exactly one worker owns a given
problem's file for that problem's entire lifetime, so there's no
multi-process race to guard against.

Any warning raised while computing a stage (numpy overflow/invalid-value,
scipy's IntegrationWarning, gx2's own ImhofClipWarning, ...) is redirected
into that problem's own log_<name>.log, tagged with which stage raised it,
instead of going to stderr where it would be interleaved with every other
worker's output and impossible to attribute afterward.

Deliberately NOT using ``joblib`` for the outer parallelism: joblib's default
``loky`` backend monkeypatches the global multiprocessing context on Windows
in a way that breaks the plain ``multiprocessing.Process`` objects used below
for per-stage subprocess isolation (nesting the two raises
``AttributeError: 'Process' object has no attribute 'env'`` -- caught directly
while testing this script, not a theoretical concern). The standard library's
``ProcessPoolExecutor`` doesn't have this problem.

Requires, in addition to gx2's own dependencies: numdifftools.
    pip install numdifftools

Usage:
    python bench_norm_err_bd.py [output_dir]
"""
import os

# Force every numeric library's internal thread pool to size 1, and do it
# *before* numpy (and whatever BLAS backend it loads) is ever imported --
# these are read once at that import/init time, not re-checked later. Since
# every process this script spawns (each ProcessPoolExecutor worker, and
# each nested per-stage multiprocessing.Process) re-executes this module
# from scratch under Windows' "spawn" start method, this line reliably runs
# again, before numpy, in every one of them -- so single-threadedness here
# doesn't depend on environment inheritance working correctly, only on this
# module being imported fresh each time, which spawn guarantees. Without
# this, a single linear-algebra call could silently fan out across every
# core via BLAS/OpenMP, and running N problems concurrently (the whole point
# of problem-level parallelism, see the module docstring) would oversubscribe
# the machine by a factor of N -- making every problem's measured time
# depend on how many *other* problems happened to be running at that moment,
# not on the problem itself.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import queue
import sys
import time
import traceback
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numdifftools as nd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gx2

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The problems. Hardcoded (no live randomness) so this file and any
# write-up of its results always agree. Every one of these was individually
# timed against gx2-py before being chosen -- see gx2_derivatives.md's
# "wider benchmark" subsection for why the parameters look the way they do
# (small mean separations were needed to avoid an unpredictable, order(s)
# of magnitude slower regime that the mixed-sign s=0 corner can fall into).
# ---------------------------------------------------------------------------
# Numbered and ordered first by dimension, then by boundary type (linear,
# elliptic, hyperbolic, parabolic) -- see gx2_derivatives.md section 2.2.1,
# whose two tables use this exact numbering and ordering.
PROBLEMS = [
    dict(name="1_D1_generic_A", mu0=[0.0], v0=[[1.0]], mu1=[3.0], v1=[[4.0]]),
    dict(name="2_D1_generic_B", mu0=[0.0], v0=[[1.0]], mu1=[1.5], v1=[[0.4]]),
    dict(name="3_D1_near_linear", mu0=[0.0], v0=[[1.0]], mu1=[2.0], v1=[[1.05]]),

    # D=2, linear: identical covariance for both classes (Q2=0 exactly), so
    # the boundary collapses to purely linear (classic LDA). Exercises
    # norm_quad_to_gx2_params' opposite degenerate corner from the same-mean
    # problem below (an *empty* w/k, pure normal term s and offset m) --
    # verified directly to work, and the cheapest case in the whole benchmark
    # (a plain gx2.cdf() call here is sub-millisecond).
    dict(name="4_D2_same_cov", mu0=[-.3, -.3], v0=[[1, 0], [0, 1]],
         mu1=[.3, .3], v1=[[1, 0], [0, 1]]),

    # D=2, elliptic: same-sign contrast -- routes through the faster
    # same-sign code path (gx2_derivatives.md open item 3.2) rather than the
    # slower mixed-sign convolution.
    dict(name="5_D2_same_sign", mu0=[-.3, -.3], v0=[[1, 0], [0, 1]],
         mu1=[.3, .3], v1=[[0.4, 0], [0, 0.6]]),

    # D=2, elliptic: same covariances as problem 5, but with unequal priors
    # p0=0.2, p1=0.8 -- exercises the p0!=p1 pathway, untested elsewhere in
    # this set.
    dict(name="6_D2_unequal_prior", mu0=[-.3, -.3], v0=[[1, 0], [0, 1]],
         mu1=[.3, .3], v1=[[0.4, 0], [0, 0.6]], p0=0.2, p1=0.8),

    dict(name="7_D2_generic_axis", mu0=[-.3, -.3], v0=[[1, 0], [0, 1]],
         mu1=[.3, .3], v1=[[3.0, 0], [0, 0.5]]),
    dict(name="8_D2_generic_rot", mu0=[-.3, -.3], v0=[[1, 0], [0, 1]],
         mu1=[.3, .3], v1=[[2.0, 0.8], [0.8, 1.0]]),

    # D=2, hyperbolic: the density-cusp case -- the boundary lands exactly on
    # the mixed-sign, k=1 offset m (see gx2_derivatives.md open item 3.3/§1.5
    # and the cusp handling in cdf_grad_bd/norm_err).
    dict(name="9_D2_generic_crossed", mu0=[.3, -.3], v0=[[0.5, 0], [0, 2.0]],
         mu1=[-.3, .3], v1=[[2.0, 0], [0, 0.5]]),

    dict(name="10_D2_near_linear", mu0=[0.0, 0.0], v0=[[1.0, .2], [.2, 1.0]],
         mu1=[.3, .2], v1=[[1.05, .2], [.2, .97]]),

    # D=2, hyperbolic: same mean vector for both classes (mu0=mu1=0), so the
    # linear boundary term q1 vanishes identically (q1=v1inv@mu1-v0inv@mu0=0
    # when mu0=mu1) and the boundary is driven purely by the covariance
    # difference -- reuses problem 7's covariance pair with the mean
    # separation zeroed out. Verified directly: despite being k=1-per-weight
    # mixed-sign like problem 9, this does NOT land on the density cusp,
    # since mu=0 forces each class's offset m to equal q0 (nonzero here,
    # since det(v0)!=det(v1)).
    dict(name="11_D2_same_mean", mu0=[0.0, 0.0], v0=[[1, 0], [0, 1]],
         mu1=[0.0, 0.0], v1=[[3.0, 0], [0, 0.5]]),

    # D=2, parabolic: a genuine parabolic (paraboloid) boundary -- Q2
    # rank-deficient (one exact zero eigenvalue, axis 1) but not the zero
    # matrix, AND the mean difference has a nonzero component along that
    # same flat axis (without that second condition, a rank-deficient Q2
    # alone would only give a degenerate cylinder over a lower-D
    # elliptic/hyperbolic cross-section, not a true paraboloid -- verified
    # directly: q1 comes out with its only nonzero entry exactly on the flat
    # axis). A flat direction always carries a nonzero normal term s, so
    # unlike every elliptic/hyperbolic problem above, this can never land on
    # the s=0 density cusp of open item 3.3, regardless of the other axes.
    dict(name="12_D2_parabolic", mu0=[-.3, 0.0], v0=[[1, 0], [0, 1]],
         mu1=[.3, 0.0], v1=[[1, 0], [0, 3.0]]),

    # D=3 counterpart of problem 4.
    dict(name="13_D3_same_cov", mu0=[-.3, -.3, -.3], v0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         mu1=[.3, .3, .3], v1=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),

    # D=3 counterpart of problem 5.
    dict(name="14_D3_same_sign", mu0=[-.3, -.3, -.3], v0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         mu1=[.3, .3, .3], v1=[[0.4, 0, 0], [0, 0.6, 0], [0, 0, 0.5]]),

    dict(name="15_D3_generic", mu0=[-.3, -.3, -.3], v0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         mu1=[.3, .3, .3], v1=[[2.0, 0, 0], [0, 0.5, 0], [0, 0, 3.0]]),
    dict(name="16_D3_generic_crossed", mu0=[.3, 0, -.3], v0=[[0.5, 0, 0], [0, 2.0, 0], [0, 0, 0.7]],
         mu1=[-.3, 0, .3], v1=[[2.0, 0, 0], [0, 0.5, 0], [0, 0, 1.4]]),
    dict(name="17_D3_near_linear", mu0=[0.0, 0.0, 0.0], v0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         mu1=[.3, .2, 0.0], v1=[[1.05, .03, 0], [.03, 1.02, .02], [0, .02, 1.04]]),

    # D=3 counterpart of problem 11 (same-mean, reuses problem 15's covariance pair).
    dict(name="18_D3_same_mean", mu0=[0.0, 0.0, 0.0], v0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         mu1=[0.0, 0.0, 0.0], v1=[[2.0, 0, 0], [0, 0.5, 0], [0, 0, 3.0]]),

    # D=3 counterpart of problem 12 (parabolic).
    dict(name="19_D3_parabolic", mu0=[-.3, 0.0, 0.0], v0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         mu1=[.3, 0.0, 0.0], v1=[[1, 0, 0], [0, 3.0, 0], [0, 0, 0.5]]),

    # D=4/D=5 same-sign vs. mixed-sign pairs (problems 20-23), deliberately
    # kept in despite the mixed-sign ones being slow. During design, problem
    # 20's covariances swapped for problem 21's mixed-sign ones didn't
    # finish its analytic Hessian in 16+ minutes (killed, not measured to
    # completion) -- both the analytic method and FD may time out here.
    # That's the point: this is a genuine, currently-unresolved limitation
    # (see gx2_derivatives.md open item 3.2's discussion of dimension
    # scaling), not something to hide by omission.
    dict(name="20_D4_same_sign", mu0=[-.3] * 4, v0=np.eye(4).tolist(),
         mu1=[.3] * 4, v1=np.diag([0.4, 0.5, 0.6, 0.7]).tolist()),
    dict(name="21_D4_mixed_sign", mu0=[-.3] * 4, v0=np.eye(4).tolist(),
         mu1=[.3] * 4, v1=np.diag([2.0, 0.5, 3.0, 1.5]).tolist()),
    dict(name="22_D5_same_sign", mu0=[-.3] * 5, v0=np.eye(5).tolist(),
         mu1=[.3] * 5, v1=np.diag([0.4, 0.5, 0.6, 0.7, 0.5]).tolist()),
    dict(name="23_D5_mixed_sign", mu0=[-.3] * 5, v0=np.eye(5).tolist(),
         mu1=[.3] * 5, v1=np.diag([2.0, 0.5, 3.0, 1.5, 0.7]).tolist()),
]

P0 = P1 = 0.5  # equal priors throughout

# ---------------------------------------------------------------------------
# The naive-QDA boundary: the same quadratic-discriminant formula as
# gx2.norm_class_opt_bd, but with each class's covariance replaced by its own
# diagonal (i.e. ignoring cross-feature correlations). Unlike a Fisher/LDA
# boundary this keeps a genuinely nonzero, generically mixed-sign quadratic
# term Q2 -- so it exercises the same non-degenerate gx2 machinery
# (Ruben/Imhof, cross-component derivatives) as the optimal-boundary Hessian
# test -- while still being deliberately non-optimal for the true (fully
# correlated) problem, so the classification error's gradient there is
# generically nonzero, unlike at the true optimum, where it is ~0 by
# construction and so useless as an accuracy benchmark. This is the boundary
# the naive_qda_* stages below use for that reason.
# ---------------------------------------------------------------------------
def naive_qda_boundary(mu0, v0, mu1, v1, p0=P0, p1=P1):
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
    v0 = np.atleast_2d(np.asarray(v0, dtype=float))
    v1 = np.atleast_2d(np.asarray(v1, dtype=float))
    dv0 = np.diag(v0)
    dv1 = np.diag(v1)
    q2 = np.diag(0.5 * (1.0 / dv1 - 1.0 / dv0))
    q1 = mu0 / dv0 - mu1 / dv1
    q0 = (0.5 * (np.sum(mu1 ** 2 / dv1) - np.sum(mu0 ** 2 / dv0))
          + 0.5 * (np.sum(np.log(dv1)) - np.sum(np.log(dv0)))
          + np.log(p0 / p1))
    return {"q2": q2, "q1": q1, "q0": float(q0)}


# ---------------------------------------------------------------------------
# Flattening: theta = [vech(Q2) (upper triangle incl. diagonal, row-major),
# q1, q0]. unflatten's off-diagonal entries mirror both (a,b) and (b,a)
# together, matching cdf_grad_bd's symmetric-perturbation convention
# for the gradient/Hessian -- verified against finite differences of a plain
# cdf() call before use here (see gx2_derivatives.md's wider-benchmark
# writeup for the cross-check).
# ---------------------------------------------------------------------------
def n_params(D):
    return D * (D + 1) // 2 + D + 1


def flatten(quad, D):
    theta = np.zeros(n_params(D))
    idx = 0
    for r in range(D):
        for c in range(r, D):
            theta[idx] = quad["q2"][r, c]
            idx += 1
    theta[idx:idx + D] = quad["q1"]
    theta[idx + D] = quad["q0"]
    return theta


def unflatten(theta, D):
    q2 = np.zeros((D, D))
    idx = 0
    for r in range(D):
        for c in range(r, D):
            q2[r, c] = theta[idx]
            q2[c, r] = theta[idx]
            idx += 1
    q1 = np.array(theta[idx:idx + D])
    q0 = float(theta[idx + D])
    return {"q2": q2, "q1": q1, "q0": q0}


def flatten_dir(i, D):
    """The (dq2, dq1, dq0) perturbation corresponding to flat coordinate i."""
    dq2 = np.zeros((D, D)); dq1 = np.zeros(D); dq0 = 0.0
    idx = 0
    for r in range(D):
        for c in range(r, D):
            if idx == i:
                dq2[r, c] = 1.0
                dq2[c, r] = 1.0
            idx += 1
    for j in range(D):
        if idx == i:
            dq1[j] = 1.0
        idx += 1
    if idx == i:
        dq0 = 1.0
    return dq2, dq1, dq0


def grad_flat(grad, D):
    """Flatten a cdf_grad_bd-style gradient dict to the theta basis."""
    out = np.zeros(n_params(D))
    idx = 0
    for r in range(D):
        for c in range(r, D):
            out[idx] = grad["q2"][r, c] if r == c else 2 * grad["q2"][r, c]
            idx += 1
    out[idx:idx + D] = grad["q1"]
    out[idx + D] = grad["q0"]
    return out


def _safe_contract(H, W):
    """sum(H*W) without 0*inf turning into nan: a zero weight means "this raw
    entry isn't part of this basis direction," so it must contribute exactly
    0 even where H itself is +-inf (as it legitimately is at a density-cusp
    boundary -- see gx2_derivatives.md open item 3.3/3.7(a)). Plain H*W
    computes IEEE 0*inf=nan for every such entry, and that nan then poisons
    the whole sum even though the raw per-block Hessian is perfectly
    determinate. dq2a/dq1a/dq0a/... (see flatten_dir) are always exactly 0/1
    indicators, and outer products of 0/1 indicators are always exactly 0/1
    too, so "W==0" unambiguously means "excluded," never "a genuinely tiny
    but nonzero weight."
    """
    H = np.asarray(H, dtype=float)
    W = np.asarray(W, dtype=float)
    prod = np.where(W == 0, 0.0, H * W)
    return float(np.sum(prod))


def _safe_max_abs_diff(a, b):
    """max(abs(a-b)) without agreeing +-inf entries turning into nan: at the
    density-cusp boundary (problem 9; see gx2_derivatives.md open item 3.3
    and Table 2.2.1.1's problem-9 footnotes), both the ground-truth and the
    default-tolerance analytic Hessian are genuinely +inf at every entry, so
    they agree exactly there and the true error is 0 -- but plain a-b computes
    IEEE inf-inf=nan for every such entry, and that nan then poisons the
    whole np.max even though every other entry may be perfectly finite and
    the two arrays otherwise agree. A mismatched pair (one side inf, the
    other finite) is a genuine, real disagreement and must still register as
    inf, which the diff already does correctly on its own -- only the
    same-sign-inf-vs-same-sign-inf case needs the override.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.abs(a - b)
    agreeing_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    diff = np.where(agreeing_inf, 0.0, diff)
    return float(np.max(diff))


def _hess_dir(hess, dq2a, dq1a, dq0a, dq2b, dq1b, dq0b):
    val = _safe_contract(hess["q2q2"], np.multiply.outer(dq2a, dq2b))
    val += _safe_contract(hess["q1q2"], np.multiply.outer(dq1a, dq2b))
    val += _safe_contract(hess["q1q2"], np.multiply.outer(dq1b, dq2a))
    val += _safe_contract(hess["q1q1"], np.multiply.outer(dq1a, dq1b))
    val += _safe_contract(hess["q0q1"], dq1a * dq0b) + _safe_contract(hess["q0q1"], dq1b * dq0a)
    val += (_safe_contract(hess["q0q2"], dq2a * dq0b)
            + _safe_contract(hess["q0q2"], dq2b * dq0a))
    val += _safe_contract(hess["q0q0"], dq0a * dq0b)
    return val


def hess_flat(hess, D):
    """Flatten a cdf_grad_bd-style Hessian dict to the theta basis."""
    P = n_params(D)
    dirs = [flatten_dir(i, D) for i in range(P)]
    H = np.zeros((P, P))
    for i in range(P):
        for j in range(i, P):
            H[i, j] = H[j, i] = _hess_dir(hess, *dirs[i], *dirs[j])
    return H


# ---------------------------------------------------------------------------
# Per-problem worker
# ---------------------------------------------------------------------------
def total_err(theta, mu0, v0, mu1, v1, D, p0=P0, p1=P1):
    return gx2.norm_err(mu0, v0, mu1, v1, unflatten(theta, D), p0=p0, p1=p1)


# ---------------------------------------------------------------------------
# Per-stage subprocess. Each stage runs in its own subprocess purely so its
# warnings can be captured and attributed (see _mp_call) -- there is no
# wall-clock cap, so a genuinely slow stage (some of these problems are here
# specifically to show that) simply runs until it finishes, however long that
# takes. Stage functions are plain top-level functions (not closures) so they
# can be pickled and sent to the subprocess; any closures they need internally
# (e.g. fd_fun below) are created only after the subprocess starts, so they
# never need to cross the pickling boundary themselves.
# ---------------------------------------------------------------------------
def _mp_call(q, func, args, tag=None, warn_log_path=None):
    if warn_log_path is not None:
        # Route every warning raised while computing this one stage (numpy
        # overflow/invalid-value, scipy IntegrationWarning, gx2's own
        # ImhofClipWarning, ...) into that problem's log file instead of
        # stderr, tagged with which stage produced it -- these are exactly
        # the symptoms of the hard-corner numerical issues already tracked
        # in gx2_derivatives.md open items 3.1-3.3 and 3.7, so keeping a
        # per-stage record makes it possible to correlate a specific warning
        # with the specific (problem, stage) that triggered it after the
        # fact, rather than an undifferentiated console stream.
        warnings.simplefilter("always")

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            try:
                with open(warn_log_path, "a") as fh:
                    fh.write(f"[{time.strftime('%H:%M:%S')}] [{tag}] {category.__name__}: "
                             f"{message} ({os.path.basename(filename)}:{lineno})\n")
            except OSError:
                pass

        warnings.showwarning = _showwarning
    try:
        q.put(("ok", func(*args)))
    except Exception:
        q.put(("error", traceback.format_exc()))


def run_in_subprocess(func, args, tag=None, warn_log_path=None):
    """Run func(*args) in a subprocess; returns (status, value) with status
    in {'ok', 'error'}. No timeout: blocks until the subprocess finishes,
    however long that takes."""
    q = mp.Queue()
    p = mp.Process(target=_mp_call, args=(q, func, args, tag, warn_log_path))
    p.start()
    p.join()
    try:
        return q.get(timeout=5)
    except queue.Empty:
        return "error", "worker exited without producing a result"


def _stage_analytic(mu0, v0, mu1, v1, quad, p0, p1, tol_kwargs, hess=True):
    return gx2.norm_err(mu0, v0, mu1, v1, quad, p0=p0, p1=p1, grad=True, hess=hess, **tol_kwargs)


def _stage_fd_grad(theta0, mu0, v0, mu1, v1, D, p0, p1, log_path):
    call_count = [0]

    def fd_fun(theta):
        call_count[0] += 1
        if call_count[0] % 20 == 0:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}]   FD grad call {call_count[0]}\n")
        return total_err(theta, mu0, v0, mu1, v1, D, p0=p0, p1=p1)

    grad = np.asarray(nd.Gradient(fd_fun)(theta0))
    return grad, call_count[0]


def _stage_fd_hess(theta0, mu0, v0, mu1, v1, D, p0, p1, log_path, num_steps=9):
    call_count = [0]

    def fd_fun(theta):
        call_count[0] += 1
        if call_count[0] % 20 == 0:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}]   FD hess call {call_count[0]}\n")
        return total_err(theta, mu0, v0, mu1, v1, D, p0=p0, p1=p1)

    hess = np.asarray(nd.Hessian(fd_fun, num_steps=num_steps)(theta0))
    return hess, call_count[0]


def _write_partial(path, result):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    os.replace(tmp, path)  # atomic on both POSIX and Windows


def _store_stage_result(entry, stage, status, value, elapsed, D):
    """Store one stage's raw (flattened) output into entry, in place. The
    4 optimal-boundary Hessian-test stages keep their original top-level
    field names (ground_truth/default_tol/fd_grad/fd_hess); the 3
    naive-QDA-boundary gradient-test stages nest under entry["naive_qda"]
    instead, keyed by the same ground_truth/default_tol/fd_grad names, so the
    two tests never collide despite sharing a stage-naming convention."""
    if stage == "ground_truth":
        if status == "ok":
            err_gt, grad_gt, hess_gt = value
            entry["ground_truth"] = dict(status=status, time_s=elapsed, err=err_gt,
                                          grad=grad_flat(grad_gt, D).tolist(),
                                          hess=hess_flat(hess_gt, D).tolist(),
                                          tol="AbsTol=1e-12, RelTol=1e-10, n_ruben=5000")
        else:
            entry["ground_truth"] = dict(status=status, time_s=elapsed, detail=str(value))
    elif stage == "default_tol":
        if status == "ok":
            err_d, grad_d, hess_d = value
            entry["default_tol"] = dict(status=status, time_s=elapsed, err=err_d,
                                         grad=grad_flat(grad_d, D).tolist(),
                                         hess=hess_flat(hess_d, D).tolist(),
                                         tol="AbsTol=1e-10, RelTol=1e-6 (package defaults)")
        else:
            entry["default_tol"] = dict(status=status, time_s=elapsed, detail=str(value))
    elif stage == "fd_grad":
        if status == "ok":
            fd_grad, n_calls = value
            entry["fd_grad"] = dict(status=status, time_s=elapsed, grad=fd_grad.tolist(),
                                     n_calls=n_calls, numdifftools_opts="defaults (num_steps=15)")
        else:
            entry["fd_grad"] = dict(status=status, time_s=elapsed, detail=str(value))
    elif stage == "fd_hess":
        if status == "ok":
            fd_hess, n_calls = value
            entry["fd_hess"] = dict(status=status, time_s=elapsed, hess=fd_hess.tolist(),
                                     n_calls=n_calls,
                                     numdifftools_opts=("num_steps=9 (capped below the default 15 to bound "
                                                         "worst-case wall-clock; num_steps<9 gave badly "
                                                         "unconverged Richardson extrapolation for this "
                                                         "Hessian -- verified directly, not assumed)"))
        else:
            entry["fd_hess"] = dict(status=status, time_s=elapsed, detail=str(value))
    elif stage in ("naive_qda_ground_truth", "naive_qda_default_tol"):
        fkey = "ground_truth" if stage == "naive_qda_ground_truth" else "default_tol"
        naive_qda = entry.setdefault("naive_qda", {})
        if status == "ok":
            err_f, grad = value  # hess=False here, so norm_err returns (err, grad)
            tol = ("AbsTol=1e-12, RelTol=1e-10, n_ruben=5000" if fkey == "ground_truth"
                   else "AbsTol=1e-10, RelTol=1e-6 (package defaults)")
            naive_qda[fkey] = dict(status=status, time_s=elapsed, err=err_f,
                                    grad=grad_flat(grad, D).tolist(), tol=tol)
        else:
            naive_qda[fkey] = dict(status=status, time_s=elapsed, detail=str(value))
    elif stage == "naive_qda_fd_grad":
        naive_qda = entry.setdefault("naive_qda", {})
        if status == "ok":
            fd_grad, n_calls = value
            naive_qda["fd_grad"] = dict(status=status, time_s=elapsed, grad=fd_grad.tolist(),
                                         n_calls=n_calls, numdifftools_opts="defaults (num_steps=15)")
        else:
            naive_qda["fd_grad"] = dict(status=status, time_s=elapsed, detail=str(value))


def finalize_problem(entry):
    """All of this problem's stages have returned -- compute the cross-stage
    errors/speeds that need ground truth, for both the optimal-boundary
    Hessian test (entry["summary"]) and the naive-QDA-boundary gradient test
    (entry["naive_qda"]["summary"])."""
    gt = entry.get("ground_truth", {})
    grad_gt_flat = np.array(gt["grad"]) if gt.get("status") == "ok" else None
    hess_gt_flat = np.array(gt["hess"]) if gt.get("status") == "ok" else None

    dt = entry.get("default_tol", {})
    analytic_err_grad = analytic_err_hess = None
    if dt.get("status") == "ok":
        if grad_gt_flat is not None:
            analytic_err_grad = _safe_max_abs_diff(dt["grad"], grad_gt_flat)
        if hess_gt_flat is not None:
            analytic_err_hess = _safe_max_abs_diff(dt["hess"], hess_gt_flat)
        dt["analytic_err_grad"] = analytic_err_grad
        dt["analytic_err_hess"] = analytic_err_hess

    fg = entry.get("fd_grad", {})
    fd_err_grad = None
    if fg.get("status") == "ok":
        if grad_gt_flat is not None:
            fd_err_grad = float(np.max(np.abs(np.array(fg["grad"]) - grad_gt_flat)))
        fg["err"] = fd_err_grad

    fh = entry.get("fd_hess", {})
    fd_err_hess = None
    if fh.get("status") == "ok":
        if hess_gt_flat is not None:
            fd_err_hess = float(np.max(np.abs(np.array(fh["hess"]) - hess_gt_flat)))
        fh["err"] = fd_err_hess

    t_default = dt.get("time_s") if dt.get("status") == "ok" else None
    t_fd_grad = fg.get("time_s") if fg.get("status") == "ok" else None
    t_fd_hess = fh.get("time_s") if fh.get("status") == "ok" else None

    # Sum of every stage's own time_s (now including the 3 naive-QDA stages
    # below) -- since every stage of a problem now runs back to back in the
    # one worker that owns it, this sum IS that problem's observed
    # wall-clock, unlike under the old stage-level scheduler where a
    # problem's stages could run on different workers at overlapping times.
    naive_qda = entry.get("naive_qda", {})
    entry["summary"] = dict(
        total_stage_time_s=(sum(s.get("time_s", 0.0) for s in (gt, dt, fg, fh))
                             + sum(s.get("time_s", 0.0) for s in naive_qda.values() if isinstance(s, dict))),
        rel_speed_grad=(t_fd_grad / t_default if t_default and t_fd_grad else None),
        rel_speed_hess=(t_fd_hess / t_default if t_default and t_fd_hess else None),
        rel_acc_grad=(fd_err_grad / analytic_err_grad
                      if fd_err_grad is not None and analytic_err_grad else None),
        rel_acc_hess=(fd_err_hess / analytic_err_hess
                      if fd_err_hess is not None and analytic_err_hess else None),
    )

    # Same cross-stage bookkeeping, but for the naive-QDA-boundary
    # gradient-only test: no Hessian, so no *_hess fields.
    if naive_qda:
        fgt = naive_qda.get("ground_truth", {})
        naive_qda_grad_gt = np.array(fgt["grad"]) if fgt.get("status") == "ok" else None

        fdt = naive_qda.get("default_tol", {})
        naive_qda_analytic_err_grad = None
        if fdt.get("status") == "ok" and naive_qda_grad_gt is not None:
            naive_qda_analytic_err_grad = float(np.max(np.abs(np.array(fdt["grad"]) - naive_qda_grad_gt)))
            fdt["analytic_err_grad"] = naive_qda_analytic_err_grad

        ffg = naive_qda.get("fd_grad", {})
        naive_qda_fd_err_grad = None
        if ffg.get("status") == "ok" and naive_qda_grad_gt is not None:
            naive_qda_fd_err_grad = float(np.max(np.abs(np.array(ffg["grad"]) - naive_qda_grad_gt)))
            ffg["err"] = naive_qda_fd_err_grad

        t_naive_qda_default = fdt.get("time_s") if fdt.get("status") == "ok" else None
        t_naive_qda_fd = ffg.get("time_s") if ffg.get("status") == "ok" else None
        naive_qda["summary"] = dict(
            rel_speed_grad=(t_naive_qda_fd / t_naive_qda_default if t_naive_qda_default and t_naive_qda_fd else None),
            rel_acc_grad=(naive_qda_fd_err_grad / naive_qda_analytic_err_grad
                          if naive_qda_fd_err_grad is not None and naive_qda_analytic_err_grad else None),
        )


# ---------------------------------------------------------------------------
# Per-problem worker: runs one problem's entire pipeline -- boundary setup,
# then all 7 stages one after another (4 for the optimal-boundary Hessian
# test, 3 for the naive-QDA-boundary gradient test) -- inside a single
# ProcessPoolExecutor worker. Parallelism is across problems only (see the
# module docstring): each stage still gets its own nested subprocess via
# run_in_subprocess (for warning capture, not timeout), but stages of the
# *same* problem are never fanned out to different workers, and this worker
# itself is pinned to one thread (the env vars at the top of this file).
# Writes result_<name>.json to disk itself after every stage -- safe here
# (unlike the old stage-level design) since this is the only process that
# will ever touch this problem's file, for that problem's whole lifetime.
# ---------------------------------------------------------------------------
def _run_problem(problem, outdir):
    name = problem["name"]
    log_path = os.path.join(outdir, f"log_{name}.log")
    result_path = os.path.join(outdir, f"result_{name}.json")

    def log(msg):
        with open(log_path, "a") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    mu0 = np.atleast_1d(np.asarray(problem["mu0"], dtype=float))
    v0 = np.atleast_2d(np.asarray(problem["v0"], dtype=float))
    mu1 = np.atleast_1d(np.asarray(problem["mu1"], dtype=float))
    v1 = np.atleast_2d(np.asarray(problem["v1"], dtype=float))
    p0 = problem.get("p0", P0)
    p1 = problem.get("p1", P1)
    D = mu0.size
    P = n_params(D)
    result = dict(name=name, D=D, P=P, mu0=mu0.tolist(), v0=v0.tolist(),
                  mu1=mu1.tolist(), v1=v1.tolist(), p0=p0, p1=p1)
    _write_partial(result_path, result)
    log(f"starting, D={D}, P={P}")

    try:
        quad = gx2.norm_class_opt_bd(mu0, v0, mu1, v1, p0=p0, p1=p1)
        theta0 = flatten(quad, D)
        quad_naive_qda = naive_qda_boundary(mu0, v0, mu1, v1, p0=p0, p1=p1)
        theta0_naive_qda = flatten(quad_naive_qda, D)
        result["quad"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in quad.items()}
        result["quad_naive_qda"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                     for k, v in quad_naive_qda.items()}
        _write_partial(result_path, result)
        log("boundaries computed (optimal + naive-QDA)")
    except Exception:
        result["error"] = traceback.format_exc()
        _write_partial(result_path, result)
        log("PROBLEM_FAILED (boundary setup):\n" + result["error"])
        return name

    tight = dict(AbsTol=1e-12, RelTol=1e-10, n_ruben=5000)
    stage_defs = [
        ("ground_truth", _stage_analytic, (mu0, v0, mu1, v1, quad, p0, p1, tight, True)),
        ("default_tol", _stage_analytic, (mu0, v0, mu1, v1, quad, p0, p1, {}, True)),
        ("fd_grad", _stage_fd_grad, (theta0, mu0, v0, mu1, v1, D, p0, p1, log_path)),
        ("fd_hess", _stage_fd_hess, (theta0, mu0, v0, mu1, v1, D, p0, p1, log_path, 9)),
        ("naive_qda_ground_truth", _stage_analytic, (mu0, v0, mu1, v1, quad_naive_qda, p0, p1, tight, False)),
        ("naive_qda_default_tol", _stage_analytic, (mu0, v0, mu1, v1, quad_naive_qda, p0, p1, {}, False)),
        ("naive_qda_fd_grad", _stage_fd_grad, (theta0_naive_qda, mu0, v0, mu1, v1, D, p0, p1, log_path)),
    ]

    try:
        for stage, func, args in stage_defs:
            t0 = time.perf_counter()
            status, value = run_in_subprocess(func, args, tag=stage, warn_log_path=log_path)
            elapsed = time.perf_counter() - t0
            _store_stage_result(result, stage, status, value, elapsed, D)
            _write_partial(result_path, result)
            log(f"{stage}: {status} in {elapsed:.2f}s")
        finalize_problem(result)
    except Exception:
        result["error"] = traceback.format_exc()
        log("PROBLEM_FAILED (mid-run):\n" + result["error"])

    _write_partial(result_path, result)
    log("PROBLEM_DONE")
    return name


# ---------------------------------------------------------------------------
# Main: submit one task per problem. GPU is not used -- the numerical
# bottleneck (scipy.integrate.quad_vec / mpmath series) has no GPU path in
# gx2-py, and a from-scratch vectorized rewrite of the integrator was
# already attempted and reverted upstream (see gx2_derivatives.md open item
# 3.6).
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    n_workers = os.cpu_count()
    meta = dict(
        gx2_version=getattr(gx2, "__version__", "unknown"),
        numpy_version=np.__version__,
        cpu_count=os.cpu_count(),
        n_workers=n_workers,
        parallelism_note=(f"problem-level: each of the {len(PROBLEMS)} problems' full 7-stage "
                           f"pipeline (4 optimal-boundary Hessian-test stages, 3 naive-QDA-boundary "
                           f"gradient-test stages) runs sequentially inside one "
                           f"ProcessPoolExecutor(max_workers={n_workers}) worker, pinned to a "
                           "single thread (OMP/OpenBLAS/MKL/numexpr num_threads=1, set before "
                           "numpy is imported in every process this script spawns). See the "
                           "module docstring for why, and for the throughput this trades away."),
        gpu_note=("Not used: the numerical bottleneck is scipy.integrate.quad_vec "
                   "and mpmath series (CPU-only, no GPU path in gx2-py); a GPU-"
                   "vectorized rewrite was already attempted and reverted "
                   "(gx2_derivatives.md open item 3.6)."),
    )
    with open(os.path.join(OUTDIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Placeholder result files for every problem, written up front (before
    # any worker has necessarily started on it) so the pending set is
    # visible on disk immediately; _run_problem overwrites its own with a
    # fuller version as soon as it actually starts.
    for problem in PROBLEMS:
        D = np.atleast_1d(np.asarray(problem["mu0"], dtype=float)).size
        placeholder = dict(name=problem["name"], D=D, P=n_params(D),
                            mu0=problem["mu0"], v0=problem["v0"],
                            mu1=problem["mu1"], v1=problem["v1"],
                            p0=problem.get("p0", P0), p1=problem.get("p1", P1))
        _write_partial(os.path.join(OUTDIR, f"result_{problem['name']}.json"), placeholder)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_problem, problem, OUTDIR): problem["name"]
                   for problem in PROBLEMS}
        n_done, n_total = 0, len(futures)
        for fut in as_completed(futures):
            name = futures[fut]
            n_done += 1
            try:
                fut.result()
                print(f"[{n_done}/{n_total}] {name}: done")
            except Exception:
                print(f"[{n_done}/{n_total}] {name}: WORKER CRASHED\n{traceback.format_exc()}",
                      file=sys.stderr)

    # aggregation: glob whatever per-problem result files exist, so this
    # still produces a combined summary even if some problems didn't finish.
    combined = {"meta": meta, "problems": {}}
    for problem in PROBLEMS:
        path = os.path.join(OUTDIR, f"result_{problem['name']}.json")
        if os.path.exists(path):
            with open(path) as fh:
                combined["problems"][problem["name"]] = json.load(fh)
        else:
            combined["problems"][problem["name"]] = {"status": "not started"}
    with open(os.path.join(OUTDIR, "bench_norm_err_bd_results.json"), "w") as fh:
        json.dump(combined, fh, indent=2, default=float)
    print(f"Done. Combined results: "
          f"{os.path.join(OUTDIR, 'bench_norm_err_bd_results.json')}")


if __name__ == "__main__":
    main()
