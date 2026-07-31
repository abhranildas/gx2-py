"""Wider benchmark of ``gx2.norm_err_grad_bd`` (analytic gradient/Hessian of
total two-class classification error w.r.t. the boundary coefficients) on
real binary-Gaussian classification problems spanning D=1..5.

For each problem this computes the optimal quadratic (Bayes) boundary, then
compares three ways of getting the gradient/Hessian of the classification
error w.r.t. that boundary's coefficients (q2,q1,q0):
  - a tight-tolerance analytic evaluation, treated as ground truth;
  - the default-tolerance analytic evaluation (production settings);
  - finite differences via ``numdifftools`` (Richardson-extrapolated).

Some problems are deliberately included *because* one or both methods may be
slow there (mixed-sign, higher-dimension boundaries) -- the point of this
benchmark is partly to demonstrate exactly that contrast, not to avoid it.
Each of the four measured stages (ground truth, default-tol, FD gradient, FD
Hessian) runs in its own subprocess with a wall-clock timeout
(``STAGE_TIMEOUT_S``); a stage that doesn't finish in time is recorded as
"timeout" rather than blocking the run indefinitely.

Designed to run on a separate, faster, multi-core machine. Parallelism is at
the *stage* level, not the problem level: all 4 stages of every problem
are submitted to one ``ProcessPoolExecutor`` up front, so a
worker that finishes a millisecond-long stage (e.g. a D=1 problem's default-
tolerance analytic Hessian) immediately pulls the next queued stage from
*anywhere* in the whole set, rather than idling once its own problem's
4-stage pipeline happens to be briefly done while a handful of slow problems
(mixed-sign, higher-D) still run for hours. Every stage is independent of
every other stage (a problem's 4 stages don't feed into each other -- they
each just need that problem's boundary, computed once up front); only the
final error/speed comparison needs a problem's ground-truth stage to have
finished, so that comparison is deferred until all 4 of a problem's stages
have returned, however they interleave with everyone else's.
Results are written incrementally (one JSON file per problem, updated after
each stage returns) so a killed or crashed run still leaves usable
diagnostics for whatever finished. All such writes happen in the main
process only (never inside a worker), so there's no multi-process race on
a problem's result file even though its 4 stages may finish on 4 different
workers at 4 different times.

Any warning raised while computing a stage (numpy overflow/invalid-value,
scipy's IntegrationWarning, gx2's own ImhofClipWarning, ...) is redirected
into that problem's own log_<name>.log, tagged with which of the 4 stages
raised it, instead of going to stderr where it would be interleaved with
every other worker's output and impossible to attribute afterward.

Deliberately NOT using ``joblib`` for the outer parallelism: joblib's default
``loky`` backend monkeypatches the global multiprocessing context on Windows
in a way that breaks the plain ``multiprocessing.Process`` objects used below
for per-stage timeouts (nesting the two raises
``AttributeError: 'Process' object has no attribute 'env'`` -- caught directly
while testing this script, not a theoretical concern). The standard library's
``ProcessPoolExecutor`` doesn't have this problem.

Requires, in addition to gx2's own dependencies: numdifftools.
    pip install numdifftools

Usage:
    python bench_norm_err_bd.py [output_dir]
"""
import json
import os
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

STAGE_TIMEOUT_S = 1800  # 30 min per stage. Generous enough for legitimately
                         # slow-but-finishing cases; bounded enough that a
                         # stage that's genuinely impractical (the point being
                         # demonstrated for some problems below) gets recorded
                         # as "timeout" instead of blocking the whole run.

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
    # and the cusp handling in cdf_grad_norm_quad/norm_err_grad_bd).
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
# Optimal quadratic (Bayes) boundary between two Gaussians. Region q(x)>0
# decides class 1. Matches IntClassNorm's opt_class_quad.m (class1<->norm_1,
# class0<->norm_2) -- cross-checked against it and against finite differences
# of the classification error itself (the gradient/Hessian of the error must
# vanish/be positive-definite at this boundary, since it's the error
# minimizer; both checks passed during development).
# ---------------------------------------------------------------------------
def opt_boundary(mu0, v0, mu1, v1, p0=P0, p1=P1):
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


# ---------------------------------------------------------------------------
# Flattening: theta = [vech(Q2) (upper triangle incl. diagonal, row-major),
# q1, q0]. unflatten's off-diagonal entries mirror both (a,b) and (b,a)
# together, matching cdf_grad_norm_quad's symmetric-perturbation convention
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
    """Flatten a cdf_grad_norm_quad-style gradient dict to the theta basis."""
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
    """Flatten a cdf_grad_norm_quad-style Hessian dict to the theta basis."""
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
    quad = unflatten(theta, D)
    w0, k0, l0, s0, m0 = gx2.norm_quad_to_gx2_params(mu0, v0, quad)
    w1, k1, l1, s1, m1 = gx2.norm_quad_to_gx2_params(mu1, v1, quad)
    F0 = gx2.cdf(0, w0, k0, l0, s0, m0)
    F1 = gx2.cdf(0, w1, k1, l1, s1, m1)
    return p0 * (1 - F0) + p1 * F1


# ---------------------------------------------------------------------------
# Per-stage timeout. Each stage runs in its own subprocess so a genuinely
# impractical case (some of these problems are here specifically to show
# that) can be killed on a wall-clock budget instead of blocking the run.
# Stage functions are plain top-level functions (not closures) so they can
# be pickled and sent to the subprocess; any closures they need internally
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


def run_with_timeout(func, args, timeout_s=STAGE_TIMEOUT_S, tag=None, warn_log_path=None):
    """Run func(*args) in a subprocess; returns (status, value) with status
    in {'ok', 'error', 'timeout'}. Terminates the subprocess on timeout."""
    q = mp.Queue()
    p = mp.Process(target=_mp_call, args=(q, func, args, tag, warn_log_path))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout", None
    try:
        return q.get(timeout=5)
    except queue.Empty:
        return "error", "worker exited without producing a result"


def _stage_analytic(mu0, v0, mu1, v1, quad, p0, p1, tol_kwargs):
    return gx2.norm_err_grad_bd(mu0, v0, mu1, v1, quad, p0=p0, p1=p1, hess=True, **tol_kwargs)


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


def _run_stage_task(name, stage, func, args, timeout_s, log_path):
    """Top-level, picklable wrapper run inside a ProcessPoolExecutor worker:
    times run_with_timeout(func, args) and tags the result with which
    (problem, stage) it belongs to, so the main process's as_completed loop
    can route it without needing the tasks to finish in submission order.
    Also routes any warnings raised during this stage into the problem's own
    log file, tagged with the stage name (see _mp_call)."""
    t0 = time.perf_counter()
    status, value = run_with_timeout(func, args, timeout_s=timeout_s, tag=stage, warn_log_path=log_path)
    elapsed = time.perf_counter() - t0
    return name, stage, status, value, elapsed


# ---------------------------------------------------------------------------
# Build the flat list of 4*len(PROBLEMS) independent stage-tasks. A problem's
# boundary (opt_boundary/flatten) is cheap and deterministic, so it's
# computed once here in the main process, not inside a worker.
# ---------------------------------------------------------------------------
def build_tasks(outdir, timeout_s):
    tasks = []              # list of (name, stage, func, args, timeout_s)
    results = {}            # name -> partial result dict, filled in as stages return
    remaining = {}          # name -> count of stages not yet returned
    log_paths = {}          # name -> its log file path

    for problem in PROBLEMS:
        name = problem["name"]
        log_path = os.path.join(outdir, f"log_{name}.log")
        log_paths[name] = log_path

        def log(msg, log_path=log_path):
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
        log(f"starting, D={D}, P={P}")

        try:
            quad = opt_boundary(mu0, v0, mu1, v1, p0=p0, p1=p1)
            theta0 = flatten(quad, D)
            result["quad"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in quad.items()}
            log("boundary computed")
            tasks.append((name, "ground_truth", _stage_analytic,
                          (mu0, v0, mu1, v1, quad, p0, p1,
                           dict(AbsTol=1e-12, RelTol=1e-10, n_ruben=5000))))
            tasks.append((name, "default_tol", _stage_analytic,
                          (mu0, v0, mu1, v1, quad, p0, p1, {})))
            tasks.append((name, "fd_grad", _stage_fd_grad,
                          (theta0, mu0, v0, mu1, v1, D, p0, p1, log_path)))
            tasks.append((name, "fd_hess", _stage_fd_hess,
                          (theta0, mu0, v0, mu1, v1, D, p0, p1, log_path, 9)))
            remaining[name] = 4
        except Exception:
            result["error"] = traceback.format_exc()
            remaining[name] = 0
            log("PROBLEM_FAILED (boundary setup):\n" + result["error"])

        results[name] = result

    return tasks, results, remaining, log_paths


def handle_stage_result(name, stage, status, value, elapsed, results):
    """Store one stage's raw (flattened) output into results[name]. Doesn't
    compute cross-stage errors yet -- stages can return in any order, so
    that's deferred to finalize_problem once all 4 have arrived."""
    D = results[name]["D"]
    entry = results[name]
    if stage == "ground_truth":
        if status == "ok":
            grad_gt, hess_gt = value
            entry["ground_truth"] = dict(status=status, time_s=elapsed,
                                          grad=grad_flat(grad_gt, D).tolist(),
                                          hess=hess_flat(hess_gt, D).tolist(),
                                          tol="AbsTol=1e-12, RelTol=1e-10, n_ruben=5000")
        else:
            entry["ground_truth"] = dict(status=status, time_s=elapsed, detail=str(value))
    elif stage == "default_tol":
        if status == "ok":
            grad_d, hess_d = value
            entry["default_tol"] = dict(status=status, time_s=elapsed,
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


def finalize_problem(name, results):
    """All 4 stages of this problem have returned (in whatever order) --
    compute the cross-stage errors/speeds that need ground truth, exactly
    the same metrics run_problem used to compute inline."""
    entry = results[name]
    gt = entry.get("ground_truth", {})
    grad_gt_flat = np.array(gt["grad"]) if gt.get("status") == "ok" else None
    hess_gt_flat = np.array(gt["hess"]) if gt.get("status") == "ok" else None

    dt = entry.get("default_tol", {})
    analytic_err_grad = analytic_err_hess = None
    if dt.get("status") == "ok":
        if grad_gt_flat is not None:
            analytic_err_grad = float(np.max(np.abs(np.array(dt["grad"]) - grad_gt_flat)))
        if hess_gt_flat is not None:
            analytic_err_hess = float(np.max(np.abs(np.array(dt["hess"]) - hess_gt_flat)))
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

    # Sum of the 4 stages' own time_s. Under the old one-worker-per-problem
    # design this sum WAS the problem's observed wall-clock (stages ran back
    # to back in one process); under stage-level scheduling the 4 stages can
    # run on different workers at overlapping times, so this number is no
    # longer directly observable as a single elapsed duration -- it's
    # reconstructed here instead, so "how long would problem X take run in
    # isolation, stage by stage" is still available even though it may now
    # differ from this run's actual wall-clock for that problem.
    entry["summary"] = dict(
        total_stage_time_s=sum(s.get("time_s", 0.0) for s in (gt, dt, fg, fh)),
        rel_speed_grad=(t_fd_grad / t_default if t_default and t_fd_grad else None),
        rel_speed_hess=(t_fd_hess / t_default if t_default and t_fd_hess else None),
        rel_acc_grad=(fd_err_grad / analytic_err_grad
                      if fd_err_grad is not None and analytic_err_grad else None),
        rel_acc_hess=(fd_err_hess / analytic_err_hess
                      if fd_err_hess is not None and analytic_err_hess else None),
    )


# ---------------------------------------------------------------------------
# Main: submit every problem's every stage as one flat pool of tasks. GPU is
# not used -- the numerical bottleneck (scipy.integrate.quad_vec / mpmath
# series) has no GPU path in gx2-py, and a from-scratch vectorized rewrite of
# the integrator was already attempted and reverted upstream (see
# gx2_derivatives.md open item 3.6). The lever that's actually available is
# keeping every core fed with the next queued stage -- see the module
# docstring for why that's done at stage granularity, not problem granularity.
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    n_workers = os.cpu_count()
    meta = dict(
        gx2_version=getattr(gx2, "__version__", "unknown"),
        numpy_version=np.__version__,
        cpu_count=os.cpu_count(),
        n_workers=n_workers,
        stage_timeout_s=STAGE_TIMEOUT_S,
        parallelism_note=(f"stage-level: all 4 stages x {len(PROBLEMS)} problems "
                           f"({4 * len(PROBLEMS)} tasks) submitted to one "
                           f"ProcessPoolExecutor(max_workers={n_workers}) up front, so "
                           "a worker that finishes a fast stage immediately dequeues "
                           "the next one from anywhere in the set instead of idling."),
        gpu_note=("Not used: the numerical bottleneck is scipy.integrate.quad_vec "
                   "and mpmath series (CPU-only, no GPU path in gx2-py); a GPU-"
                   "vectorized rewrite was already attempted and reverted "
                   "(gx2_derivatives.md open item 3.6)."),
    )
    with open(os.path.join(OUTDIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    tasks, results, remaining, log_paths = build_tasks(OUTDIR, STAGE_TIMEOUT_S)
    for name in results:
        _write_partial(os.path.join(OUTDIR, f"result_{name}.json"), results[name])

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_stage_task, name, stage, func, args, STAGE_TIMEOUT_S, log_paths[name]): (name, stage)
                   for name, stage, func, args in tasks}
        n_done, n_total = 0, len(futures)
        for fut in as_completed(futures):
            name, stage, status, value, elapsed = fut.result()
            n_done += 1
            handle_stage_result(name, stage, status, value, elapsed, results)
            remaining[name] -= 1
            if remaining[name] == 0:
                finalize_problem(name, results)
            _write_partial(os.path.join(OUTDIR, f"result_{name}.json"), results[name])
            with open(log_paths[name], "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] {stage}: {status} in {elapsed:.2f}s"
                          f"{' -- PROBLEM_DONE' if remaining[name] == 0 else ''}\n")
            print(f"[{n_done}/{n_total}] {name} {stage}: {status} ({elapsed:.2f}s)")

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
