<p align="center">
  <img src="https://raw.githubusercontent.com/abhranildas/gx2-py/main/gx2_icon.png" alt="gx2" width="260">
</p>

# gx2 — Generalized chi-square distribution

`gx2` is a python package that computes the statistics, characteristic function, pdf, cdf, inverse cdf
and random numbers of the **generalized chi-square distribution**.
This is the python port of the
[MATLAB toolbox](https://www.mathworks.com/matlabcentral/fileexchange/85028-generalized-chi-square-distribution).

## Author and citation

Abhranil Das, Center for Perceptual Systems, The University of Texas at Austin.
Bugs / comments / questions / suggestions to abhranil.das@utexas.edu.

If you use this code, please cite:
 - [Methods to integrate multinormals and compute classification measures](https://arxiv.org/abs/2012.14331)
 - New methods to compute the generalized chi-square distribution: [journal](https://www.tandfonline.com/doi/abs/10.1080/00949655.2025.2501401) / [arxiv](https://arxiv.org/abs/2404.05062)

A generalized chi-square variable is a weighted sum of independent non-central
chi-square variables plus a normal variable — equivalently, the quadratic form
of a normal random vector. It is parametrized by:

| parameter | meaning |
|-----------|---------|
| `w`       | weights of the non-central chi-square terms |
| `k`       | their degrees of freedom |
| `l` | their non-centralities (named `l` because `lambda` is a Python keyword) |
| `s`       | scale (standard deviation) of the added normal term |
| `m`       | constant offset |

## Installation

```bash
pip install gx2
```

Requires `numpy`, `scipy` and `mpmath`. `matplotlib` is optional, for plotting
in the getting-started notebook.

To install from a local clone instead:

```bash
pip install .
# or, for development (editable install with test/plot extras):
pip install -e ".[plot,test]"
```

## Getting started

```python
import gx2

w, k, l, s, m = [1, -5, 2], [1, 2, 3], [2, 3, 7], 0, 5

gx2.stat(w, k, l, s, m)          # mean and variance
gx2.cdf(25, w, k, l, s, m)       # cdf at x = 25
gx2.pdf(25, w, k, l, s, m)       # pdf at x = 25
gx2.inv(0.9, w, k, l, s, m)      # 90th percentile
gx2.rnd(w, k, l, s, m, size=5)   # random numbers
```

Open [`GettingStarted.ipynb`](GettingStarted.ipynb) for an interactive tour
with worked examples and plots. For any function, see its full documentation
with `help(gx2.cdf)` (or `gx2.cdf?` in Jupyter).

## Public functions

| function | purpose |
|----------|---------|
| `stat(w, k, l, s, m)` | mean and variance |
| `char(t, w, k, l, s, m)` | characteristic function |
| `rnd(w, k, l, s, m, size=, method=)` | random numbers |
| `cdf(x, w, k, l, s, m, side=, method=, ...)` | cdf |
| `pdf(x, w, k, l, s, m, side=, method=, ...)` | pdf |
| `inv(p, w, k, l, s, m, side=, method=, ...)` | inverse cdf |
| `gx2_to_norm_quad_params(w, k, l, s, m)` | gx2 → quadratic-form coefficients of a standard normal |
| `norm_quad_to_gx2_params(mu, v, quad, merge=)` | quadratic form of a normal → gx2 parameters |

The individual computation routines (`imhof`, `ruben`, `ifft`, `pearson`,
`tail`, `ellipse`, `cdf_ray`, `pdf_ray`, …) and numerical helpers
(`log_sum_exp`, `signed_log_sum_exp`, `phi_ray`, …) are also exposed.

## Computation methods for `cdf` / `pdf`

`method='auto'` (default) picks a good method for the given parameters. You can
also force one:

| method | notes |
|--------|-------|
| `'imhof'`   | Imhof–Davies numerical integration (`precision='basic'` or `'vpa'`) |
| `'ray'`     | ray-trace method (`precision='basic'`, `'log'` or `'vpa'`; tune with `n_rays`, `force_mc`) |
| `'ifft'`    | inverse-FFT method; `x='full'` returns the cdf/pdf over a spanning grid |
| `'ruben'`   | Ruben's series — requires all `w` the same sign and `s=0` |
| `'tail'`    | infinite-tail approximation |
| `'pearson'` | Pearson's 3-moment approximation |
| `'ellipse'` | ellipse approximation near a finite tail — requires all `w` the same sign and `s=0` |

## Usage notes

* `cdf` and `pdf` return just the probability/density by default. Pass
  `full_output=True` (auto-enabled for `x='full'`) to also receive the error
  estimate and, for `x='full'`, the grid of x-values.
* In the far tails, probabilities can fall below double precision (~1e-308).
  The `'tail'`, `'ellipse'` and `'ray'` methods then return the **base-10
  logarithm** of such values (a negative number); `inv` likewise accepts a
  negative `p` as a log10 probability. `precision='log'` (the ray default) is
  the easiest way to reach this regime.
* The `'ray'` method runs on the CPU with NumPy and batches automatically over
  rays to bound memory. `precision='vpa'` uses `mpmath` and returns
  `mpmath.mpf` objects for sub-`realmin` values.

## License

MIT — see [LICENSE](LICENSE).
