"""
Core Functions

gx2_params_norm_quad: Transforms normal parameters and a quadratic form into a generalized chi-squared distribution's parameters.
gx2cdf_davies: Computes the CDF using Davies' method.
gx2cdf_imhof: Computes the CDF using Imhof's method.
gx2cdf_ruben: Computes the CDF using Ruben's method.
gx2cdf: Generalized function to compute the CDF using the most appropriate method.
gx2inv: Finds the inverse CDF.
gx2pdf: Computes the PDF either by differentiation or convolution.
gx2rnd: Generates random numbers from the generalized chi-squared distribution.
gx2stat: Returns the mean and variance.
"""

import numpy as np
from scipy.stats import ncx2, chi2, norm
from scipy.integrate import quad
from scipy.fft import fft, ifft, ifftshift
from scipy.linalg import sqrtm
from scipy.special import logsumexp
from scipy.optimize import root
from typing import Tuple, Union


# Function to calculate the mean and variance of a generalized chi-squared variable
def gx2stat(w, k, lambda_, m, s):
    """
    Returns the mean and variance of a generalized chi-squared variable (a weighted sum of non-central chi-squares).

    Required inputs:
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality parameters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term
    s         sd of normal term

    Outputs:
    mu        mean
    v         variance
    """
    
    # Validating inputs, making sure they are numpy arrays and have the right dimensions
    if not (isinstance(w, np.ndarray) and w.ndim == 1):
        raise ValueError("w must be a row vector (1D numpy array)")
    if not (isinstance(k, np.ndarray) and k.ndim == 1):
        raise ValueError("k must be a row vector (1D numpy array)")
    if not (isinstance(lambda_, np.ndarray) and lambda_.ndim == 1):
        raise ValueError("lambda_ must be a row vector (1D numpy array)")
    if not (isinstance(m, (int, float))):
        raise ValueError("m must be a scalar")
    if not (isinstance(s, (int, float))):
        raise ValueError("s must be a scalar")

    w = np.atleast_1d(w)  # Ensure w is an array, even if it's a single value
    k = np.atleast_1d(k)  # Ensure k is an array, even if it's a single value
    lambda_ = np.atleast_1d(lambda_)  # Ensure lambda_ is an array, even if it's a single value

    
    # Calculate mean and variance
    mu = np.dot(w, k + lambda_) + m
    v = 2 * np.dot(w**2, k + 2 * lambda_) + s**2
    
    return mu, v

# # Define the parameters
# w = np.array([1, -10, 2])
# k = np.array([1, 2, 3])
# lambda_ = np.array([2, 3, 7])
# m = 10
# s = 5

# # Calculate mean and variance
# mu, v = gx2stat(w, k, lambda_, m, s)
# mu, v


def gx2rnd(w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, s: float, size: Union[int, Tuple[int, ...]] = 1) -> np.ndarray:
    """
    Generates generalized chi-squared random numbers.

    Required inputs:
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term
    s         sd of normal term

    Optional positional input:
    size      size(s) of the requested array

    Output:
    r         random number(s)
    """
    ncxs = [w_i * ncx2.rvs(k_i, lambda_i, size=size) for w_i, k_i, lambda_i in zip(w, k, lambda_)]
    r = np.sum(ncxs, axis=0) + norm.rvs(m, s, size=size)
    return r

# # Define parameters
# w = np.array([1, -10, 2])
# k = np.array([1, 2, 3])
# lambda_ = np.array([2, 3, 7])
# m = 10
# s = 5
# size = (1, int(1e5))

# # Generate random samples
# r = gx2rnd(w, k, lambda_, m, s, size)
# r.shape, r.mean(), r.std()

# import matplotlib.pyplot as plt

# # Plotting the histogram of generated random numbers
# plt.hist(r[0], bins=50, density=True, histtype='step', label='Sampled Data')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Histogram of Generated Generalized Chi-Squared Random Numbers')
# plt.show()

def gx2_params_norm_quad(mu, v, quad):
    """
    Compute the parameters of a generalized chi-squared distribution based on
    a quadratic form of a normal variable.
    
    This function calculates the parameters of a generalized chi-squared distribution
    based on the given normal parameters (mean and covariance matrix) and quadratic coefficients.
    
    Parameters:
    -----------
    mu : numpy.ndarray
        Column vector of the normal mean.
    v : numpy.ndarray
        Covariance matrix of the normal distribution.
    quad : dict
        Dictionary containing quadratic form coefficients.
        - q2: numpy.ndarray, matrix of quadratic coefficients
        - q1: numpy.ndarray, column vector of linear coefficients
        - q0: float, scalar constant
        
    Returns:
    --------
    w : numpy.ndarray
        Row vector of weights of the non-central chi-squares.
    k : numpy.ndarray
        Row vector of degrees of freedom of the non-central chi-squares.
    lambda : numpy.ndarray
        Row vector of non-centrality parameters (sum of squares of means) of the non-central chi-squares.
    m : float
        Mean of the normal term.
    s : float
        Standard deviation of the normal term.
        
    Example:
    --------
    mu = np.array([1, 2])
    v = np.array([[2, 1], [1, 3]])
    quad = {'q2': np.array([[1, 1], [1, 1]]), 'q1': np.array([-1, 0]), 'q0': -1}
    w, k, lambda_, m, s = gx2_params_norm_quad(mu, v, quad)
    
    See Also:
    ---------
    gx2rnd, gx2stat, gx2cdf, gx2pdf
    """
    # Standardize the space
    q2s = 0.5 * (quad['q2'] + quad['q2'].T)  # Symmetrize q2
    q2 = np.dot(sqrtm(v), np.dot(q2s, sqrtm(v)))
    q1 = np.dot(sqrtm(v), (2 * np.dot(q2s, mu) + quad['q1']))
    q0 = mu.T @ q2s @ mu + quad['q1'].T @ mu + quad['q0']
    
    # Eigenvalue decomposition
    D, R = np.linalg.eig(q2)
    d = np.real(D)
    b = np.real(R.T @ q1)
    
    # Unique non-zero eigenvalues and their counts (degrees of freedom)
    uniquetol_d, counts = np.unique(np.round(d, 8), return_counts=True)
    w = uniquetol_d[uniquetol_d != 0]
    k = counts[uniquetol_d != 0]
    
    # Total non-centrality for each eigenvalue
    lambda_ = np.array([np.sum(b[np.isclose(d, x, atol=1e-8)]**2) for x in w]) / (4 * w**2)
    
    # Remaining parameters
    m = q0 - np.dot(w, lambda_)
    s = np.linalg.norm(b[np.isclose(d, 0, atol=1e-8)])
    
    return w, k, lambda_, m, s


def gx2inv(p: Union[float, np.ndarray], w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, s: float, AbsTol: float = 1e-10, RelTol: float = 1e-6, side="lower") -> np.ndarray:
    """
    Returns the inverse cdf of a generalized chi-squared, using Ruben's [1962] method, Davies' [1973] method, or the native ncx2inv, depending on the input.

    Required inputs:
    p         probabilities at which to evaluate the inverse cdf
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term
    s         sd of normal term

    Optional name-value inputs:
    'AbsTol'  absolute error tolerance for the cdf function that is inverted
    'RelTol'  relative error tolerance for the cdf function that is inverted
               The absolute OR the relative tolerance is satisfied.
    'side'    "lower" or "upper", when "upper" calculate the inverse survival function

    Output:
    x         computed point
    """
    w = np.atleast_1d(w)  # Ensure w is an array, even if it's a single value
    k = np.atleast_1d(k)  # Ensure k is an array, even if it's a single value
    lambda_ = np.atleast_1d(lambda_)  # Ensure lambda_ is an array, even if it's a single value

    if s == 0 and len(set(w)) == 1:
        # native ncx2 fallback
        if np.sign(np.unique(w)[0]) == 1:
            if side.lower == "lower":
                x = ncx2.ppf(p, sum(k), sum(lambda_)) * np.unique(w) + m
            else:
                x = ncx2.isf(p, sum(k), sum(lambda_)) * np.unique(w) + m
        elif np.sign(np.unique(w)[0]) == -1:
            if side.lower == "lower":
                x = ncx2.ppf(1 - p, sum(k), sum(lambda_)) * np.unique(w) + m
            else:
                x = ncx2.isf(1 - p, sum(k), sum(lambda_)) * np.unique(w) + m
    else:
        mu = gx2stat(w, k, lambda_, s, m)[0]
        x = np.array([root(lambda x: gx2cdf(x, w, k, lambda_, m, s, AbsTol=AbsTol, RelTol=RelTol, side=side) - p_val, mu).x[0] for p_val in np.atleast_1d(p)])

    return x


def gx2pdf(x: Union[str, np.ndarray], w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, s: float, method: str = 'diff', AbsTol: float = 1e-10, RelTol: float = 1e-6, dx: float = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns the pdf of a generalized chi-squared (a weighted sum of non-central chi-squares and a normal).

    Required inputs:
    x         points at which to evaluate the pdf, or 'full', to return points xfull and f over a full range of the pdf (this uses the 'conv' method)
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term
    s         sd of normal term

    Optional name-value inputs:
    method    'diff' (default) for differentiating the generalized chi-square cdf, or 'conv' for convolving noncentral chi-square pdf's
    dx        step-size of fineness (for convolving or differentiating)
    AbsTol    absolute error tolerance for the output when using 'diff'
    RelTol    relative error tolerance for the output when using 'diff'
               The absolute OR the relative tolerance is satisfied.

    Output:
    f         computed pdf at points, or over full range
    xfull     x-values over the full pdf range, returned when x is 'full'
    """
    w = np.atleast_1d(w)  # Ensure w is an array, even if it's a single value
    k = np.atleast_1d(k)  # Ensure k is an array, even if it's a single value
    lambda_ = np.atleast_1d(lambda_)  # Ensure lambda_ is an array, even if it's a single value

    if dx is None:
        _, v = gx2stat(w, k, lambda_, s, m)
        dx = np.sqrt(v) / 1e4  # default derivative step-size is sd/100.

    if x == 'full':
        method = 'conv'

    if s == 0 and len(set(w)) == 1 and x != 'full':
        f = ncx2.pdf((x - m) / np.unique(w), sum(k), sum(lambda_)) / abs(np.unique(w))
    elif method == 'conv':
        span = np.nan * np.ones(len(w))
        for i in range(len(w)):
            span[i] = gx2inv(1 - np.finfo(float).eps, abs(w[i]), k[i], lambda_[i], 0, 0)
        if s:
            span = np.append(span, 2 * norm.ppf(1 - np.finfo(float).eps, 0, s))
        span = np.max(2 * span)
        span = span - np.mod(span, dx)  # to center around 0
        xfull = np.arange(-span, span + dx, dx)

        ncpdfs = np.empty((len(w), len(xfull)))
        for i in range(len(w)):
            pdf = gx2pdf(xfull, w[i], k[i], lambda_[i], 0, 0)
            pdf[np.isinf(pdf)] = np.max(pdf[~np.isinf(pdf)])
            ncpdfs[i, :] = pdf
        if s:
            ncpdfs = np.vstack([ncpdfs, norm.pdf(xfull, 0, abs(s))])

        f = ifft(np.prod(fft(ncpdfs, axis=1), axis=0))
        if any(np.isnan(f)) or any(np.isinf(f)):
            raise ValueError('Convolution method failed. Try differentiation method.')
        if ncpdfs.shape[0] % 2 == 0:
            f = ifftshift(f)
        f = f / (np.sum(f) * dx)
        xfull = xfull + m

        if x != 'full':
            from scipy.interpolate import interp1d
            F = interp1d(xfull, f, fill_value="extrapolate")
            f = F(x)
    elif method == 'diff':
        p_left = gx2cdf(x - dx, w, k, lambda_, m, s, AbsTol=AbsTol, RelTol=RelTol)
        p_right = gx2cdf(x + dx, w, k, lambda_, m, s, AbsTol=AbsTol, RelTol=RelTol)
        f = (p_right - p_left) / (2 * dx)

    f = np.maximum(f, 0)

    if x == 'full':
        return f, xfull
    else:
        return f


def gx2cdf_davies(x: Union[float, np.ndarray], w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, s: float, side: str = 'lower', AbsTol: float = 1e-10, RelTol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the cdf of a generalized chi-squared (a weighted sum of non-central chi-squares and a normal), 
    using Davies' [1973] method.

    Required inputs:
    x         points at which to evaluate the cdf
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term
    s         sd of normal term

    Optional positional input:
    'upper'   more accurate estimate of the complementary CDF when it's small

    Optional name-value inputs:
    'AbsTol'  absolute error tolerance for the output
    'RelTol'  relative error tolerance for the output
               The absolute OR the relative tolerance is satisfied.

    Outputs:
    p         computed cdf
    flag      =true if output was too close to 0 or 1 to compute exactly with default settings. Try stricter tolerances.
    """
    w = np.atleast_1d(w)  # Ensure w is an array, even if it's a single value
    k = np.atleast_1d(k)  # Ensure k is an array, even if it's a single value
    lambda_ = np.atleast_1d(lambda_)  # Ensure lambda_ is an array, even if it's a single value

    def davies_integrand(u, x, w, k, lambda_, s):
        # theta = np.sum(k * np.arctan(w * u) + (lambda_ * (w * u)) / (1 + w ** 2 * u ** 2), axis=1) / 2 - u * x / 2
        theta = np.sum(k * np.arctan(w * u) + (lambda_ * (w * u)) / (1 + w ** 2 * u ** 2)) / 2 - u * x / 2
        # rho = np.prod(((1 + w ** 2 * u ** 2) ** (k / 4)) * np.exp(((w ** 2 * u ** 2) * lambda_) / (2 * (1 + w ** 2 * u ** 2))), axis=1) * np.exp(u ** 2 * s ** 2 / 8)
        rho = np.prod(((1 + w ** 2 * u ** 2) ** (k / 4)) * np.exp(((w ** 2 * u ** 2) * lambda_) / (2 * (1 + w ** 2 * u ** 2)))) * np.exp(u ** 2 * s ** 2 / 8)
        return np.sin(theta) / (u * rho)
    # def davies_integrand(u, x, w, k, lambda_, s):
    #     log_rho = logsumexp(np.log1p(w ** 2 * u ** 2) * (k / 4) + ((w ** 2 * u ** 2) * lambda_) / (2 * (1 + w ** 2 * u ** 2))) + u ** 2 * s ** 2 / 8
    #     theta = np.sum(k * np.arctan(w * u) + (lambda_ * (w * u)) / (1 + w ** 2 * u ** 2)) / 2 - u * x / 2
    #     return np.sin(theta) / (u * np.exp(log_rho))

    davies_integral = np.array([quad(lambda u: davies_integrand(u, x_val, w, k, lambda_, s), 0, np.inf, epsabs=AbsTol, epsrel=RelTol)[0] for x_val in np.atleast_1d(x)])

    if side.lower() == 'lower':
        p = 0.5 - davies_integral / np.pi
    elif side.lower() == 'upper':
        p = 0.5 + davies_integral / np.pi

    flag = (p < 0) | (p > 1)
    p = np.maximum(p, 0)
    p = np.minimum(p, 1)

    return p, flag


def gx2cdf_imhof(x: Union[float, np.ndarray], w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, side: str = 'lower', AbsTol: float = 1e-10, RelTol: float = 1e-6, approx: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the cdf of a generalized chi-squared (a weighted sum of non-central chi-squares), using Imhof's [1961] method.

    Required inputs:
    x         points at which to evaluate the cdf
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term

    Optional positional input:
    'upper'   more accurate estimate of the complementary CDF when it's small

    Optional name-value inputs:
    'AbsTol'  absolute error tolerance for the output
    'RelTol'  relative error tolerance for the output
               The absolute OR the relative tolerance is satisfied.
    'approx'  set to 'tail' for Pearson's approximation of the tail probabilities. Works best for the upper (lower) tail when all w are positive (negative).

    Outputs:
    p         computed cdf
    flag      =true if output was too close to 0 or 1 to compute exactly with default settings. Try stricter tolerances or tail approx. for more accuracy.
    """
    w = np.atleast_1d(w)  # Ensure w is an array, even if it's a single value
    k = np.atleast_1d(k)  # Ensure k is an array, even if it's a single value
    lambda_ = np.atleast_1d(lambda_)  # Ensure lambda_ is an array, even if it's a single value

    def imhof_integrand(u, x, w, k, lambda_):
        theta = np.sum(k * np.arctan(w * u) + (lambda_ * (w * u)) / (1 + w ** 2 * u ** 2), axis=1) / 2 - u * x / 2
        rho = np.prod(((1 + w ** 2 * u ** 2) ** (k / 4)) * np.exp(((w ** 2 * u ** 2) * lambda_) / (2 * (1 + w ** 2 * u ** 2))), axis=1)
        return np.sin(theta) / (u * rho)

    if approx.lower() == 'tail':  # compute tail approximations
        j = np.arange(1, 4)
        g = np.sum((w ** j)[:, None] * (j * lambda_ + k), axis=0)
        h = g[1] ** 3 / g[2] ** 2
        if g[2] > 0:
            y = (x - m - g[0]) * np.sqrt(h / g[1]) + h
            if side.lower() == 'lower':
                p = chi2.cdf(y, h)
            elif side.lower() == 'upper':
                p = chi2.sf(y, h)  # survival function (1 - cdf) for 'upper'
        else:
            g = np.sum(((-w) ** j)[:, None] * (j * lambda_ + k), axis=0)
            y = (-(x - m) - g[0]) * np.sqrt(h / g[1]) + h
            if side.lower() == 'lower':
                p = chi2.sf(y, h)  # survival function (1 - cdf) for 'upper'
            elif side.lower() == 'upper':
                p = chi2.cdf(y, h)
    else:
        # compute the integral
        imhof_integral = np.array([quad(lambda u: imhof_integrand(u, x_val, w, k, lambda_), 0, np.inf, epsabs=AbsTol, epsrel=RelTol)[0] for x_val in np.atleast_1d(x)])

        if side.lower() == 'lower':
            p = 0.5 - imhof_integral / np.pi
        elif side.lower() == 'upper':
            p = 0.5 + imhof_integral / np.pi

    flag = (p < 0) | (p > 1)
    p = np.maximum(p, 0)
    p = np.minimum(p, 1)

    return p, flag

def gx2cdf_ruben(x: Union[float, np.ndarray], w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, side: str = 'lower', N: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Returns the cdf of a generalized chi-squared (a weighted sum of non-central chi-squares with all weights the same sign), using Ruben's [1962] method.

    Required inputs:
    x         points at which to evaluate the cdf
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term

    Optional positional input:
    N         no. of terms in the approximation. Default = 1000.

    Outputs:
    p         computed cdf
    errbnd    upper error bound of the computed cdf
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))  # Ensure w is an array, even if it's a single value
    k = np.atleast_1d(k)  # Ensure k is an array, even if it's a single value
    lambda_ = np.atleast_1d(lambda_)  # Ensure lambda_ is an array, even if it's a single value

    lambda_pos = True

    if all(w < 0):
        w = -w
        x = -x
        m = -m
        lambda_pos = False

    beta = 0.90625 * min(w)
    M = sum(k)
    n = np.arange(1, N)

    # compute the g's
    g = np.sum(k * (1 - beta / w) ** n[:, None] + beta * n[:, None] * ((1 - beta / w) ** (n[:, None] - 1)) * (lambda_ / w), axis=1)

    # compute the expansion coefficients
    a = np.empty(N)
    a[0] = np.sqrt(np.exp(-np.sum(lambda_)) * beta ** M * np.prod(w ** -k))
    if a[0] < np.finfo(float).tiny:
        raise ValueError('Underflow error: some series coefficients are smaller than machine precision.')

    for j in range(1, N):
        a[j] = np.dot(np.flip(g[:j]), a[:j]) / (2 * j)

    # compute the central chi-squared integrals
    xg, mg = np.meshgrid((x - m) / beta, M + 2 * np.arange(N))
    F = chi2.cdf(xg, mg)

    # compute the integral
    p = np.dot(a, F)

    # flip if necessary
    if (lambda_pos and side.lower() == 'upper') or (not lambda_pos and side.lower() == 'lower'):
        p = 1 - p

    # compute the truncation error
    errbnd = (1 - np.sum(a)) * chi2.cdf((x - m) / beta, M + 2 * N)

    return p, errbnd


def gx2cdf(x: Union[float, np.ndarray], w: np.ndarray, k: np.ndarray, lambda_: np.ndarray, m: float, s: float, side: str = 'lower', AbsTol: float = 1e-10, RelTol: float = 1e-6) -> np.ndarray:
    """
    Returns the cdf of a generalized chi-squared (a weighted sum of non-central chi-squares and a normal), using Ruben's [1962] method, Davies' [1973] method, or the native ncx2cdf, depending on the input.

    Required inputs:
    x         points at which to evaluate the CDF
    w         row vector of weights of the non-central chi-squares
    k         row vector of degrees of freedom of the non-central chi-squares
    lambda_   row vector of non-centrality paramaters (sum of squares of means) of the non-central chi-squares
    m         mean of normal term
    s         sd of normal term

    Optional positional input:
    'upper'   more accurate estimate of the complementary CDF when it's small

    Optional name-value inputs:
    'AbsTol'  absolute error tolerance for the output
    'RelTol'  relative error tolerance for the output
               The absolute OR the relative tolerance is satisfied.

    Output:
    p         computed cdf
    """
    if s == 0 and len(set(w)) == 1:
        # native ncx2 fallback
        if (np.sign(np.unique(w)[0]) == 1 and side.lower() == 'lower') or (np.sign(np.unique(w)[0]) == -1 and side.lower() == 'upper'):
            p = ncx2.cdf((x - m) / np.unique(w), sum(k), sum(lambda_))
        else:
            p = ncx2.sf((x - m) / np.unique(w), sum(k), sum(lambda_))  # survival function (1 - cdf) for 'upper'
    elif s == 0 and (all(w > 0) or all(w < 0)):
        try:
            p, _ = gx2cdf_ruben(x, w, k, lambda_, m, side=side, N=1000)
        except:
            p, _ = gx2cdf_davies(x, w, k, lambda_, m, s, side=side, AbsTol=AbsTol, RelTol=RelTol)
    else:
        p, _ = gx2cdf_davies(x, w, k, lambda_, m, s, side=side, AbsTol=AbsTol, RelTol=RelTol)

    return p
