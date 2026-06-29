"""gx2 - Generalized chi-square distribution
============================================

Python port of the MATLAB *Generalized chi-square distribution* toolbox by
Abhranil Das (Center for Perceptual Systems, The University of Texas at
Austin). It computes the statistics, characteristic function, pdf, cdf,
inverse cdf and random numbers of the generalized chi-square distribution --
the distribution of a weighted sum of non-central chi-square variables plus a
normal variable, equivalently the quadratic form of a normal vector.

A generalized chi-square is parametrised by:

    w        weights of the non-central chi-square terms
    k        their degrees of freedom
    l  their non-centralities (named ``l`` since ``lambda`` is a
             Python keyword)
    s        scale of the added normal term
    m        offset

If you use this code, please cite:
  1. A method to integrate and classify normal distributions
     (https://arxiv.org/abs/2012.14331)
  2. New methods for computing the generalized chi-square distribution
     (https://arxiv.org/abs/2404.05062)
"""

from ._basic import stat, char, rnd
from ._convert import gx2_to_norm_quad_params, norm_quad_to_gx2_params
from ._distribution import cdf, pdf, inv, log_cdf
from ._methods import (imhof, imhof_integrand, ruben, ifft,
                       pearson, cdf_pearson, tail, ellipse)
from ._ray import (cdf_ray, pdf_ray, ray_integrand, int_norm_ray,
                   norm_prob_across_rays, norm_prob_across_angles)
from ._helpers import (log_sum_exp, signed_log_sum_exp, phi_ray,
                       Phibar_ray_split, Phibar_sym, prob_ray_sym, standard_quad)

__version__ = "1.0.2"

__all__ = [
    # core distribution API
    "stat", "char", "rnd", "cdf", "pdf", "inv", "log_cdf",
    # parameter conversions
    "gx2_to_norm_quad_params", "norm_quad_to_gx2_params",
    # individual methods
    "imhof", "imhof_integrand", "ruben", "ifft", "pearson",
    "cdf_pearson", "tail", "ellipse",
    # ray method internals
    "cdf_ray", "pdf_ray", "ray_integrand", "int_norm_ray",
    "norm_prob_across_rays", "norm_prob_across_angles",
    # numerical helpers
    "log_sum_exp", "signed_log_sum_exp", "phi_ray", "Phibar_ray_split",
    "Phibar_sym", "prob_ray_sym", "standard_quad",
]
