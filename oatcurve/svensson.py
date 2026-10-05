"""
Svensson (1994) parametric term structure - Section 4.2 of the paper.

Parameter vector (kept in this order everywhere in the code base):

    theta = (beta0, beta1, beta2, beta3, tau1, tau2)

Instantaneous forward rate (eq. 11):

    f(m) = beta0
         + beta1 * exp(-m/tau1)
         + beta2 * (m/tau1) * exp(-m/tau1)
         + beta3 * (m/tau2) * exp(-m/tau2)

Continuously-compounded zero-coupon yield (eq. 12), obtained by integrating the
forward curve over [0, m] (eq. 8):

    y(m) = beta0
         + beta1 * G(m/tau1)
         + beta2 * (G(m/tau1) - exp(-m/tau1))
         + beta3 * (G(m/tau2) - exp(-m/tau2))

with the loadings function  G(x) = (1 - exp(-x)) / x  and  G(0) = 1.

All rates are in DECIMAL units (0.03 == 3%) and all maturities ``m`` are in
years.  Functions are fully vectorised over ``m``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _loading(x: np.ndarray) -> np.ndarray:
    """G(x) = (1 - e^{-x}) / x, with the analytic limit G(0)=1 handled safely."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > EPS
    out[nz] = (1.0 - np.exp(-x[nz])) / x[nz]
    return out


def zero_yield(m, theta) -> np.ndarray:
    """Continuously-compounded zero-coupon yield y(m) - eq. (12)."""
    b0, b1, b2, b3, t1, t2 = theta
    m = np.asarray(m, dtype=float)
    x1, x2 = m / t1, m / t2
    g1, g2 = _loading(x1), _loading(x2)
    return b0 + b1 * g1 + b2 * (g1 - np.exp(-x1)) + b3 * (g2 - np.exp(-x2))


def forward_rate(m, theta) -> np.ndarray:
    """Instantaneous forward rate f(m) - eq. (11)."""
    b0, b1, b2, b3, t1, t2 = theta
    m = np.asarray(m, dtype=float)
    x1, x2 = m / t1, m / t2
    return b0 + b1 * np.exp(-x1) + b2 * x1 * np.exp(-x1) + b3 * x2 * np.exp(-x2)


def discount_factor(m, theta) -> np.ndarray:
    """Zero-coupon bond price B(m) = exp(-y(m) * m) - eq. (1)."""
    m = np.asarray(m, dtype=float)
    return np.exp(-zero_yield(m, theta) * m)


def par_yield(m, theta) -> np.ndarray:
    """
    Par yield y_c(m) - eq. (5): the annual coupon at which a bond maturing in
    ``m`` years trades at par.

        y_c(m) = (1 - B(m)) / sum_i B(t_i)

    Coupon dates t_i are annual, anchored at maturity (t_i = m, m-1, ... > 0),
    matching the actual OAT cash-flow structure.  Scalar or vector ``m``.
    """
    m_arr = np.atleast_1d(np.asarray(m, dtype=float))
    out = np.empty_like(m_arr)
    for i, mat in enumerate(m_arr):
        if mat <= 0:
            out[i] = np.nan
            continue
        n_full = int(np.floor(mat - 1e-9))
        times = [mat - k for k in range(n_full + 1) if mat - k > 1e-9]
        times = np.array(sorted(times))
        b = discount_factor(times, theta)
        annuity = b.sum()
        out[i] = (1.0 - discount_factor(mat, theta)) / annuity if annuity > 0 else np.nan
    return out if np.ndim(m) else float(out[0])
