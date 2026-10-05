"""
oatcurve
========

Replication of the French nominal zero-coupon yield curve following

    Grishchenko, O. V., Moraux, F., & Pakulyak, O. (2020).
    "Fuel up with OATmeals! The case of the French nominal yield curve."
    The Journal of Finance and Data Science, 6, 49-85.

The estimation strategy follows Gurkaynak, Sack & Wright (GSW, 2007) applied to
the Svensson (1994) parametric forward-rate curve.  The package is organised so
that each module maps to a precise part of the paper:

    daycount    -> Section 4.1  (year fractions, accrual conventions)
    svensson    -> Section 4.2  (eqs. 11, 12; forward / zero / par yields)
    bonds       -> Section 4.1  (eqs. 3-5, 9-10; pricing, YTM, durations)
    data_loading-> Section 3    (security master + daily bid prices)
    estimation  -> Section 4.3-4.4 (filters + weighted-least-squares fit, eq. 14)
    metrics     -> Section 5.1 / 8.2 (MAE eqs. 15-16, noise measure eq. 18)
    analysis    -> Section 5.2 (zero curve, level/slope/curvature, PCA)
    otr         -> Section 6   (on-the-run premium, eq. 17)

All public functions are documented with their corresponding equation numbers.
"""

from . import (analysis, bonds, config, data_loading, daycount,  # noqa: F401
               estimation, metrics, otr, pipeline, svensson)

__version__ = "1.0.0"
__all__ = ["config", "daycount", "svensson", "bonds", "data_loading",
           "estimation", "metrics", "pipeline", "analysis", "otr"]
