"""
Bond cash-flow mechanics, pricing, yield-to-maturity and durations.

Implements equations (3)-(5) and (9)-(10) of Section 4.1 for annual-coupon
French government securities.  A :class:`Bond` precomputes its anniversary
(coupon) dates once; the per-day quantities (future cash flows, accrued
interest, dirty price, YTM, modified duration) are then cheap to evaluate over
the ~10,000 trading days of the sample.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq

from .config import FACE_VALUE
from .daycount import accrual_fraction, year_fraction


# --------------------------------------------------------------------------- #
#  Coupon schedule                                                            #
# --------------------------------------------------------------------------- #
def anniversary_dates(issue: dt.date, maturity: dt.date) -> list[dt.date]:
    """
    Annual coupon (anniversary) dates anchored at the maturity day/month.

    Returns every anniversary from maturity backwards down to and *including*
    the first one on or before the issue date.  That earliest date is the
    quasi-coupon anchor of the first period and is used only for ACT/ACT
    accrued-interest computations - it is never an actual payment.
    """
    dates = [maturity]
    k = 1
    while True:
        d = maturity - relativedelta(years=k)
        dates.append(d)
        if d <= issue:
            break
        k += 1
    return sorted(dates)


@dataclass
class Bond:
    """A single fixed-coupon OAT/BTAN security."""
    isin: str
    issue: dt.date
    maturity: dt.date
    coupon: float                      # annual coupon per 100 of face (e.g. 3.0)
    currency: str = "EUR"
    security_type: str = "OAT"
    face: float = FACE_VALUE
    _anniv: list[dt.date] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._anniv = anniversary_dates(self.issue, self.maturity)

    # ----- cash flows ----------------------------------------------------- #
    def future_cashflows(self, valuation: dt.date):
        """
        Future cash-flow times (years, Act/365.25) and amounts as of
        ``valuation`` (the quote date is used as the settlement date).

        Coupons fall on every anniversary strictly after ``valuation``; the
        redemption of ``face`` is added on the maturity date.  Returns
        ``(times, amounts)`` as float arrays.
        """
        times, amounts = [], []
        for d in self._anniv:
            if d <= valuation or d <= self.issue:
                continue
            amt = self.coupon
            if d == self.maturity:
                amt += self.face
            times.append(year_fraction(valuation, d))
            amounts.append(amt)
        # Pure discount security (coupon == 0): only the redemption remains.
        if not amounts and valuation < self.maturity:
            times = [year_fraction(valuation, self.maturity)]
            amounts = [self.face]
        return np.asarray(times, dtype=float), np.asarray(amounts, dtype=float)

    # ----- accrued interest ----------------------------------------------- #
    def accrued_interest(self, valuation: dt.date) -> float:
        """Accrued coupon at ``valuation`` (ACT/ACT ICMA) - turns clean into dirty."""
        if self.coupon == 0.0:
            return 0.0
        prev_d, next_d = None, None
        for d in self._anniv:
            if d <= valuation:
                prev_d = d
            elif next_d is None:
                next_d = d
                break
        if prev_d is None or next_d is None:
            return 0.0
        return self.coupon * accrual_fraction(prev_d, valuation, next_d)


# --------------------------------------------------------------------------- #
#  Pricing, yield-to-maturity and durations                                   #
# --------------------------------------------------------------------------- #
def price_from_ytm(ytm: float, times: np.ndarray, amounts: np.ndarray) -> float:
    """Dirty price from an annually-compounded yield-to-maturity (eq. 4)."""
    return float(np.sum(amounts * (1.0 + ytm) ** (-times)))


def ytm_from_dirty(dirty: float, times: np.ndarray, amounts: np.ndarray,
                   lo: float = -0.95, hi: float = 2.0) -> float:
    """
    Annually-compounded yield-to-maturity that reprices ``dirty`` (eq. 4),
    solved with Brent's method.  Returns NaN if no root is bracketed.
    """
    if times.size == 0 or dirty <= 0:
        return np.nan
    f = lambda y: price_from_ytm(y, times, amounts) - dirty
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return np.nan
    try:
        return brentq(f, lo, hi, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def continuously_compounded(ytm: float) -> float:
    """Convert an annually-compounded yield to its cc counterpart y = ln(1+Y)."""
    return np.log1p(ytm) if (ytm is not None and ytm > -1.0) else np.nan


def macaulay_duration(ytm: float, times: np.ndarray, amounts: np.ndarray,
                      dirty: float) -> float:
    """Macaulay duration (eq. 10): PV-weighted average time to cash flow."""
    if times.size == 0 or dirty <= 0 or np.isnan(ytm):
        return np.nan
    pv = amounts * (1.0 + ytm) ** (-times)
    return float(np.sum(times * pv) / dirty)


def modified_duration(ytm: float, times: np.ndarray, amounts: np.ndarray,
                      dirty: float) -> float:
    """Modified duration (eq. 9): D = D_Mac / (1 + Y).  Used as the WLS weight."""
    dmac = macaulay_duration(ytm, times, amounts, dirty)
    return dmac / (1.0 + ytm) if not np.isnan(dmac) else np.nan
