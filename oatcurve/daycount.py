"""
Day-count utilities (Section 4.1).

OATs/BTANs are annual-coupon bonds quoted on an Actual/Actual (ICMA) basis.
Two distinct measurements are needed and intentionally kept separate:

* ``year_fraction`` - the horizon (in years) used to discount a cash flow and
  to define the maturity argument ``m`` of the Svensson curve.  We use
  Actual/365.25, the GSW convention, so it is consistent with the
  yield-to-maturity definition used elsewhere.

* ``accrual_fraction`` - the Actual/Actual (ICMA) fraction of the current
  coupon period that has elapsed, used to turn a quoted (clean) bid price into
  the full (dirty) price that the no-arbitrage pricing equation (3) returns.
"""
from __future__ import annotations

import datetime as dt

from .config import DAYS_PER_YEAR


def year_fraction(start: dt.date, end: dt.date) -> float:
    """Actual/365.25 year fraction between two dates (used for discounting)."""
    return (end - start).days / DAYS_PER_YEAR


def accrual_fraction(prev_coupon: dt.date,
                     settlement: dt.date,
                     next_coupon: dt.date) -> float:
    """
    Actual/Actual (ICMA) accrued fraction of the running coupon period.

    Returns ``(settlement - prev_coupon) / (next_coupon - prev_coupon)`` clipped
    to [0, 1].  For an annual coupon this is the share of the year's coupon that
    has accrued to the bondholder at ``settlement``.
    """
    period = (next_coupon - prev_coupon).days
    if period <= 0:
        return 0.0
    frac = (settlement - prev_coupon).days / period
    return min(1.0, max(0.0, frac))
