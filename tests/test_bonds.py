"""Bond mechanics: cash flows, accrued interest, YTM and duration (eqs. 3-4, 9-10)."""
import datetime as dt

import numpy as np
import pytest

from oatcurve import bonds
from oatcurve.bonds import Bond


def make_bond(coupon=3.0):
    return Bond(isin="TEST", issue=dt.date(2020, 5, 25),
                maturity=dt.date(2030, 5, 25), coupon=coupon)


def test_cashflows_after_valuation():
    b = make_bond()
    times, amounts = b.future_cashflows(dt.date(2025, 1, 15))
    assert len(times) == 6                      # coupons 2025..2030
    assert np.all(np.diff(times) > 0)
    assert amounts[-1] == pytest.approx(103.0)  # last coupon + redemption
    assert np.all(amounts[:-1] == 3.0)


def test_accrued_interest_act_act():
    b = make_bond()
    assert b.accrued_interest(dt.date(2024, 5, 25)) == pytest.approx(0.0)
    # Halfway through the 2024-25 coupon period (365 days)
    mid = dt.date(2024, 5, 25) + dt.timedelta(days=182)
    assert b.accrued_interest(mid) == pytest.approx(3.0 * 182 / 365)


@pytest.mark.parametrize("ytm", [-0.005, 0.0, 0.025, 0.08])
def test_ytm_round_trip(ytm):
    times, amounts = make_bond().future_cashflows(dt.date(2025, 3, 3))
    price = bonds.price_from_ytm(ytm, times, amounts)
    assert bonds.ytm_from_dirty(price, times, amounts) == pytest.approx(ytm, abs=1e-9)


def test_modified_duration_is_price_sensitivity():
    """Eq. (9): D_mod = -(1/P) dP/dY, checked by central finite difference."""
    times, amounts = make_bond().future_cashflows(dt.date(2025, 3, 3))
    y, h = 0.03, 1e-6
    p = bonds.price_from_ytm(y, times, amounts)
    dp = (bonds.price_from_ytm(y + h, times, amounts)
          - bonds.price_from_ytm(y - h, times, amounts)) / (2 * h)
    assert bonds.modified_duration(y, times, amounts, p) == pytest.approx(-dp / p, rel=1e-6)


def test_ytm_unbracketed_returns_nan():
    times, amounts = make_bond().future_cashflows(dt.date(2025, 3, 3))
    assert np.isnan(bonds.ytm_from_dirty(-1.0, times, amounts))
