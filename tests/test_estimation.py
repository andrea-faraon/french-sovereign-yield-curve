"""End-to-end check of the daily calibration (eq. 14) on synthetic OAT prices.

Bonds are priced exactly off a known Svensson curve; the estimator must recover
that curve and report (near-)zero fitting errors.  No Bloomberg data needed.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from oatcurve import estimation, metrics, svensson
from oatcurve.bonds import Bond

TRUE_THETA = np.array([0.032, -0.018, 0.012, -0.006, 1.6, 8.5])
DATE = pd.Timestamp("2025-03-03")


def synthetic_market(theta=TRUE_THETA, n=40, seed=0):
    """A cross-section of annual-coupon bullets and their clean prices."""
    rng = np.random.default_rng(seed)
    bonds, prices = {}, {}
    for k in range(n):
        years = rng.uniform(1.5, 40.0)
        maturity = (DATE + pd.Timedelta(days=int(years * 365.25))).date()
        issue = maturity - relativedelta(years=int(np.ceil(years)) + 1)
        b = Bond(isin=f"SYN{k:03d}", issue=issue, maturity=maturity,
                 coupon=round(rng.uniform(0.0, 6.0), 2))
        times, amounts = b.future_cashflows(DATE.date())
        dirty = float(np.sum(amounts * svensson.discount_factor(times, theta)))
        bonds[b.isin] = b
        prices[b.isin] = dirty - b.accrued_interest(DATE.date())
    return bonds, pd.Series(prices)


def test_prepare_day_applies_maturity_filter():
    bonds, prices = synthetic_market()
    short = Bond(isin="SHORT", issue=dt.date(2020, 1, 1),
                 maturity=dt.date(2025, 9, 1), coupon=1.0)    # < 12 months left
    bonds["SHORT"] = short
    prices["SHORT"] = 99.5
    dd = estimation.prepare_day(DATE, bonds, prices)
    assert "SHORT" not in dd.isins
    assert dd.n == 40


def test_fit_recovers_true_curve():
    bonds, prices = synthetic_market()
    dd = estimation.prepare_day(DATE, bonds, prices)
    fit = estimation.fit_day(dd)
    tenors = np.array([2.0, 5.0, 10.0, 20.0, 30.0])
    err_bp = (svensson.zero_yield(tenors, fit.theta)
              - svensson.zero_yield(tenors, TRUE_THETA)) * 1e4
    assert np.max(np.abs(err_bp)) < 0.5
    assert fit.price_rmse < 1e-3


def test_metrics_near_zero_on_exact_prices():
    bonds, prices = synthetic_market()
    dd = estimation.prepare_day(DATE, bonds, prices)
    fit = estimation.fit_day(dd)
    row = metrics.day_metrics(fit.theta, dd)
    assert row["mae"] < 0.1          # bp
    assert row["noise"] < 0.1        # bp


def corrupt_quote(prices, dd, position, shock_bp):
    """Lower one clean price by the amount that raises its yield ~shock_bp."""
    isin = dd.isins[position]
    prices = prices.copy()
    prices[isin] -= shock_bp * 1e-4 * dd.mod_dur[position] * dd.p_obs[position]
    return isin, prices


def test_robust_fit_drops_a_bad_quote():
    """A 150 bp quote error on a 10-year bond is removed and the curve is intact."""
    bonds, prices = synthetic_market()
    dd0 = estimation.prepare_day(DATE, bonds, prices)
    ten_year = int(np.argmin(np.abs(dd0.mat_years - 10.0)))
    bad, prices = corrupt_quote(prices, dd0, ten_year, shock_bp=150)
    dd = estimation.prepare_day(DATE, bonds, prices)
    fit, cleaned = estimation.fit_day_robust(dd)
    assert set(dd.isins) - set(cleaned.isins) == {bad}
    err_bp = (svensson.zero_yield(10.0, fit.theta)
              - svensson.zero_yield(10.0, TRUE_THETA)) * 1e4
    assert abs(err_bp) < 0.5


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation: outliers are screened on the residuals of a non-robust "
    "first fit, so a gross error on the shortest bond (highest leverage, "
    "smallest duration weight) bends the curve towards itself and the "
    "rejection hits its neighbour instead (masking)."))
def test_robust_fit_edge_bond_masking():
    bonds, prices = synthetic_market()
    dd0 = estimation.prepare_day(DATE, bonds, prices)
    shortest = int(np.argmin(dd0.mat_years))
    bad, prices = corrupt_quote(prices, dd0, shortest, shock_bp=300)
    dd = estimation.prepare_day(DATE, bonds, prices)
    _, cleaned = estimation.fit_day_robust(dd)
    assert set(dd.isins) - set(cleaned.isins) == {bad}
