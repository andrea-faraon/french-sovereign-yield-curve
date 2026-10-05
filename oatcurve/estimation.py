"""
Daily curve calibration - Sections 4.3 (dynamic filters) and 4.4 (estimation).

For every trading day we

  1. assemble the eligible cross-section (``prepare_day``): apply the maturity
     filter, convert clean bid quotes into dirty prices, and compute each
     bond's observed yield-to-maturity and modified duration (the WLS weight);
  2. estimate the six Svensson parameters (``fit_day``) by minimising the
     duration-weighted sum of squared *price* errors (eq. 14)

        theta_hat = argmin_theta  sum_k [ (P_obs_k - P_model_k(theta)) / D_k ]^2

     under the economic constraints tau1, tau2, beta0 > 0 (beta0+beta1 is left
     free so the short rate may be negative).

A warm start from the previous trading day keeps the full-sample run fast; a
small multi-start grid is used on the first day and whenever the warm-started
fit is poor, to guard against local minima.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import bonds as bondmod
from . import config, svensson
from .daycount import year_fraction


# --------------------------------------------------------------------------- #
#  Per-day cross-section                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class DayData:
    """Everything needed to fit (and later score) one trading day."""
    date: pd.Timestamp
    isins: list
    mat_years: np.ndarray         # time to maturity (years) - for binning
    p_obs: np.ndarray             # observed dirty prices
    mod_dur: np.ndarray           # modified durations (WLS weights = 1/D)
    y_obs_cc: np.ndarray          # observed cc yields, ln(1+Y)
    cfs: list                     # per-bond (times, amounts) arrays
    # Flattened cash flows for vectorised model pricing:
    all_times: np.ndarray
    all_amounts: np.ndarray
    bond_idx: np.ndarray

    @property
    def n(self) -> int:
        return len(self.isins)

    def subset(self, idx) -> "DayData":
        """Return a new DayData keeping only the bonds at positions ``idx``."""
        idx = np.asarray(list(idx), dtype=int)
        cfs = [self.cfs[i] for i in idx]
        all_times = np.concatenate([t for t, _ in cfs]) if cfs else np.array([])
        all_amounts = np.concatenate([a for _, a in cfs]) if cfs else np.array([])
        bond_idx = (np.concatenate([np.full(t.size, j) for j, (t, _) in enumerate(cfs)])
                    if cfs else np.array([], dtype=int))
        return DayData(date=self.date, isins=[self.isins[i] for i in idx],
                       mat_years=self.mat_years[idx], p_obs=self.p_obs[idx],
                       mod_dur=self.mod_dur[idx], y_obs_cc=self.y_obs_cc[idx],
                       cfs=cfs, all_times=all_times, all_amounts=all_amounts,
                       bond_idx=bond_idx)


def prepare_day(date: pd.Timestamp, bonds: dict, panel_row: pd.Series,
                min_months: int = None) -> DayData | None:
    """
    Build the eligible cross-section for ``date`` or return ``None`` if fewer
    than :data:`config.MIN_BONDS_PER_DAY` usable quotes are available.

    Applies filter (3): securities with less than ``min_months`` months to
    maturity are dropped.  Bonds whose YTM/duration cannot be computed (e.g.
    a stray non-priceable quote) are also dropped.
    """
    min_months = config.FILTERS.min_months_to_maturity if min_months is None else min_months
    min_mat_years = min_months / 12.0
    ymin, ymax = config.FILTERS.plausible_ytm_range
    d = date.date()

    isins, mats, p_obs, mod_dur, y_cc, cfs = [], [], [], [], [], []
    for isin, bond in bonds.items():
        price = panel_row.get(isin, np.nan)
        if not np.isfinite(price):
            continue
        m = year_fraction(d, bond.maturity)
        if m < min_mat_years:
            continue
        times, amounts = bond.future_cashflows(d)
        if times.size == 0:
            continue
        dirty = price + bond.accrued_interest(d)
        ytm = bondmod.ytm_from_dirty(dirty, times, amounts)
        if not np.isfinite(ytm) or not (ymin <= ytm <= ymax):
            continue                          # drop non-priceable / abnormal quotes
        dmod = bondmod.modified_duration(ytm, times, amounts, dirty)
        if not np.isfinite(dmod) or dmod <= 0:
            continue
        isins.append(isin)
        mats.append(m)
        p_obs.append(dirty)
        mod_dur.append(dmod)
        y_cc.append(bondmod.continuously_compounded(ytm))
        cfs.append((times, amounts))

    if len(isins) < config.MIN_BONDS_PER_DAY:
        return None

    all_times = np.concatenate([t for t, _ in cfs])
    all_amounts = np.concatenate([a for _, a in cfs])
    bond_idx = np.concatenate([np.full(t.size, i) for i, (t, _) in enumerate(cfs)])

    return DayData(date=date, isins=isins,
                   mat_years=np.asarray(mats), p_obs=np.asarray(p_obs),
                   mod_dur=np.asarray(mod_dur), y_obs_cc=np.asarray(y_cc),
                   cfs=cfs, all_times=all_times, all_amounts=all_amounts,
                   bond_idx=bond_idx)


# --------------------------------------------------------------------------- #
#  Model pricing                                                              #
# --------------------------------------------------------------------------- #
def model_dirty_prices(theta, dd: DayData) -> np.ndarray:
    """Vectorised model dirty prices for every bond in the cross-section."""
    df = np.exp(-svensson.zero_yield(dd.all_times, theta) * dd.all_times)
    pv = dd.all_amounts * df
    return np.bincount(dd.bond_idx, weights=pv, minlength=dd.n)


# --------------------------------------------------------------------------- #
#  Daily fit                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class DayFit:
    theta: np.ndarray
    success: bool
    cost: float            # 0.5 * sum of squared weighted residuals
    price_rmse: float      # unweighted RMSE of dirty-price errors (price pts)
    n_bonds: int


def _residuals(theta, dd: DayData) -> np.ndarray:
    """Duration-weighted price residuals (eq. 14 integrand)."""
    return (dd.p_obs - model_dirty_prices(theta, dd)) / dd.mod_dur


def _solve(dd: DayData, x0, optim) -> tuple:
    res = least_squares(
        _residuals, x0, args=(dd,), method="trf",
        bounds=(optim.lower_bounds, optim.upper_bounds),
        max_nfev=optim.max_nfev, ftol=1e-8, xtol=1e-8,
    )
    price_err = dd.p_obs - model_dirty_prices(res.x, dd)
    rmse = float(np.sqrt(np.mean(price_err ** 2)))
    return res.x, float(res.cost), rmse, res.success


def fit_day(dd: DayData, x0=None, optim=None) -> DayFit:
    """
    Estimate the six Svensson parameters for one day (eq. 14).

    Tries the warm start ``x0`` first; if it is missing or yields a poor fit,
    falls back to the multi-start grid and keeps the lowest-cost solution.
    """
    optim = optim or config.OPTIM
    candidates = []

    if x0 is not None:
        x0 = np.clip(x0, optim.lower_bounds, optim.upper_bounds)
        theta, cost, rmse, ok = _solve(dd, x0, optim)
        candidates.append((cost, theta, rmse, ok))
        if rmse <= optim.refit_rmse_threshold:
            return DayFit(theta, ok, cost, rmse, dd.n)

    for start in optim.fallback_starts:
        theta, cost, rmse, ok = _solve(dd, np.asarray(start, float), optim)
        candidates.append((cost, theta, rmse, ok))

    cost, theta, rmse, ok = min(candidates, key=lambda c: c[0])
    return DayFit(theta, ok, cost, rmse, dd.n)


def fitted_cc_yields(theta, dd: DayData) -> np.ndarray:
    """Fitted (model-implied) continuously-compounded YTM for each bond: price
    the bond off the Svensson curve, then invert to its yield."""
    p_model = model_dirty_prices(theta, dd)
    out = np.full(dd.n, np.nan)
    for k, (times, amounts) in enumerate(dd.cfs):
        y = bondmod.ytm_from_dirty(p_model[k], times, amounts)
        out[k] = bondmod.continuously_compounded(y)
    return out


def fit_day_robust(dd: DayData, x0=None, optim=None, max_passes: int = 2):
    """
    Robust daily fit with automated outlier rejection (extends the paper's
    abnormal-quote filter to the post-2018 sample).

    After an initial fit, any bond whose fitted-vs-observed yield error exceeds
    a robust threshold - ``max(outlier_bp, median + 6*1.4826*MAD)`` - is treated
    as an abnormal quote, dropped, and the curve re-fitted (up to ``max_passes``
    times).  The adaptive MAD term keeps genuine pre-euro noise (high median
    error) intact while the absolute floor removes off-curve data glitches in
    the low-error euro era.  Returns ``(DayFit, cleaned DayData)``.
    """
    optim = optim or config.OPTIM
    fit = fit_day(dd, x0, optim)
    for _ in range(max_passes):
        err_bp = np.abs(dd.y_obs_cc - fitted_cc_yields(fit.theta, dd)) * 1e4
        med = np.nanmedian(err_bp)
        mad = np.nanmedian(np.abs(err_bp - med))
        thr = max(optim.outlier_bp, med + 6.0 * 1.4826 * mad)
        keep = err_bp <= thr
        if keep.all() or int(keep.sum()) < config.MIN_BONDS_PER_DAY:
            break
        dd = dd.subset(np.where(keep)[0])
        fit = fit_day(dd, x0=fit.theta, optim=optim)
    return fit, dd
