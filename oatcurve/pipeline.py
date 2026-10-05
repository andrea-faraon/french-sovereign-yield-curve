"""
End-to-end daily estimation loop.

``run_fit`` walks the trading calendar once, warm-starting each day from the
previous day's solution, and returns a tidy per-day table with the six Svensson
parameters, fit diagnostics and (optionally) the MAE / noise diagnostics of
Section 5.1.  Helper functions then turn the parameter panel into zero-coupon,
par and forward curves at arbitrary tenors.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, estimation, metrics, svensson

PARAM_COLS = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]


def run_fit(bonds: dict, panel: pd.DataFrame, dates=None, optim=None,
            compute_metrics: bool = True, progress: bool = True) -> pd.DataFrame:
    """
    Fit the Svensson curve on every requested date.

    Parameters
    ----------
    bonds   : dict[isin -> Bond]      (already statically filtered)
    panel   : daily clean bid-price panel (index = date, columns = ISIN)
    dates   : optional subset of dates (default: the whole panel)
    Returns a DataFrame indexed by date with the parameters, ``n_bonds``,
    ``price_rmse``, ``success`` and, if requested, the fitting-error metrics.
    """
    optim = optim or config.OPTIM
    dates = panel.index if dates is None else pd.DatetimeIndex(dates)
    n_total = len(dates)
    t0 = time.time()

    rows, index, warm = [], [], None
    for i, date in enumerate(dates):
        dd = estimation.prepare_day(date, bonds, panel.loc[date])
        if dd is None:
            continue
        fit, dd = estimation.fit_day_robust(dd, x0=warm, optim=optim)
        warm = fit.theta                      # carry forward as next warm start

        row = dict(zip(PARAM_COLS, fit.theta))
        row.update(n_bonds=fit.n_bonds, price_rmse=fit.price_rmse,
                   success=fit.success)
        if compute_metrics:
            row.update(metrics.day_metrics(fit.theta, dd))
        rows.append(row)
        index.append(date)

        if progress and (i % 500 == 0 or i == n_total - 1):
            el = time.time() - t0
            print(f"  [{i + 1:>5}/{n_total}] {date.date()}  "
                  f"fitted={len(rows)}  elapsed={el:5.1f}s", flush=True)

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="date"))
    return out


# --------------------------------------------------------------------------- #
#  Curve reconstruction from the fitted parameters                            #
# --------------------------------------------------------------------------- #
def _theta_matrix(params: pd.DataFrame) -> np.ndarray:
    return params[PARAM_COLS].to_numpy()


def zero_curve(params: pd.DataFrame, tenors=None) -> pd.DataFrame:
    """Continuously-compounded zero-coupon yields (decimals) at each tenor."""
    tenors = tenors or config.STANDARD_TENORS
    thetas = _theta_matrix(params)
    data = {f"{m}y": [svensson.zero_yield(m, th) for th in thetas] for m in tenors}
    return pd.DataFrame(data, index=params.index)


def par_curve(params: pd.DataFrame, tenors=None) -> pd.DataFrame:
    """Par yields (decimals) at each tenor (eq. 5)."""
    tenors = tenors or config.STANDARD_TENORS
    thetas = _theta_matrix(params)
    data = {f"{m}y": [float(svensson.par_yield(m, th)) for th in thetas] for m in tenors}
    return pd.DataFrame(data, index=params.index)


def forward_curve(params: pd.DataFrame, tenors=None) -> pd.DataFrame:
    """Instantaneous forward rates (decimals) at each tenor (eq. 11)."""
    tenors = tenors or config.STANDARD_TENORS
    thetas = _theta_matrix(params)
    data = {f"{m}y": [float(svensson.forward_rate(m, th)) for th in thetas] for m in tenors}
    return pd.DataFrame(data, index=params.index)


def maturity_span(bonds: dict, panel: pd.DataFrame, dates=None,
                  min_months: int = None) -> pd.DataFrame:
    """
    Per-day min/max available maturity (years) of the eligible cross-section.

    Used to flag tenors that are *extrapolations* of the fitted curve (outside
    the range of traded bonds), which the Svensson form cannot pin down - most
    visibly the short end before ~1993, when no short OATs existed.
    """
    from .daycount import year_fraction
    min_months = config.FILTERS.min_months_to_maturity if min_months is None else min_months
    min_y = min_months / 12.0
    dates = panel.index if dates is None else pd.DatetimeIndex(dates)
    mats = {i: b.maturity for i, b in bonds.items()}
    rows, idx = [], []
    for date in dates:
        d = date.date()
        row = panel.loc[date]
        ms = [year_fraction(d, mats[i]) for i in bonds
              if pd.notna(row.get(i)) and year_fraction(d, mats[i]) >= min_y]
        if len(ms) >= config.MIN_BONDS_PER_DAY:
            rows.append((min(ms), max(ms)))
            idx.append(date)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"),
                        columns=["mat_min", "mat_max"])


def mask_extrapolated(curve: pd.DataFrame, span: pd.DataFrame) -> pd.DataFrame:
    """Blank out tenors outside the daily traded-maturity range (extrapolation)."""
    out = curve.copy()
    span = span.reindex(out.index)
    for col in out.columns:
        t = float(col.rstrip("y"))
        out.loc[(t < span["mat_min"]) | (t > span["mat_max"]), col] = float("nan")
    return out
