"""
Goodness-of-fit diagnostics - Sections 5.1 and 8.2.

All measures compare *observed* and *fitted* yields-to-maturity, where a bond's
fitted YTM is the yield that reprices its model (Svensson) dirty price.  We
report, in basis points:

  * the overall mean absolute error MAE_t (eq. 16);
  * the MAE within each maturity bin MAE_t(tau) (eq. 15);
  * the HPW noise measure - the RMSE of the yield errors (eq. 18).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, estimation
from .estimation import DayData, fitted_cc_yields, model_dirty_prices

BP = 1e4  # decimals -> basis points


def day_metrics(theta, dd: DayData,
                bin_edges=None, bin_labels=None) -> dict:
    """
    Fitting-error diagnostics for a single day, returned as a flat dict
    (one row of the per-day metrics table).  Errors are in basis points.
    """
    bin_edges = bin_edges or config.MATURITY_BIN_EDGES
    bin_labels = bin_labels or config.MATURITY_BIN_LABELS

    y_fit = fitted_cc_yields(theta, dd)
    err = (dd.y_obs_cc - y_fit) * BP            # yield error in bp
    abs_err = np.abs(err)

    row = {
        "n_bonds": dd.n,
        "mae": float(np.nanmean(abs_err)),                      # eq. (16)
        "noise": float(np.sqrt(np.nanmean(err ** 2))),          # eq. (18)
        "price_rmse": float(np.sqrt(np.nanmean(
            (dd.p_obs - model_dirty_prices(theta, dd)) ** 2))),
    }

    # Per-bin MAE (eq. 15).  np.digitize gives bin index 1..len(edges)-1.
    idx = np.digitize(dd.mat_years, bin_edges, right=False)
    for b, label in enumerate(bin_labels, start=1):
        sel = idx == b
        row[f"mae_{label}"] = float(np.nanmean(abs_err[sel])) if sel.any() else np.nan
        row[f"n_{label}"] = int(sel.sum())
    return row


def cross_section_fit(date, bonds: dict, panel, theta=None, optim=None):
    """
    Observed vs fitted yields for every bond available on ``date`` - the data
    behind the snapshot plots of Figs. 5 and 15.

    Returns ``(table, theta)`` where ``table`` has, per bond, the maturity
    (years), observed and fitted cc YTM (percent) and the dirty-price error.
    If ``theta`` is None the curve is fitted on the fly.
    """
    date = pd.Timestamp(date)
    dd = estimation.prepare_day(date, bonds, panel.loc[date])
    if dd is None:
        return None, None
    if theta is None:
        theta = estimation.fit_day(dd, optim=optim).theta
    y_fit = fitted_cc_yields(theta, dd)
    table = pd.DataFrame({
        "isin": dd.isins,
        "maturity": dd.mat_years,
        "y_obs": dd.y_obs_cc * 100.0,
        "y_fit": y_fit * 100.0,
        "price_err": dd.p_obs - model_dirty_prices(theta, dd),
    }).sort_values("maturity").reset_index(drop=True)
    return table, np.asarray(theta)
