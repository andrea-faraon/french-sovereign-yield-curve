"""
On-the-run premium - Section 6 (eq. 17).

The on-the-run (OTR) security of a maturity sector is the most recently issued
bond in that sector; the first off-the-run is the next most recent.  To measure
the premium we, every day:

  1. identify the OTR and first off-the-run bonds in each of the paper's
     maturity ranges (1-4, 5-6, 8-12, 15-16, 20-32 years);
  2. re-fit the Svensson curve on the cross-section that *excludes* those bonds;
  3. for the 5- and 10-year on-the-run bonds, compare the synthetic yield
     predicted by the re-fitted curve with the observed yield (eq. 17):

         OTR_premium = y_predicted(synthetic) - y_observed   (basis points)

A positive premium means the OTR bond trades rich (low yield) relative to the
curve estimated without it - the classic U.S. Treasury pattern.  The paper
finds this premium is negligible on the French market, which is why the OTR
bonds are *kept* in the baseline fit (filter 4).

This module re-uses the baseline estimation machinery; it is comparatively
expensive (a second daily fit) so it is exposed as its own analysis step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bonds as bondmod
from . import estimation, metrics, svensson
from .daycount import year_fraction

# Paper's maturity ranges from which OTR + first off-the-run are removed.
OTR_RANGES = [(1, 4), (5, 6), (8, 12), (15, 16), (20, 32)]


def _otr_and_first_off(date, bonds, isins_present, lo, hi):
    """Return (otr_isin, first_off_isin) for a maturity range, by issue date."""
    d = date.date()
    sector = [(bonds[i].issue, i) for i in isins_present
              if lo <= year_fraction(d, bonds[i].maturity) <= hi]
    sector.sort(reverse=True)                     # most recent issue first
    otr = sector[0][1] if len(sector) >= 1 else None
    first_off = sector[1][1] if len(sector) >= 2 else None
    return otr, first_off


def on_the_run_premium(bonds: dict, panel: pd.DataFrame, dates=None,
                       target_tenors=(5, 10)) -> pd.DataFrame:
    """
    Daily on-the-run premium (bp) for each target tenor (default 5y and 10y).

    For every date we re-fit the curve excluding the OTR/first-off-the-run
    bonds of all ranges, then evaluate eq. (17) for the on-the-run bond closest
    to each target tenor.  Returns a DataFrame indexed by date with one column
    per tenor (e.g. ``otr_5y``, ``otr_10y``) plus the maturity actually used.
    """
    dates = panel.index if dates is None else pd.DatetimeIndex(dates)
    rows, index, warm = [], [], None

    for date in dates:
        row_prices = panel.loc[date]
        present = [i for i in bonds if np.isfinite(row_prices.get(i, np.nan))]
        if len(present) < estimation.config.MIN_BONDS_PER_DAY + 2:
            continue

        # 1) collect bonds to exclude (OTR + first off-the-run per range)
        excluded, otr_by_range = set(), []
        for lo, hi in OTR_RANGES:
            otr, off = _otr_and_first_off(date, bonds, present, lo, hi)
            otr_by_range.append((lo, hi, otr))
            excluded.update(x for x in (otr, off) if x is not None)

        # 2) re-fit on the reduced cross-section
        reduced = {i: b for i, b in bonds.items() if i not in excluded}
        dd_red = estimation.prepare_day(date, reduced, row_prices)
        if dd_red is None:
            continue
        fit, dd_red = estimation.fit_day_robust(dd_red, x0=warm)
        warm = fit.theta

        # observed yields of the full cross-section (for the OTR bonds)
        dd_full = estimation.prepare_day(date, bonds, row_prices)
        y_obs_full = dict(zip(dd_full.isins, dd_full.y_obs_cc))

        rec = {}
        for tenor in target_tenors:
            # on-the-run bond closest to the target tenor among range OTRs
            cands = [(abs(year_fraction(date.date(), bonds[o].maturity) - tenor), o)
                     for _, _, o in otr_by_range if o is not None]
            if not cands:
                rec[f"otr_{tenor}y"] = np.nan
                continue
            _, otr_isin = min(cands)
            bond = bonds[otr_isin]
            times, amounts = bond.future_cashflows(date.date())
            p_model = float(np.sum(
                amounts * np.exp(-svensson.zero_yield(times, fit.theta) * times)))
            y_pred = bondmod.continuously_compounded(
                bondmod.ytm_from_dirty(p_model, times, amounts))
            y_obs = y_obs_full.get(otr_isin, np.nan)
            rec[f"otr_{tenor}y"] = (y_pred - y_obs) * metrics.BP
            rec[f"mat_{tenor}y"] = year_fraction(date.date(), bond.maturity)
        rows.append(rec)
        index.append(date)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="date"))
