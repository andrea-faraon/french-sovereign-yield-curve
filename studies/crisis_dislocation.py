"""
Study 3 - Market dislocation: March 2020 vs past crises (extends Section 8.2).

The paper reads the fitting error (MAE) and the HPW noise measure as proxies for
illiquidity / the (un)availability of arbitrage capital, and shows they spike
during the 2008 Lehman failure and the 2011-12 sovereign crisis.  We add the
COVID-19 episode and ask, numerically:

  * Did the French bond market dislocate in March 2020 as severely as in 2008?
  * Did the ECB's PEPP (announced 18 Mar 2020) compress the noise far faster,
    restoring pricing efficiency?

We quantify the peak dislocation and the *recovery half-life* (trading days for
the noise to fall halfway back from its peak to the pre-crisis baseline) for
each crisis, and overlay the episodes in event time around their peaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common

# Each crisis is anchored at a fixed, well-known ONSET date (not a data-driven
# peak, which would be sensitive to later unrelated humps), and analysed over
# the following ~6 months (POST trading days).  ``policy`` is the key central-
# bank response shown on the detail plot.
CRISES = {
    "GFC / Lehman (2008)": ("2008-09-15", "Lehman bankruptcy"),
    "Euro sovereign (2011)": ("2011-08-01", "euro crisis hits core"),
    "COVID-19 (2020)": ("2020-02-24", "pandemic sell-off / PEPP"),
    "Inflation shock (2022)": ("2022-02-24", "war / inflation surge"),
}
POST = 250    # trading days after onset (~1 year) - long enough to capture the
              # full recovery (e.g. COVID's, in late 2020) and not mistake the
              # separate 2023-25 structural floor for a non-recovery.
PRE = 20


def _baseline(series: pd.Series, onset, lookback: int = 60) -> float:
    """Pre-crisis baseline: median level over the ``lookback`` days before onset."""
    pre = series[series.index < pd.Timestamp(onset)].iloc[-lookback:]
    return float(pre.median())


def _window(series: pd.Series, onset, post: int = POST):
    """Slice [onset, onset+post trading days] of a date-indexed series."""
    i = series.index.searchsorted(pd.Timestamp(onset))
    return series.iloc[i: i + post + 1]


def crisis_summary(metric: str = "noise", params=None, crises=None,
                   lookback: int = 60, smooth: int = 5, post: int = POST) -> pd.DataFrame:
    """
    Peak dislocation and recovery speed per crisis, for ``metric`` in
    {'noise','mae'}.  Measured on a ``smooth``-day moving average within
    [onset, onset+post] so the figures reflect each crisis's own 6-month episode
    rather than later, unrelated volatility.

    Reports the pre-crisis baseline, the peak value/date, the peak as a multiple
    of baseline, and the recovery half-life: the number of trading days after the
    peak before the metric falls halfway back to baseline.  ``recovery_halflife
    = NaN`` strictly means *not recovered within ``post`` trading days*, NOT that
    the metric never recovered - with ``post`` too short this hides genuine
    recoveries (e.g. COVID's), which is why the window is now ~1 year.

    NOTE: this overall measure is not directly comparable across crises because
    the cross-section (number and maturity mix of bonds) differs.  Use
    :func:`per_bin_severity` for the cross-section-normalised comparison.
    """
    params = common.load_base()["params"] if params is None else params
    crises = crises or CRISES
    s = params[metric].rolling(smooth, min_periods=1).mean()
    rows = {}
    for label, (onset, _) in crises.items():
        win = _window(s, onset, post)
        if win.empty:
            continue
        base = _baseline(s, onset, lookback)
        peak_date = win.idxmax()
        peak = float(win.max())
        half_level = base + 0.5 * (peak - base)
        after = win.loc[peak_date:]
        below = after[after <= half_level]
        half_life = (after.index.get_loc(below.index[0]) if len(below) else np.nan)
        rows[label] = {
            "baseline_bp": round(base, 2),
            "peak_bp": round(peak, 2),
            "peak_date": peak_date.date(),
            "peak/baseline": round(peak / base, 1) if base > 0 else np.nan,
            "recovery_halflife_days": half_life,
        }
    return pd.DataFrame(rows).T


def event_time_overlay(metric: str = "noise", params=None, crises=None,
                       pre: int = PRE, post: int = POST, smooth: int = 5) -> pd.DataFrame:
    """
    Each crisis's (smoothed) ``metric`` path aligned in event time (trading days
    relative to its **onset**), for an overlay of magnitude and recovery speed.
    """
    params = common.load_base()["params"] if params is None else params
    crises = crises or CRISES
    s = params[metric].rolling(smooth, min_periods=1).mean()
    out = {}
    for label, (onset, _) in crises.items():
        i = s.index.searchsorted(pd.Timestamp(onset))
        seg = s.iloc[max(0, i - pre): i + post + 1]
        offset = np.arange(len(seg)) - (i - max(0, i - pre))
        out[label] = pd.Series(seg.values, index=offset)
    return pd.DataFrame(out)


def per_bin_severity(params=None, crises=None, bins=("2-5yr", "5-10yr", "10-20yr"),
                     metric_prefix: str = "mae", smooth: int = 5,
                     post: int = POST) -> pd.DataFrame:
    """
    Peak fitting error WITHIN each maturity bin, per crisis - the cross-section-
    normalised severity comparison.

    Comparing crises within a fixed maturity bin (rather than the raw overall
    RMSE) controls for differences in the number and maturity mix of bonds quoted
    in each episode.  This matters here because the OAT cross-section is small
    (~30-50 bonds vs HPW's >100), so its composition sways the overall measure;
    HPW themselves exclude < 1y and note the short end is structurally noisier, so
    segmenting by bin is principled, not ad hoc.  The per-bin MAE is the paper's
    own eq. (15).  Returns the peak (smoothed) bp per (crisis x maturity bin).
    """
    params = common.load_base()["params"] if params is None else params
    crises = crises or CRISES
    rows = {}
    for label, (onset, _) in crises.items():
        rec = {}
        for b in bins:
            s = params[f"{metric_prefix}_{b}"].rolling(smooth, min_periods=1).mean()
            w = _window(s, onset, post)
            rec[f"{b} (bp)"] = round(float(w.max()), 2) if len(w) else float("nan")
        rows[label] = rec
    return pd.DataFrame(rows).T


def leave_one_out_bin_mae(date, lo: float = 5, hi: float = 10,
                          bonds=None, panel=None) -> dict:
    """
    Robustness for the HPW outlier concern: the within-[lo,hi]y MAE on ``date``,
    computed on all bonds in the bin vs after dropping the single worst-fit bond.
    If the two are close, the bin's dislocation is broad-based - not driven by one
    squeezed (e.g. low-free-float) security.  ``bonds``/``panel`` come from
    ``common.load_base(with_bonds=True)``; pass them in to avoid reloading.
    """
    from oatcurve import metrics
    if bonds is None or panel is None:
        base = common.load_base(with_bonds=True)
        bonds, panel = base["bonds"], base["panel"]
    tab, _ = metrics.cross_section_fit(date, bonds, panel)
    sub = tab[(tab["maturity"] >= lo) & (tab["maturity"] <= hi)]
    err = (sub["y_obs"] - sub["y_fit"]).abs() * 100.0          # percent -> bp
    full = err.mean()
    loo = err.drop(err.idxmax()).mean() if len(err) > 1 else float("nan")
    return {"date": str(pd.Timestamp(date).date()), "n_in_bin": int(len(sub)),
            "MAE_full_bp": round(full, 2), "MAE_drop_worst_bp": round(loo, 2),
            "worst_bond_contribution_bp": round(full - loo, 2)}


def march2020_detail(params=None) -> pd.DataFrame:
    """Daily MAE and noise from the COVID onset through the 2021 normalisation
    (shows the March spike, the August peak and the recovery)."""
    params = common.load_base()["params"] if params is None else params
    return params.loc["2020-02-01":"2021-06-30", ["mae", "noise"]]


def compare_to_2008(params=None) -> pd.DataFrame:
    """
    Headline comparison for the thesis question: COVID vs 2008 vs 2011 on peak
    severity and recovery speed of the noise measure.
    """
    return crisis_summary("noise", params)[
        ["baseline_bp", "peak_bp", "peak/baseline", "recovery_halflife_days"]]
