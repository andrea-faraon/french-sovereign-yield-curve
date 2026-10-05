"""
Study 1.2 - Decomposing the OAT-Bund spread: common-euro vs idiosyncratic
French political premium, along the whole term structure (extends the single-3y
INSEE Focus, 18 Mar 2025, "Note de conjoncture").

Two blocks, both reloading the already-fitted Svensson OAT and Bund curves from
the cached parquet (no re-fit):

  Block A (core)  - decompose the daily spread into a *common-euro* component
                    (systemic peripheral risk + global risk aversion + the Bund
                    itself) and an *idiosyncratic French* residual, at 2/5/10/30y,
                    and characterise the term structure of that residual.
  Block B (cond.) - INSEE-style two-stage transmission to the cost of new credit
                    to French non-financial corporations (NFC), using ECB MIR.

Common-euro factors are fully ECB-sourced and daily: the CISS (systemic stress)
and the all-bonds-minus-AAA curve (euro periphery dispersion, an Italy/Spain-type
risk proxy that needs no country-level data).  Regressions use OLS with HAC
(Newey-West) standard errors.  All spreads are in basis points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bund_curve, common, external_data
from .political_risk import ols_hac

BP = 1e4
TENORS = (2, 3, 5, 10, 30)


# --------------------------------------------------------------------------- #
#  OAT-Bund spread (own Svensson curves)                                      #
# --------------------------------------------------------------------------- #
def oat_bund_spread(tenors=TENORS) -> pd.DataFrame:
    """OAT minus Bund zero-coupon spread (bp) per tenor, on common trading days."""
    oat = common.load_base()["zero"]
    bund = bund_curve.load_bund_zero(tenors=tenors)
    cols = {}
    for t in tenors:
        k = f"{t}y"
        pair = common.align(oat[k].rename("oat"), bund[k].rename("bund"))
        cols[k] = (pair["oat"] - pair["bund"]) * BP
    return pd.DataFrame(cols).dropna(how="all")


# --------------------------------------------------------------------------- #
#  Section 0 - external validation against the INSEE Focus                    #
# --------------------------------------------------------------------------- #
def insee_validation(spread: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compare our 3y OAT-Bund spread at end-May 2024 and end-Jan 2025 with the
    INSEE Focus figures (≈23 -> ≈41 bp).  Reports our end-of-period value and
    monthly mean (the likely source of any small gap), and the widening.
    """
    spread = oat_bund_spread() if spread is None else spread
    s3 = spread["3y"]
    rows = {}
    for label, month, insee in [("end-May 2024", "2024-05", 23), ("end-Jan 2025", "2025-01", 41)]:
        win = s3.loc[month]
        rows[label] = {"ours_eop_bp": round(win.iloc[-1], 1),
                       "ours_month_mean_bp": round(win.mean(), 1),
                       "INSEE_bp": insee}
    out = pd.DataFrame(rows).T
    out.loc["widening", :] = [out["ours_eop_bp"].diff().iloc[-1],
                              out["ours_month_mean_bp"].diff().iloc[-1],
                              out["INSEE_bp"].diff().iloc[-1]]
    return out


# --------------------------------------------------------------------------- #
#  Section 2 - decompose the spread (Block A core)                            #
# --------------------------------------------------------------------------- #
def _common_factors(tenor: int, benchmark: str = "bund") -> pd.DataFrame:
    """Aligned daily frame: spread, euro periphery dispersion, CISS, Bund level."""
    k = f"{tenor}y"
    spread = oat_bund_spread((tenor,))[k] if benchmark == "bund" else \
        ((common.load_base()["zero"][k] - external_data.fetch_aaa_curve((tenor,))[k]) * BP)
    disp = external_data.fetch_periphery_dispersion((tenor,))[k]
    ciss = external_data.fetch_ciss()
    bund = bund_curve.load_bund_zero((tenor,))[k] * 100.0
    df = common.align(spread.rename("spread"), disp.rename("disp"),
                      ciss.rename("ciss"), bund.rename("bund"))
    return df


def decompose_changes(tenor: int = 10, benchmark: str = "bund") -> dict:
    """
    Daily change regression (OLS-HAC):

        dSpread = a + b1 dDispersion + b2 dCISS + b3 dBund + e

    The fit R^2 is the share of daily spread variation explained by *common-euro*
    factors; the residual ``e`` is the idiosyncratic French daily shock.  Returns
    the coefficient table, R^2, and the cumulative idiosyncratic series (bp).
    """
    df = _common_factors(tenor, benchmark).diff().dropna()
    y = df["spread"].to_numpy()
    X = df[["disp", "ciss", "bund"]].to_numpy()
    beta, se, t, r2 = ols_hac(y, X, lags=5)
    table = pd.DataFrame({"coef": beta, "se": se, "t": t},
                         index=["const", "d_dispersion", "d_CISS", "d_Bund"]).round(3)
    resid = pd.Series(y - X @ beta[1:] - beta[0], index=df.index, name="idio_shock")
    return {"table": table, "r2": round(r2, 3), "n": len(df),
            "idiosyncratic_cum": resid.cumsum().rename("idio_premium_cum_bp")}


def idiosyncratic_premium(tenor: int = 10, benchmark: str = "bund") -> pd.DataFrame:
    """
    Descriptive level decomposition: project the spread *level* on the common-euro
    factors (periphery dispersion + CISS) and treat the residual as the
    idiosyncratic French premium (bp).  Returns spread, fitted common component
    and the residual premium.  (A projection, not a structural model - the
    change regression in ``decompose_changes`` is the rigorous attribution.)
    """
    df = _common_factors(tenor, benchmark)
    y = df["spread"].to_numpy()
    X = df[["disp", "ciss"]].to_numpy()
    beta, _, _, _ = ols_hac(y, X, lags=20)
    common_comp = X @ beta[1:] + beta[0]
    out = pd.DataFrame({"spread": df["spread"],
                        "common_euro": common_comp,
                        "idiosyncratic_FR": df["spread"].to_numpy() - common_comp},
                       index=df.index)
    return out


def variance_share(tenor: int = 10) -> pd.Series:
    """Share of daily spread-change variance that is common-euro vs idiosyncratic."""
    d = decompose_changes(tenor)
    return pd.Series({"common_euro_R2": d["r2"], "idiosyncratic": round(1 - d["r2"], 3)})


# --------------------------------------------------------------------------- #
#  Section 3 - term structure of the political premium (Block A)              #
# --------------------------------------------------------------------------- #
def term_structure_response(event_date, tenors=TENORS, pre: int = 5,
                            post: int = 10) -> pd.DataFrame:
    """
    Change in the OAT-Bund spread at each tenor around an event: the pre-event
    average (window ``[-pre,-1]``) vs the post-event average (``[0,post]``) and
    their difference (bp).  A short-end-concentrated jump signals rollover/near-
    term risk; a long-end one signals structural fiscal deterioration.
    """
    spread = oat_bund_spread(tenors)
    e = common.nearest_trading_day(spread.index, event_date)
    ei = spread.index.get_loc(e)
    rows = {}
    for t in tenors:
        s = spread[f"{t}y"]
        pre_avg = s.iloc[max(0, ei - pre):ei].mean()
        post_avg = s.iloc[ei:ei + post + 1].mean()
        rows[f"{t}y"] = {"pre_bp": round(pre_avg, 1), "post_bp": round(post_avg, 1),
                         "change_bp": round(post_avg - pre_avg, 1)}
    return pd.DataFrame(rows).T


def spread_slope(short: int = 2, long: int = 10) -> pd.Series:
    """Spread-curve slope = (long-tenor spread) - (short-tenor spread), bp.
    Positive/steepening => long-end (structural) risk; flat/short => near-term."""
    sp = oat_bund_spread((short, long))
    return (sp[f"{long}y"] - sp[f"{short}y"]).rename("spread_slope_bp")


# --------------------------------------------------------------------------- #
#  Section 1 - reduced Bund Stage-1 (INSEE Stage-1, US driver omitted)        #
# --------------------------------------------------------------------------- #
def bund_stage1(tenor: int = 3) -> dict:
    """
    Reduced INSEE Stage-1: monthly regression of the German Bund level on the
    common macro drivers we can source - 3m Euribor and the ECB balance sheet -
    plus a post-2010 dummy.  (INSEE also include the US Treasury yield, omitted
    here as FRED is unavailable in this environment.)  Reports the balance-sheet
    coefficient for comparison with INSEE's -0.72 bp per GDP-point elasticity.
    """
    bund = (bund_curve.load_bund_zero((tenor,))[f"{tenor}y"] * 100.0).resample("MS").mean()
    eur = external_data.fetch_euribor_3m()
    bs = external_data.fetch_ecb_balance_sheet().resample("MS").mean()  # EUR bn
    df = common.align(bund.rename("bund"), eur.rename("euribor"), bs.rename("bs"))
    df["bs_trn"] = df["bs"] / 1000.0                       # EUR trillions
    df["post2010"] = (df.index >= "2010-01-01").astype(float)
    y = df["bund"].to_numpy()
    X = df[["euribor", "bs_trn", "post2010"]].to_numpy()
    beta, se, t, r2 = ols_hac(y, X, lags=6)
    table = pd.DataFrame({"coef": beta, "se": se, "t": t},
                         index=["const", "euribor", "bs_EURtrn", "post2010"]).round(3)
    table.attrs["r2"] = round(r2, 3)
    table.attrs["n"] = len(df)
    return {"table": table, "data": df}


# --------------------------------------------------------------------------- #
#  Section 4 - Block B: transmission to NFC credit (INSEE Stage-2, reduced)    #
# --------------------------------------------------------------------------- #
def nfc_stage2(spread_tenor: int = 10) -> dict:
    """
    Reduced INSEE Stage-2: quarterly regression of the rate on new loans to
    French NFCs on the 3m Euribor, the ECB balance sheet and the OAT-Bund spread
    (the French risk premium).  The spread's coefficient measures pass-through of
    the political premium to the real cost of corporate credit.  US driver
    omitted (FRED unavailable); short post-2024 sample -> descriptive, no strong
    causal claim.  Returns the regression and the fitted vs observed series.
    """
    nfc = external_data.fetch_mir_nfc_rate().resample("QS").mean()
    eur = external_data.fetch_euribor_3m().resample("QS").mean()
    bs = (external_data.fetch_ecb_balance_sheet() / 1000.0).resample("QS").mean()
    spread = (oat_bund_spread((spread_tenor,))[f"{spread_tenor}y"] / 100.0).resample("QS").mean()  # bp->pct pts
    df = common.align(nfc.rename("nfc"), eur.rename("euribor"),
                      bs.rename("bs_trn"), spread.rename("spread_pp"))
    y = df["nfc"].to_numpy()
    X = df[["euribor", "bs_trn", "spread_pp"]].to_numpy()
    beta, se, t, r2 = ols_hac(y, X, lags=4)
    table = pd.DataFrame({"coef": beta, "se": se, "t": t},
                         index=["const", "euribor", "bs_EURtrn", "spread_pp"]).round(3)
    table.attrs["r2"] = round(r2, 3)
    table.attrs["n"] = len(df)
    fitted = pd.Series(X @ beta[1:] + beta[0], index=df.index, name="nfc_fitted")
    df["nfc_fitted"] = fitted
    return {"table": table, "data": df}
