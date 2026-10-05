"""
Study 1 - Political risk and the OAT-Bund safety premium (extends Section 7.1).

Idea: France's safety premium is the spread of the OAT zero-coupon curve over a
euro-area "safe" benchmark (the ECB AAA curve, a clean Bund proxy estimated with
the same Svensson method).  We then study how the 2024-2026 French political
instability repriced that premium, using a Brexit-style event study (the UK
paper's model, eq. on its p.: ``dSpread = a + g*event + d*dRiskAversion +
phi*Spread_{t-1} + e``) plus the rolling OAT-vs-Bund beta that measures whether
the OAT still behaves as a "core" asset (beta ~ 1, comoves with the Bund) or has
drifted toward a riskier "middle-ground" asset (beta < 1, decoupling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bund_curve, common, external_data

BP = 1e4


def _get_benchmark(tenors, benchmark: str = "bund", benchmark_csv=None) -> pd.DataFrame:
    """
    Safe-benchmark zero curve (decimals) used to define the safety premium.

      * ``benchmark="bund"`` (default) - the German Bund curve, fitted with the
        same Svensson engine as the OATs (most faithful to the paper).
      * ``benchmark="aaa"``           - the ECB AAA euro-area proxy (auto-fetched).
      * ``benchmark_csv=...``         - a user CSV (e.g. a Bloomberg German curve).
    """
    if benchmark_csv is not None:
        return external_data.load_benchmark_curve(benchmark_csv, tenors=tenors)
    if benchmark == "aaa":
        return external_data.fetch_aaa_curve(tenors=tenors)
    return bund_curve.load_bund_zero(tenors=tenors)


# --------------------------------------------------------------------------- #
#  Safety-premium spread                                                       #
# --------------------------------------------------------------------------- #
def build_spreads(tenors=(2, 5, 10, 30), benchmark="bund", benchmark_csv=None) -> pd.DataFrame:
    """
    OAT-minus-benchmark zero-coupon spreads in **basis points**, one column per
    tenor, on the common trading days.  Benchmark defaults to the German Bund
    curve (own Svensson fit); see :func:`_get_benchmark` for alternatives.
    """
    oat = common.load_base()["zero"]
    bench = _get_benchmark(tenors, benchmark, benchmark_csv)
    cols = {}
    for t in tenors:
        key = f"{t}y"
        pair = common.align(oat[key].rename("oat"), bench[key].rename("bench"))
        cols[key] = (pair["oat"] - pair["bench"]) * BP
    return pd.DataFrame(cols).dropna(how="all")


def benchmark_levels(tenors=(2, 5, 10, 30), benchmark="bund", benchmark_csv=None):
    """Return aligned OAT and benchmark yields (decimals) for the given tenors."""
    oat = common.load_base()["zero"]
    bench = _get_benchmark(tenors, benchmark, benchmark_csv)
    return oat, bench


# --------------------------------------------------------------------------- #
#  Data-driven shock detection                                                #
# --------------------------------------------------------------------------- #
def detect_shocks(spread_bp: pd.Series, k: float = 3.0, window: int = 60) -> pd.DataFrame:
    """
    Flag days whose 1-day spread change exceeds ``k`` rolling standard
    deviations - the market's own view of which days were shocks, independent of
    any hand-coded calendar.  Returns the flagged days sorted by magnitude.
    """
    d = spread_bp.diff()
    sd = d.rolling(window, min_periods=window // 2).std()
    z = d / sd
    flagged = pd.DataFrame({"spread_bp": spread_bp, "change_bp": d, "zscore": z})
    flagged = flagged[z.abs() >= k].dropna()
    return flagged.reindex(flagged["change_bp"].abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- #
#  Event study (cumulative abnormal spread change)                            #
# --------------------------------------------------------------------------- #
def event_study(spread_bp: pd.Series, events: dict = None,
                windows=((0, 1), (0, 5), (0, 10)),
                est_window: int = 60, gap: int = 6) -> pd.DataFrame:
    """
    For each event, the cumulative *abnormal* change in the spread over several
    post-event windows.  "Normal" daily drift mu and volatility sigma are
    estimated on the ``est_window`` trading days ending ``gap`` days before the
    event; the abnormal cumulative change over an n-day window is
    ``sum(dSpread) - n*mu`` with t-stat ``CASC / (sigma*sqrt(n))``.
    """
    ev = common.events_frame(events)
    d = spread_bp.diff()
    idx = spread_bp.index
    rows = []
    for date, info in ev.iterrows():
        e = common.nearest_trading_day(idx, date)
        ei = idx.get_loc(e)
        est = d.iloc[max(0, ei - gap - est_window):max(0, ei - gap)]
        if len(est) < est_window // 2:
            continue
        mu, sigma = est.mean(), est.std()
        rec = {"event": info["label"], "tier": int(info["tier"]),
               "date": e.date(), "spread_pre_bp": round(spread_bp.iloc[ei - 1], 1)}
        for a, b in windows:
            seg = d.iloc[ei + a: ei + b + 1]
            n = len(seg)
            casc = seg.sum() - n * mu
            t = casc / (sigma * np.sqrt(n)) if (sigma > 0 and n) else np.nan
            rec[f"CASC[{a},{b}]"] = round(casc, 1)
            rec[f"t[{a},{b}]"] = round(t, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  "Core vs middle-ground": rolling OAT-Bund beta and correlation             #
# --------------------------------------------------------------------------- #
def rolling_beta_corr(tenor: int = 10, window: int = 60,
                      benchmark="bund", benchmark_csv=None) -> pd.DataFrame:
    """
    Rolling correlation and beta of daily OAT yield changes on benchmark (Bund)
    yield changes at ``tenor``.  Beta ~ 1 and corr ~ 1 => the OAT trades as a
    core asset comoving with the Bund; a fall in beta/corr signals decoupling
    (a risk repricing of France).
    """
    oat, bench = benchmark_levels(tenors=(tenor,), benchmark=benchmark,
                                  benchmark_csv=benchmark_csv)
    key = f"{tenor}y"
    pair = common.align(oat[key].rename("oat"), bench[key].rename("bund")).diff().dropna()
    cov = pair["oat"].rolling(window).cov(pair["bund"])
    var = pair["bund"].rolling(window).var()
    corr = pair["oat"].rolling(window).corr(pair["bund"])
    return pd.DataFrame({"beta": cov / var, "corr": corr}).dropna()


# --------------------------------------------------------------------------- #
#  Brexit-style regression with HAC (Newey-West) standard errors              #
# --------------------------------------------------------------------------- #
def ols_hac(y: np.ndarray, X: np.ndarray, lags: int = 5):
    """OLS with Newey-West HAC standard errors.  Returns beta, se, t, r2."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    S = (resid[:, None] * X).T @ (resid[:, None] * X)            # lag 0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = (resid[L:, None] * X[L:]).T @ (resid[:-L, None] * X[:-L])
        S += w * (u + u.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
    return beta, se, t, r2


def event_regression(spread_bp: pd.Series, events: dict = None,
                     event_window: int = 2, hac_lags: int = 5,
                     split_date: str = "2024-06-09") -> dict:
    """
    Estimate the UK/Brexit-style model on the daily spread change:

        dSpread_t = a + g*EventWin_t + d*dCISS_t + phi*Spread_{t-1} + e_t

    EventWin_t is 1 on each event day and the following ``event_window`` days.
    Also reports a pre/post structural comparison around ``split_date`` (the
    dissolution): spread level, volatility and OAT-Bund 10y beta in each regime.
    Returns a dict with the regression table and the structural summary.
    """
    ev = common.events_frame(events)
    ciss = external_data.fetch_ciss()
    df = common.align(spread_bp.rename("spread"), ciss.rename("ciss"))
    df["dspread"] = df["spread"].diff()
    df["dciss"] = df["ciss"].diff()
    df["spread_lag"] = df["spread"].shift(1)

    # event-window dummy
    dummy = pd.Series(0, index=df.index)
    for date in ev.index:
        e = common.nearest_trading_day(df.index, date)
        ei = df.index.get_loc(e)
        dummy.iloc[ei: ei + event_window + 1] = 1
    df["event"] = dummy
    reg = df.dropna()

    y = reg["dspread"].to_numpy()
    X = reg[["event", "dciss", "spread_lag"]].to_numpy()
    beta, se, t, r2 = ols_hac(y, X, lags=hac_lags)
    table = pd.DataFrame(
        {"coef": beta, "se": se, "t": t},
        index=["const", "event_dummy", "d_CISS", "spread_lag"]).round(3)
    table.attrs["r2"] = round(r2, 3)
    table.attrs["nobs"] = len(reg)

    # structural pre/post comparison (level, vol, 10y beta)
    bc = rolling_beta_corr(10).reindex(spread_bp.index).ffill()
    split = pd.Timestamp(split_date)
    struct = {}
    for label, sl in [("pre", spread_bp[spread_bp.index < split]),
                      ("post", spread_bp[spread_bp.index >= split])]:
        bsl = bc["beta"][bc.index.isin(sl.index)]
        struct[label] = {
            "spread_mean_bp": round(sl.mean(), 1),
            "spread_vol_bp": round(sl.diff().std(), 2),
            "beta_mean": round(bsl.mean(), 3),
        }
    return {"regression": table, "structural": pd.DataFrame(struct).T}
