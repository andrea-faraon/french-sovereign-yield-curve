"""
External market data for the supplementary studies (auto-downloaded, cached).

These studies extend the OATmeals replication to the most recent years, so they
need a euro-area "safe" benchmark curve (for the OAT-Bund-style safety premium,
Section 7.1) and a risk-aversion proxy (for the Brexit-style event-study
regression).  Both come from the **ECB Data Portal** SDMX API and are estimated
with the *same* Svensson / continuous-compounding methodology as our OAT curve,
which keeps the spread internally consistent:

  * ``fetch_aaa_curve`` - the AAA-rated euro-area government spot curve
    (``YC ... G_N_A``).  Germany dominates the euro AAA bucket, so this is the
    standard euro-area safe benchmark (a clean proxy for the Bund curve used in
    the paper).  Available daily from 2004-09-06.
  * ``fetch_ciss`` - the Composite Indicator of Systemic Stress, the ECB's
    daily euro-area systemic-stress / risk-aversion measure.

Everything is cached under ``output/studies_cache`` so the notebooks run offline
after the first download.  A user-supplied Bloomberg German curve can be dropped
in instead via :func:`load_benchmark_curve`.
"""
from __future__ import annotations

import io
import os
import ssl
import urllib.request

import pandas as pd

from oatcurve import config

CACHE = config.OUTPUT_DIR / "studies_cache"
CACHE.mkdir(parents=True, exist_ok=True)

_ECB = "https://data-api.ecb.europa.eu/service/data"

# TLS certificates are verified by default.  Behind a TLS-intercepting proxy or
# antivirus (whose root certificate Python does not trust) set
# ECB_SSL_VERIFY=0 to disable verification for these public, read-only calls.
_SSL = ssl.create_default_context()
if os.environ.get("ECB_SSL_VERIFY", "1") == "0":
    _SSL.check_hostname = False
    _SSL.verify_mode = ssl.CERT_NONE

# ECB AAA spot-rate keys per tenor (Svensson model, continuous compounding).
AAA_KEYS = {1: "SR_1Y", 2: "SR_2Y", 3: "SR_3Y", 5: "SR_5Y",
            7: "SR_7Y", 10: "SR_10Y", 15: "SR_15Y", 20: "SR_20Y", 30: "SR_30Y"}
CISS_KEY = "D.U2.Z0Z.4F.EC.SS_CIN.IDX"   # flow ("CISS") is added by _fetch_ecb


def _http_get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        return r.read().decode("utf-8-sig", "replace")


def _parse_periods(periods: pd.Series) -> pd.Series:
    """Parse ECB TIME_PERIOD: daily/monthly parse directly; ISO weekly
    (e.g. ``1998-W53``) is mapped to the Monday of that week."""
    p = periods.astype(str)
    if p.str.contains("W").any():
        return pd.to_datetime(p + "-1", format="%G-W%V-%u", errors="coerce")
    return pd.to_datetime(p, errors="coerce")


def _fetch_ecb(flow: str, key: str) -> pd.Series:
    """Fetch one ECB SDMX series as a date-indexed Series (raw units)."""
    url = f"{_ECB}/{flow}/{key}?format=csvdata"
    df = pd.read_csv(io.StringIO(_http_get(url)))
    s = (df.assign(date=_parse_periods(df["TIME_PERIOD"]))
           .set_index("date")["OBS_VALUE"].sort_index())
    s.name = key
    return s


# --------------------------------------------------------------------------- #
#  Euro-area government curves (AAA = safe benchmark; all-bonds = incl. periphery)
# --------------------------------------------------------------------------- #
def _fetch_curve(rating: str, fname: str, tenors, force: bool) -> pd.DataFrame:
    """Daily ECB spot curve (decimals) for a rating group (G_N_A / G_N_C)."""
    path = CACHE / fname
    if path.exists() and not force:
        cached = pd.read_parquet(path)
        if all(f"{t}y" in cached.columns for t in tenors):
            return cached[[f"{t}y" for t in tenors]]
    try:
        cols = {f"{t}y": _fetch_ecb("YC", f"B.U2.EUR.4F.{rating}.SV_C_YM.{AAA_KEYS[t]}") / 100.0
                for t in tenors}
        out = pd.DataFrame(cols).sort_index()
        out.to_parquet(path)
        return out
    except Exception as exc:
        if path.exists():
            print(f"[external_data] download failed ({exc}); using cached {fname}.")
            return pd.read_parquet(path)[[f"{t}y" for t in tenors]]
        raise RuntimeError(f"Could not download ECB curve {rating} and no cache exists.") from exc


def fetch_aaa_curve(tenors=(2, 5, 10, 30), force: bool = False) -> pd.DataFrame:
    """Daily AAA euro-area zero-coupon yields (decimals) - the safe benchmark."""
    return _fetch_curve("G_N_A", "ecb_aaa_curve.parquet", tenors, force)


def fetch_allbonds_curve(tenors=(2, 5, 10, 30), force: bool = False) -> pd.DataFrame:
    """Daily *all* euro-area government zero-coupon yields (decimals); includes
    the periphery, so it sits above the AAA curve."""
    return _fetch_curve("G_N_C", "ecb_allbonds_curve.parquet", tenors, force)


def fetch_periphery_dispersion(tenors=(2, 5, 10, 30), force: bool = False) -> pd.DataFrame:
    """
    Daily euro-area sovereign risk dispersion = all-bonds minus AAA curve, in
    **basis points**, per tenor.  A clean, fully ECB-sourced proxy for systemic
    peripheral (Italy/Spain-type) risk, used as the *common-euro* factor in the
    spread decomposition (no daily country-level data needed).
    """
    allb = fetch_allbonds_curve(tenors, force)
    aaa = fetch_aaa_curve(tenors, force)
    allb, aaa = allb.align(aaa, join="inner")      # common trading days only
    return (allb - aaa) * 1e4


def fetch_ciss(force: bool = False) -> pd.Series:
    """Daily ECB Composite Indicator of Systemic Stress (risk-aversion proxy)."""
    path = CACHE / "ecb_ciss.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)["ciss"]
    try:
        s = _fetch_ecb("CISS", CISS_KEY).rename("ciss")
        s.to_frame().to_parquet(path)
        return s
    except Exception as exc:
        if path.exists():
            print(f"[external_data] download failed ({exc}); using cached CISS.")
            return pd.read_parquet(path)["ciss"]
        raise RuntimeError("Could not download ECB CISS and no cache exists.") from exc


def _cached_series(name: str, flow: str, key: str, scale: float = 1.0,
                   force: bool = False) -> pd.Series:
    """Generic cache-or-fetch for a single ECB macro series."""
    path = CACHE / f"{name}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)[name]
    try:
        s = (_fetch_ecb(flow, key) * scale).rename(name)
        s.to_frame().to_parquet(path)
        return s
    except Exception as exc:
        if path.exists():
            print(f"[external_data] download failed ({exc}); using cached {name}.")
            return pd.read_parquet(path)[name]
        raise RuntimeError(f"Could not download {name} and no cache exists.") from exc


def fetch_ecb_balance_sheet(force: bool = False) -> pd.Series:
    """ECB total assets, weekly, in EUR billions (Eurosystem balance sheet -
    the QE/QT measure used in the INSEE Stage-1/Stage-2 models)."""
    return _cached_series("ecb_balance_sheet", "ILM",
                          "W.U2.C.T000000.Z5.Z01", scale=1e-3, force=force)  # EUR mn -> bn


def fetch_euribor_3m(force: bool = False) -> pd.Series:
    """3-month Euribor, monthly (%), the short-rate / base-rate proxy."""
    return _cached_series("euribor_3m", "FM",
                          "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA", force=force)


def fetch_mir_nfc_rate(force: bool = False) -> pd.Series:
    """Rate on new loans to French non-financial corporations, monthly (%),
    from ECB MIR statistics - the dependent variable of INSEE Stage-2."""
    return _cached_series("mir_nfc_fr", "MIR",
                          "M.FR.B.A2A.A.R.A.2240.EUR.N", force=force)


def load_benchmark_curve(csv_path=None, tenors=(2, 5, 10, 30)) -> pd.DataFrame:
    """
    Load the safe-benchmark curve, in **decimals**, indexed by date.

    Default: the ECB AAA euro-area curve (auto-downloaded).  To use a pure
    Bloomberg German (Bund) curve instead, pass ``csv_path`` to a file with a
    ``date`` column and yield columns named ``2y, 5y, 10y, 30y`` (in percent);
    set the values' scale with care - this loader divides by 100 if the median
    magnitude looks like percent.
    """
    if csv_path is None:
        return fetch_aaa_curve(tenors)
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[[f"{t}y" for t in tenors]]
    if df.stack().abs().median() > 1.0:    # looks like percent -> to decimals
        df = df / 100.0
    return df
