"""
Data layer (Section 3): turn the two raw Bloomberg exports into

  1. a tidy security master (one row per ISIN), and
  2. a wide panel of daily *clean* bid prices (index = date, columns = ISIN),

then build the in-memory :class:`~oatcurve.bonds.Bond` objects after applying
the *static* filters of Section 4.3 (special features, currency, retail
tranche).  The maturity- and date-dependent filters live in ``estimation.py``
because they change every trading day.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config
from .bonds import Bond

# Bloomberg "missing value" tokens that may appear in price cells.
_NA_TOKENS = ["#N/A N/A", "#N/A Field Not Applicable", "#N/A Invalid Security",
              "#N/A Requesting Data...", "#N/A", "", " "]


# --------------------------------------------------------------------------- #
#  Security master                                                            #
# --------------------------------------------------------------------------- #
def load_security_master(path=None) -> pd.DataFrame:
    """Parse ``Analisi OATs 1.csv`` into a clean per-ISIN reference table."""
    path = path or config.MASTER_CSV
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    raw.columns = [c.strip() for c in raw.columns]

    df = pd.DataFrame({
        "isin": raw["ISIN"].str.strip(),
        "issuer": raw["Nome emittente"].str.strip(),
        "coupon": pd.to_numeric(raw["Cedola"], errors="coerce"),
        "issue": pd.to_datetime(raw["Data emissione"], dayfirst=True, errors="coerce"),
        "maturity": pd.to_datetime(raw["Scadenza"], dayfirst=True, errors="coerce"),
        "tipo_scad": raw["Tipo scad"].str.strip(),
        "series": raw["Serie"].str.strip(),
        "currency": raw["Valuta"].str.strip(),
    })
    # Drop junk rows (no ISIN, missing dates or coupon).
    df = df[(df["isin"] != "") & df["maturity"].notna() &
            df["issue"].notna() & df["coupon"].notna()].copy()

    # Classify legacy medium-term notes (BTANs) vs OATs.  Bloomberg's "Serie"
    # field is not a reliable tag here (e.g. the month-coded series are in fact
    # 10-year OATs), so we use the economic definition: BTANs were the
    # medium-term segment (2-5y) issued under that name until 1 Jan 2013.  A
    # security is labelled BTAN if it was issued before 2013 with an original
    # term to maturity of at most 7 years; everything else is an OAT.  This is
    # purely descriptive - both are priced identically as fixed-coupon bullets.
    df["orig_term"] = (df["maturity"] - df["issue"]).dt.days / 365.25
    df["security_type"] = np.where(
        (df["issue"] < pd.Timestamp("2013-01-01")) & (df["orig_term"] <= 7.0),
        "BTAN", "OAT")
    df["is_retail"] = df["issuer"].str.contains("Physiques", case=False, na=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Daily clean bid-price panel                                                #
# --------------------------------------------------------------------------- #
def load_bid_panel(path=None) -> pd.DataFrame:
    """
    Parse the wide ``Analisi OATs (Bid-Ask Data).csv`` export into a
    ``DataFrame`` of clean bid prices: index = trading date, columns = ISIN.

    The raw layout stores two rows per bond (Ask then Bid); the ISIN only
    appears on the Ask row, so it is forward-filled before the Bid rows are
    selected (in line with GSW, we use bid prices, Section 3).
    """
    path = path or config.BIDASK_CSV
    raw = pd.read_csv(path, dtype=str, header=0, encoding="utf-8",
                      encoding_errors="replace").fillna("")
    cols = list(raw.columns)
    raw.columns = ["BONDS", "PRICES", "FIELD"] + cols[3:]

    raw["BONDS"] = raw["BONDS"].replace("", np.nan).ffill()
    bid = raw[raw["PRICES"].str.strip() == "Bid Price"].copy()
    bid["isin"] = bid["BONDS"].str.replace(" Govt", "", regex=False).str.strip()

    date_cols = cols[3:]
    panel = bid.set_index("isin")[date_cols].T
    panel.index = pd.to_datetime(pd.Series(date_cols), dayfirst=True,
                                 errors="coerce").values
    panel = panel[panel.index.notna()].sort_index()

    # pd.to_numeric(..., errors="coerce") maps every Bloomberg NA token (and
    # blanks) to NaN, so no explicit token replacement is needed.
    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.dropna(axis=1, how="all").dropna(axis=0, how="all")
    panel = panel.loc[:, ~panel.columns.duplicated()]
    return panel


def drop_abnormal_quotes(panel: pd.DataFrame,
                         abnormal: dict | None = None) -> pd.DataFrame:
    """
    Filter 5 (Section 4.3): blank out the hand-checked abnormal quotes.  Each
    entry is either a single date string or an inclusive ``(start, end)`` range.
    """
    abnormal = config.FILTERS.abnormal_quotes if abnormal is None else abnormal
    panel = panel.copy()
    for isin, spec in abnormal.items():
        if isin not in panel.columns:
            continue
        for item in spec:
            if isinstance(item, (tuple, list)):
                mask = (panel.index >= pd.Timestamp(item[0])) & \
                       (panel.index <= pd.Timestamp(item[1]))
            else:
                mask = panel.index == pd.Timestamp(item)
            panel.loc[mask, isin] = np.nan
    return panel


# --------------------------------------------------------------------------- #
#  Build Bond objects (static filters of Section 4.3)                         #
# --------------------------------------------------------------------------- #
def build_bonds(master: pd.DataFrame, panel: pd.DataFrame,
                filters=None) -> dict[str, Bond]:
    """
    Construct :class:`Bond` objects for the securities that survive the
    *static* filters and that actually have price data.

    Static filters applied here:
      * (1) regular bonds only - drop CALL/SINK and PUTABLE special features;
      * (1) currency at issuance in {EUR, FRF, XEU};
      *     optional exclusion of the retail France-Physiques tranche.
    Floaters / linkers / STRIPS are not present in this data set (all rows are
    fixed-coupon OAT/BTAN bullets), but the currency and special-feature
    screens are kept for completeness.
    """
    filters = filters or config.FILTERS
    available = set(panel.columns)
    bonds: dict[str, Bond] = {}

    for _, row in master.iterrows():
        isin = row["isin"]
        if isin not in available:
            continue
        if filters.exclude_special_features and row["tipo_scad"] in ("CALL/SINK", "PUTABLE"):
            continue
        if row["currency"] not in filters.allowed_currencies:
            continue
        if (not filters.include_retail_oat) and row["is_retail"]:
            continue
        bonds[isin] = Bond(
            isin=isin,
            issue=row["issue"].date(),
            maturity=row["maturity"].date(),
            coupon=float(row["coupon"]),
            currency=row["currency"],
            security_type=row["security_type"],
        )
    return bonds


def load_all(filters=None):
    """Convenience loader: returns ``(master, panel, bonds)`` ready to fit."""
    filters = filters or config.FILTERS
    master = load_security_master()
    panel = drop_abnormal_quotes(load_bid_panel(), filters.abnormal_quotes)
    bonds = build_bonds(master, panel, filters)
    return master, panel, bonds
