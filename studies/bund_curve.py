"""
German Bund zero-coupon curve - fitted with the *same* Svensson/GSW engine as
the OAT curve, so the OAT-Bund spread of Study 1 is computed from two internally
consistent curves (exactly the spirit of the paper's Section 7.1, which used a
German curve downloaded from Bloomberg).

Inputs are two Bloomberg exports parallel to the OAT ones (see data/README.md):
    Analisi Bunds 1.csv               - security master
    Analisi Bunds (Bid-Ask Data).csv  - daily bid/ask panel (1999-2026)

The German nominal universe spans Bundesschatzanweisungen (Schätze, ~2y),
Bundesobligationen (Bobls, ~5y) and Bundesanleihen (Bunds, 10-30y); all are
fixed-coupon EUR bullets, so the existing filters and the robust daily fit apply
unchanged.  Results are cached under ``output/studies_cache``.
"""
from __future__ import annotations

import pandas as pd

from oatcurve import config, data_loading as dl, pipeline

BUND_MASTER_CSV = config.DATA_DIR / "Analisi Bunds 1.csv"
BUND_BIDASK_CSV = config.DATA_DIR / "Analisi Bunds (Bid-Ask Data).csv"
CACHE = config.OUTPUT_DIR / "studies_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# German federal security types (by Bloomberg issuer name) - descriptive only.
_GERMAN_TYPE = {"bundesschatzanweisung": "Schatz", "bundesobligation": "Bobl",
                "bundesanleihe": "Bund"}


def load_bund_master() -> pd.DataFrame:
    """Bund security master, with ``security_type`` set to Schatz/Bobl/Bund."""
    m = dl.load_security_master(BUND_MASTER_CSV)

    def classify(name: str) -> str:
        low = name.lower()
        for key, val in _GERMAN_TYPE.items():
            if key in low:
                return val
        return "Bund"

    m["security_type"] = m["issuer"].map(classify)
    return m


def load_bund_data():
    """Return ``(master, panel, bonds)`` for the German nominal universe."""
    master = load_bund_master()
    panel = dl.load_bid_panel(BUND_BIDASK_CSV)
    bonds = dl.build_bonds(master, panel, config.FILTERS)
    return master, panel, bonds


def fit_and_cache(force: bool = False, progress: bool = True):
    """
    Fit the Bund Svensson curve on every trading day (1999-2026) and cache the
    parameters + reconstructed zero curve.  Returns ``(params, zero)``.
    """
    ppath, zpath = CACHE / "bund_params.parquet", CACHE / "bund_zero.parquet"
    if ppath.exists() and zpath.exists() and not force:
        return pd.read_parquet(ppath), pd.read_parquet(zpath)

    master, panel, bonds = load_bund_data()
    if progress:
        print(f"Fitting Bund curve: {len(panel)} days, {len(bonds)} bonds ...")
    params = pipeline.run_fit(bonds, panel, compute_metrics=True, progress=progress)
    tenors = sorted(set(config.STANDARD_TENORS) | set(config.PCA_TENORS))
    zero = pipeline.zero_curve(params, tenors)
    params.to_parquet(ppath)
    zero.to_parquet(zpath)
    return params, zero


def load_bund_zero(tenors=(2, 5, 10, 30)) -> pd.DataFrame:
    """Cached Bund zero-coupon yields (decimals) at the requested tenors;
    fits and caches the curve on first call."""
    zpath = CACHE / "bund_zero.parquet"
    zero = pd.read_parquet(zpath) if zpath.exists() else fit_and_cache()[1]
    return zero[[f"{t}y" for t in tenors]]


def load_bund_params() -> pd.DataFrame:
    """Cached per-day Bund Svensson parameters + fit diagnostics."""
    ppath = CACHE / "bund_params.parquet"
    return pd.read_parquet(ppath) if ppath.exists() else fit_and_cache()[0]
