"""
Shared infrastructure for the supplementary studies.

Loads the (read-only) cached results of the base OATmeals replication, defines
the output folders, a common plotting style, and the calendar of French
political-instability events used by the event study.  Nothing here modifies
the base ``oatcurve`` package or its outputs.
"""
from __future__ import annotations

import pandas as pd

from oatcurve import config, data_loading as dl, pipeline, plotstyle

# ---- output folders (kept separate from the base project's output) --------- #
FIG_DIR = config.OUTPUT_DIR / "studies_figures"
TABLE_DIR = config.OUTPUT_DIR / "studies_tables"
for _d in (FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Shared thesis look, single source of truth in ``oatcurve.plotstyle``:
# default font, light grid, Dark2 categorical palette.
FRANCE_BLUE = plotstyle.FRANCE_BLUE   # single-series accent (see the OAT notebook)
FRANCE_RED = plotstyle.FRANCE_RED     # paired with FRANCE_BLUE for two-series contrast


def apply_style():
    plotstyle.apply_theme()


# --------------------------------------------------------------------------- #
#  Load the base replication results (read-only)                              #
# --------------------------------------------------------------------------- #
def load_base(with_bonds: bool = False):
    """
    Return the cached base results as a dict:
        params : per-day Svensson params + fit diagnostics (mae, noise, ...)
        zero   : fitted OAT zero-coupon yields at standard tenors (decimals)
        (optionally) panel, master, bonds for cross-section work.
    Run ``scripts/01_build_panel.py`` and ``02_fit_curves.py`` first.
    """
    cache = config.CACHE_DIR
    out = {
        "params": pd.read_parquet(cache / "fit_params.parquet"),
        "zero": pd.read_parquet(cache / "zero_curve.parquet"),
    }
    if with_bonds:
        out["panel"] = pd.read_parquet(cache / "panel.parquet")
        out["master"] = pd.read_parquet(cache / "master.parquet")
        out["bonds"] = dl.build_bonds(out["master"], out["panel"], config.FILTERS)
    return out


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"  figure -> {path.name}")


# --------------------------------------------------------------------------- #
#  French political-instability event calendar                                #
# --------------------------------------------------------------------------- #
# Dated political events, 2024-2025.  The data-driven shock detector in
# ``political_risk.detect_shocks`` flags the market-moving days independently,
# so results do not hinge on this list being exhaustive.  ``tier`` marks how
# market-relevant an event is (1 = major), used only for plotting emphasis.
FRENCH_POLITICAL_EVENTS = {
    "2024-06-09": ("EU elections; Macron dissolves the National Assembly", 1),
    "2024-06-30": ("Legislative elections, 1st round", 2),
    "2024-07-07": ("Legislative elections, 2nd round (hung parliament)", 1),
    "2024-09-05": ("Barnier appointed Prime Minister", 2),
    "2024-12-04": ("Barnier government falls (no-confidence vote)", 1),
    "2024-12-13": ("Bayrou appointed Prime Minister", 2),
    # --- 2025 sequence ------------------------------------------------------ #
    "2025-02-05": ("2025 budget adopted; no-confidence motion fails", 2),
    "2025-09-08": ("Bayrou loses confidence vote (debt plan)", 1),
    "2025-09-09": ("Lecornu appointed Prime Minister", 2),
    "2025-09-12": ("Fitch downgrades France AA- to A+", 1),
    "2025-10-06": ("Lecornu resigns (<24h after naming cabinet)", 1),
    "2025-10-10": ("Lecornu reappointed (Lecornu II)", 2),
}


def events_frame(events: dict = None) -> pd.DataFrame:
    """Event calendar as a tidy, date-indexed DataFrame."""
    events = events or FRENCH_POLITICAL_EVENTS
    rows = [{"date": pd.Timestamp(d), "label": lbl, "tier": tier}
            for d, (lbl, tier) in events.items()]
    return pd.DataFrame(rows).set_index("date").sort_index()


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #
def align(*series, how: str = "inner") -> pd.DataFrame:
    """Align several date-indexed Series/DataFrames on common dates."""
    return pd.concat(series, axis=1, join=how).dropna()


def nearest_trading_day(index: pd.DatetimeIndex, date) -> pd.Timestamp:
    date = pd.Timestamp(date)
    pos = index.searchsorted(date)
    pos = min(pos, len(index) - 1)
    return index[pos]
