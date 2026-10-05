"""
Central configuration: file paths, estimation parameters and the exact filter
choices described in Grishchenko, Moraux & Pakulyak (2020), Sections 3-4.

Everything that is a *modelling choice* lives here so the whole replication can
be re-run, audited and stress-tested by editing a single file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #
# Project root = the folder that contains the ``oatcurve`` package.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

# Raw Bloomberg exports.  Bloomberg data cannot be redistributed, so they are
# not part of the repository: place them in ``data/`` (see data/README.md) or
# point the OATCURVE_DATA_DIR environment variable at the folder holding them.
import os

_default_data = PROJECT_DIR / "data"
DATA_DIR = Path(os.environ.get("OATCURVE_DATA_DIR", _default_data))

MASTER_CSV = DATA_DIR / "Analisi OATs 1.csv"                 # security master
BIDASK_CSV = DATA_DIR / "Analisi OATs (Bid-Ask Data).csv"    # daily bid/ask panel

# Generated artefacts (parquet caches, figures, tables).
OUTPUT_DIR = PROJECT_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for _d in (OUTPUT_DIR, CACHE_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Day-count / pricing conventions                                            #
# --------------------------------------------------------------------------- #
# French government bonds (OATs/BTANs) pay ANNUAL coupons on an Actual/Actual
# (ICMA) basis.  We measure continuous-compounding horizons in years using
# Actual/365.25, which is the standard GSW convention and is internally
# consistent with the yield-to-maturity definition used throughout.
DAYS_PER_YEAR = 365.25
FACE_VALUE = 100.0          # prices are quoted per 100 of face
COUPON_FREQUENCY = 1        # annual coupons (eq. 3, footnote 14)


# --------------------------------------------------------------------------- #
#  Sample window                                                              #
# --------------------------------------------------------------------------- #
# The paper runs 22 Oct 1987 -> 10 Apr 2018, with the *euro benchmark* sample
# starting 1 Jan 1999.  Our Bloomberg extract extends to mid-2026, which is the
# whole point of "Part 1": re-running the methodology up to today.
SAMPLE_START = "1987-10-22"     # first day with >= 6 quotes (Section 3)
EURO_START = "1999-01-01"       # onset of the euro area (benchmark sample)
PAPER_END = "2018-04-10"        # last day in the published paper
SAMPLE_END = None               # None -> use the last date available in the data

# France lost its AAA (S&P) rating on this date - used to annotate figures.
SP_DOWNGRADE = "2012-01-09"


# --------------------------------------------------------------------------- #
#  Maturity bins for fitting-error diagnostics (Figs. 3-4, Table 1)           #
# --------------------------------------------------------------------------- #
MATURITY_BIN_EDGES = [0, 2, 5, 10, 20, 30, 50]            # years
MATURITY_BIN_LABELS = ["0-2yr", "2-5yr", "5-10yr",
                       "10-20yr", "20-30yr", "30-50yr"]

# Standard tenors (years) at which the fitted zero curve is sampled for the
# term-structure / PCA analysis (Section 5.2).
STANDARD_TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
PCA_TENORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]              # Table 4 uses 1-10y


# --------------------------------------------------------------------------- #
#  Filters (Section 4.3)                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Filters:
    """The five filters of Section 4.3, expressed as toggles/parameters."""

    # (3) Exclude securities with less than this many months to maturity.
    #     The paper uses 12 months (they note 18 months gives similar results).
    min_months_to_maturity: int = 12

    # (1) Exclude special-feature securities: floaters, linkers, callable /
    #     sinkable / putable bonds and STRIPS.  In this data set the only
    #     special-feature instruments carry Bloomberg "Tipo scad" of
    #     CALL/SINK or PUTABLE.
    exclude_special_features: bool = True

    # (1) Currencies allowed at issuance: French franc, ECU and euro.
    allowed_currencies: tuple = ("EUR", "FRF", "XEU")

    # Include the 17 retail "France-Physiques" OATs (tranches reserved to
    # individuals).  Default True ("all available public OATs"); set False for
    # a robustness check on the wholesale cross-section only.
    include_retail_oat: bool = True

    # (5) Hand-checked abnormal quotes to drop (ISIN -> list of date strings or
    #     (start, end) ranges).  These are exactly the ones flagged in the paper.
    abnormal_quotes: dict = field(default_factory=lambda: {
        "FR0000041410": [("1987-10-30", "1987-12-01")],   # 21 abnormal days
        "FR0000570509": ["2002-02-19", "2002-02-21", "2002-02-25"],
    })

    # (5, extended) Automated abnormal-quote screen.  The paper hand-checked
    # abnormal quotes over its 1987-2018 sample; extending to 2026 uncovers new
    # gross Bloomberg quote errors the authors never saw (e.g. a frozen price of
    # ~40 on FR0012938116/FR0013508470 in 2021-22, and ~1/10-scaled 1988 franc
    # quotes), which imply yields of 24-31%.  Since legitimate French nominal
    # yields never exceeded ~11% over 1987-2026, we drop any quote whose implied
    # yield-to-maturity falls outside this generous plausibility band.  This is
    # a data-quality filter only: it removes errors without touching any
    # economically plausible observation.
    plausible_ytm_range: tuple = (-0.05, 0.20)


FILTERS = Filters()

# Minimum number of bonds required to identify the 6 Svensson parameters
# (Section 3, footnote 13).
MIN_BONDS_PER_DAY = 6


# --------------------------------------------------------------------------- #
#  Optimiser settings (Section 4.4)                                           #
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    """Bounds and multi-start grid for the daily WLS calibration (eq. 14).

    Yields are handled in DECIMAL units (0.03 == 3%).  Constraints follow the
    paper: tau1, tau2 and beta0 are strictly positive; beta0+beta1 is left
    unconstrained so the short rate can go negative (post-2014 reality).
    """
    # Parameter order: (beta0, beta1, beta2, beta3, tau1, tau2)
    lower_bounds: tuple = (1e-6, -0.50, -1.00, -1.00, 1e-2, 1e-2)
    upper_bounds: tuple = (0.20,  0.50,  1.00,  1.00, 30.0, 40.0)

    # Warm start (previous trading day) is tried first.  If its fit is poor,
    # or on the very first day, we fall back to this small grid of starts.
    fallback_starts: tuple = (
        (0.03,  0.00,  0.00,  0.00,  1.5,  8.0),
        (0.05, -0.02,  0.00,  0.00,  2.0, 10.0),
        (0.02,  0.01,  0.01, -0.01,  1.0,  5.0),
        (0.08, -0.03,  0.02,  0.01,  3.0, 12.0),
        (0.01,  0.00, -0.01,  0.01,  0.5,  3.0),
    )
    # Trigger the multi-start search only when the warm-started fit is bad,
    # i.e. its dirty-price RMSE (in price points) exceeds this threshold.  Good
    # fits sit at ~0.3-0.8 price points (a few bp in yield), so 2.0 isolates
    # genuine convergence failures.  Warm-starting from the previous day (as in
    # GSW) is otherwise kept, which also yields smooth parameter paths.
    refit_rmse_threshold: float = 2.0
    max_nfev: int = 600

    # Robust outlier rejection (extends the paper's filter 5): a bond whose
    # fitted-vs-observed yield error exceeds max(outlier_bp, median + 6*MAD) is
    # treated as an abnormal quote, dropped, and the curve re-fitted.  The
    # 75 bp floor isolates off-curve data glitches in the low-error euro era
    # without disturbing genuine pre-euro noise.
    outlier_bp: float = 75.0


OPTIM = OptimConfig()
