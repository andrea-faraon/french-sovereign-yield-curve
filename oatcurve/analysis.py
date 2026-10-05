"""
Term-structure analysis - Section 5.2.

From the fitted zero-coupon curve we derive the three classic factors (level,
slope, curvature), run the principal-component decomposition of Table 4, and
build the period summary statistics of Table 1 (full vs euro vs post-paper
samples).  These are the tools used in "Part 1" to document how the results
change when the methodology is run all the way to today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
#  Level / slope / curvature (Section 5.2)                                    #
# --------------------------------------------------------------------------- #
def term_structure_factors(zero: pd.DataFrame) -> pd.DataFrame:
    """
    Level, slope and curvature from the fitted zero curve, following the paper:

      * level     = 10-year zero-coupon yield;
      * slope     = 10-year minus 2-year yield;
      * curvature = 2 * 5-year  -  (2-year + 10-year).

    ``zero`` must contain the columns ``2y``, ``5y`` and ``10y`` (decimals);
    the output is in percent.
    """
    y2, y5, y10 = zero["2y"], zero["5y"], zero["10y"]
    return pd.DataFrame({
        "level": y10 * 100,
        "slope": (y10 - y2) * 100,
        "curvature": (2 * y5 - y2 - y10) * 100,
    }, index=zero.index)


# --------------------------------------------------------------------------- #
#  Principal component decomposition (Table 4)                                #
# --------------------------------------------------------------------------- #
def pca_decomposition(zero: pd.DataFrame, tenors=None, n_components: int = 3):
    """
    PCA of the fitted 1-10y zero-coupon yields (Table 4).

    Returns ``(explained_ratio, loadings)`` where ``explained_ratio`` is the
    share of variance carried by each principal component and ``loadings`` are
    the eigenvectors (tenor x component).  Computed on demeaned yield levels.
    """
    tenors = tenors or config.PCA_TENORS
    cols = [f"{t}y" for t in tenors]
    X = zero[cols].dropna().to_numpy()
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    ratio = eigvals / eigvals.sum()
    loadings = pd.DataFrame(eigvecs[:, :n_components],
                            index=cols,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    return pd.Series(ratio[:n_components],
                     index=[f"PC{i+1}" for i in range(n_components)],
                     name="explained_variance_ratio"), loadings


# --------------------------------------------------------------------------- #
#  Fitting-error summary statistics (Table 1)                                 #
# --------------------------------------------------------------------------- #
def fit_error_summary(metrics_df: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """
    Build a Table-1-style summary of the fitting errors for each named period.

    ``periods`` maps a label -> ``(start, end)`` (either bound may be ``None``).
    For every period we report the mean, standard deviation and maximum of the
    overall MAE and of each maturity-bin MAE, all in basis points.
    """
    bin_cols = ["mae"] + [f"mae_{lbl}" for lbl in config.MATURITY_BIN_LABELS]
    records = {}
    for label, (start, end) in periods.items():
        sl = metrics_df.loc[
            (metrics_df.index >= (start or metrics_df.index.min())) &
            (metrics_df.index <= (end or metrics_df.index.max()))]
        stats = {}
        for c in bin_cols:
            stats[(c, "mean")] = sl[c].mean()
            stats[(c, "std")] = sl[c].std()
            stats[(c, "max")] = sl[c].max()
        stats[("n_days", "")] = len(sl)
        records[label] = stats
    out = pd.DataFrame(records).T
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def standard_periods(last_date) -> dict:
    """The canonical comparison windows used throughout the project."""
    return {
        "Full (1987-2018, paper)": (config.SAMPLE_START, config.PAPER_END),
        "Euro (1999-2018, paper)": (config.EURO_START, config.PAPER_END),
        "Pre-euro (1987-1998)": (config.SAMPLE_START, "1998-12-31"),
        "Post-paper (2018-today)": (config.PAPER_END, last_date),
        "Euro extended (1999-today)": (config.EURO_START, last_date),
        "Full extended (1987-today)": (config.SAMPLE_START, last_date),
    }
