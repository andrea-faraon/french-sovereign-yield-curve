"""
Study 2 - PCA of the curve across monetary regimes & the 2022-2024 inversion
(extends Section 5.3, hardened following Holtz, 2023 and Lord & Pelsser, 2007).

What this module adds over a plain level-PCA:

* **PCA on yield CHANGES as well as levels.**  Litterman-Scheinkman (1991) factor
  analysis is properly done on *changes*: yield *levels* share a strong common
  trend, so PC1 trivially dominates and the slope's role is invisible.  On daily
  changes the slope (PC2) share jumps in 2022-2024, confirming the hypothesis.
* **Lord & Pelsser (2007) sign-change test.**  The level/slope/curvature reading
  is valid only if the first three loading vectors have respectively 0, 1 and 2
  sign changes.  We certify (or reject) every regime - and find the test passes
  on each homogeneous sub-regime but fails on the full multi-regime change-PCA,
  which is itself the statistical justification for analysing regime by regime.
* **Window-length robustness (Holtz, 2023).**  Decompositions are unreliable over
  short spans, and it is the *length of time* that matters, not the number of
  samples.  We verify this by re-running with weekly (not daily) data, and flag
  short regimes (e.g. the ~1.5-year 2025 window).
* **Covariance estimator choice (Holtz, 2023).**  An optional constant-correlation
  Ledoit-Wolf shrinkage estimator, which Holtz finds yields more reliable
  loadings on short windows than the sample covariance.

All rates are decimals; PCA is on the demeaned (optionally standardised) data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oatcurve import config

from . import common

# Monetary-policy regimes (the 2022-2024 block is the inflation/tightening one).
REGIMES = {
    "Convergence (1999-2007)": ("1999-01-01", "2007-12-31"),
    "GFC & sovereign crisis (2008-2012)": ("2008-01-01", "2012-12-31"),
    "Low / negative rates (2013-2021)": ("2013-01-01", "2021-12-31"),
    "Inflation & ECB tightening (2022-2024)": ("2022-01-01", "2024-12-31"),
    "Normalisation (2025-today)": ("2025-01-01", None),
}

# Holtz: decompositions need enough *elapsed time*; flag regimes below this.
MIN_RELIABLE_YEARS = 2.0


# --------------------------------------------------------------------------- #
#  Covariance estimators                                                      #
# --------------------------------------------------------------------------- #
def _sample_cov(Xc: np.ndarray) -> np.ndarray:
    return (Xc.T @ Xc) / Xc.shape[0]


def _constant_correlation_shrinkage(Xc: np.ndarray):
    """
    Ledoit-Wolf shrinkage toward a constant-correlation target (the estimator
    Holtz, 2023 finds most reliable for yield-curve PCA).  Returns
    ``(Sigma_shrunk, delta)`` with optimal intensity ``delta`` in [0, 1].
    """
    T, N = Xc.shape
    S = (Xc.T @ Xc) / T
    var = np.diag(S).copy()
    std = np.sqrt(var)
    corr = S / np.outer(std, std)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(std, std)
    np.fill_diagonal(F, var)

    Y = Xc ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = pi_mat.sum()

    term = (Xc ** 3).T @ Xc / T                     # term[i,j] = E[x_i^3 x_j]
    theta_ii_ij = term - var[:, None] * S           # (x_i^2 - s_ii)(x_i x_j - s_ij)
    theta_jj_ij = term.T - var[None, :] * S
    sqrt_ratio = np.sqrt(np.outer(var, 1.0 / var))  # sqrt(var_i / var_j)
    off = (rbar / 2.0) * ((1.0 / sqrt_ratio) * theta_ii_ij + sqrt_ratio * theta_jj_ij)
    np.fill_diagonal(off, 0.0)
    rho_hat = np.diag(pi_mat).sum() + off.sum()

    gamma_hat = ((F - S) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / T))
    return delta * F + (1.0 - delta) * S, delta


# --------------------------------------------------------------------------- #
#  PCA core + Lord (2007) sign-change test                                    #
# --------------------------------------------------------------------------- #
def _pca(X: np.ndarray, estimator: str = "sample", standardize: bool = False):
    """Return ``(explained_ratio, loadings)`` (loadings columns = PC1, PC2, …)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if standardize:
        Xc = Xc / Xc.std(axis=0, keepdims=True)
    if estimator == "cc_shrink":
        cov, _ = _constant_correlation_shrinkage(Xc)
    else:
        cov = _sample_cov(Xc)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    return w / w.sum(), V


def sign_changes(vec: np.ndarray) -> int:
    """Number of sign changes along a loading vector (sign-flip invariant)."""
    s = np.sign(vec[np.abs(vec) > 1e-10])
    return int((np.diff(s) != 0).sum())


def lord_test(loadings: np.ndarray) -> tuple:
    """
    Lord & Pelsser (2007): the level/slope/curvature interpretation is valid iff
    the first three loading vectors have 0, 1 and 2 sign changes respectively.
    Returns ``(counts, is_valid)``.
    """
    counts = [sign_changes(loadings[:, k]) for k in range(3)]
    return counts, counts == [0, 1, 2]


# --------------------------------------------------------------------------- #
#  Data preparation                                                           #
# --------------------------------------------------------------------------- #
def _prepare(zero: pd.DataFrame, on: str, tenors, freq: str = "D") -> pd.DataFrame:
    """Tenor block as levels or daily/weekly changes."""
    cols = [f"{t}y" for t in (tenors or config.PCA_TENORS)]
    X = zero[cols].dropna()
    if freq == "W":
        X = X.resample("W-FRI").last().dropna()
    if on == "changes":
        X = X.diff().dropna()
    return X


def _slice(zero, start, end):
    return zero.loc[(start or zero.index.min()):(end or zero.index.max())]


# --------------------------------------------------------------------------- #
#  Regime decomposition (levels or changes) with Lord certification           #
# --------------------------------------------------------------------------- #
def pca_by_regime(zero=None, regimes=None, on: str = "changes", tenors=None,
                  estimator: str = "sample", standardize: bool = False) -> pd.DataFrame:
    """
    Explained-variance shares of PC1/PC2/PC3 per regime, on ``on`` in
    {'changes','levels'}, with the Lord sign-change certification.
    """
    zero = common.load_base()["zero"] if zero is None else zero
    regimes = regimes or REGIMES
    rows = {}
    for label, (a, b) in regimes.items():
        X = _prepare(_slice(zero, a, b), on, tenors)
        if len(X) < 30:
            continue
        ratio, load = _pca(X.to_numpy(), estimator, standardize)
        counts, valid = lord_test(load)
        years = (X.index.max() - X.index.min()).days / 365.25
        rows[label] = {"PC1 (level)": ratio[0], "PC2 (slope)": ratio[1],
                       "PC3 (curv.)": ratio[2], "n_obs": len(X),
                       "years": round(years, 1), "lord_signs": str(counts),
                       "lord_valid": valid,
                       "reliable_len": years >= MIN_RELIABLE_YEARS}
    return pd.DataFrame(rows).T


def levels_vs_changes(zero=None, regimes=None, tenors=None,
                      estimator: str = "sample") -> pd.DataFrame:
    """
    Headline comparison: the PC2 (slope) share under the *levels* metric vs the
    *changes* metric, per regime, with Lord validity for each.  The slope story
    is visible on changes, not on levels.
    """
    lv = pca_by_regime(zero, regimes, on="levels", tenors=tenors, estimator=estimator)
    ch = pca_by_regime(zero, regimes, on="changes", tenors=tenors, estimator=estimator)
    out = pd.DataFrame({
        "PC2 levels (%)": (lv["PC2 (slope)"].astype(float) * 100).round(2),
        "Lord levels": lv["lord_valid"],
        "PC2 changes (%)": (ch["PC2 (slope)"].astype(float) * 100).round(2),
        "Lord changes": ch["lord_valid"],
        "years": lv["years"],
    })
    return out


def pca_loadings(zero=None, start="2022-01-01", end="2024-12-31", on: str = "changes",
                 tenors=None, estimator: str = "sample") -> pd.DataFrame:
    """PC loadings (tenor x component) for one regime - the factor shapes."""
    zero = common.load_base()["zero"] if zero is None else zero
    X = _prepare(_slice(zero, start, end), on, tenors)
    _, load = _pca(X.to_numpy(), estimator)
    cols = [f"{t}y" for t in (tenors or config.PCA_TENORS)]
    return pd.DataFrame(load[:, :3], index=cols, columns=["PC1", "PC2", "PC3"])


# --------------------------------------------------------------------------- #
#  Inversion diagnostics (unchanged)                                          #
# --------------------------------------------------------------------------- #
def inversion_stats(zero=None, regimes=None) -> pd.DataFrame:
    """Curve-inversion diagnostics per regime (slope = 10y - 2y)."""
    zero = common.load_base()["zero"] if zero is None else zero
    regimes = regimes or REGIMES
    rows = {}
    for label, (a, b) in regimes.items():
        sl = _slice(zero, a, b)
        if len(sl) < 50:
            continue
        slope_bp = (sl["10y"] - sl["2y"]) * 1e4
        rows[label] = {
            "slope_mean_bp": round(slope_bp.mean(), 1),
            "slope_min_bp": round(slope_bp.min(), 1),
            "pct_days_inverted": round((slope_bp < 0).mean() * 100, 1),
            "slope_vol_bp": round(slope_bp.diff().std(), 2),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
#  Rolling shares + rolling Lord validity                                     #
# --------------------------------------------------------------------------- #
def rolling_pc_shares(zero=None, window: int = 252, on: str = "changes",
                      tenors=None, estimator: str = "sample") -> pd.DataFrame:
    """
    Rolling PCA on a moving window: PC1/PC2/PC3 variance shares plus a
    ``lord_valid`` flag (Holtz: short windows are unreliable, so the flag warns
    when the level/slope/curvature reading breaks down).
    """
    zero = common.load_base()["zero"] if zero is None else zero
    X = _prepare(zero, on, tenors)
    out, idx = [], []
    for i in range(window, len(X) + 1):
        ratio, load = _pca(X.iloc[i - window:i].to_numpy(), estimator)
        _, valid = lord_test(load)
        out.append([ratio[0], ratio[1], ratio[2], valid])
        idx.append(X.index[i - 1])
    return pd.DataFrame(out, index=pd.DatetimeIndex(idx, name="date"),
                        columns=["PC1", "PC2", "PC3", "lord_valid"])


def window_length_robustness(zero=None, regimes=None, tenors=None) -> pd.DataFrame:
    """
    Holtz check: re-estimate the PC2 (changes) share per regime using **weekly**
    data and compare with the daily estimate.  If the two are close, the result
    is driven by the *length* of the regime, not the *number* of samples - so a
    regime with enough elapsed time (>= ~2 years) is trustworthy even with fewer
    observations, while a short regime (e.g. 2025) is flagged regardless.
    """
    zero = common.load_base()["zero"] if zero is None else zero
    regimes = regimes or REGIMES
    rows = {}
    for label, (a, b) in regimes.items():
        sl = _slice(zero, a, b)
        Xd = _prepare(sl, "changes", tenors, freq="D")
        Xw = _prepare(sl, "changes", tenors, freq="W")
        if len(Xd) < 30 or len(Xw) < 15:
            continue
        pc2_d = _pca(Xd.to_numpy())[0][1]
        pc2_w = _pca(Xw.to_numpy())[0][1]
        years = (sl.index.max() - sl.index.min()).days / 365.25
        rows[label] = {
            "PC2 daily (%)": round(pc2_d * 100, 2),
            "PC2 weekly (%)": round(pc2_w * 100, 2),
            "n_daily": len(Xd), "n_weekly": len(Xw),
            "years": round(years, 1),
            "reliable_len": years >= MIN_RELIABLE_YEARS,
        }
    return pd.DataFrame(rows).T
