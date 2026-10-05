"""
Step 3 - Reproduce the paper's figures and tables from the cached fit, and
build the "Part 1" comparison that documents how the results change when the
methodology is run up to today.

Run:  python scripts/03_make_outputs.py     (after 01 and 02)

Figures -> output/figures/*.png   Tables -> output/tables/*.csv
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oatcurve import (analysis, config, data_loading as dl, metrics, pipeline,
                      plotstyle, svensson)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

EURO = config.EURO_START
DOWN = pd.Timestamp(config.SP_DOWNGRADE)


def _load():
    params = pd.read_parquet(config.CACHE_DIR / "fit_params.parquet")
    zero = pd.read_parquet(config.CACHE_DIR / "zero_curve.parquet")
    par = pd.read_parquet(config.CACHE_DIR / "par_curve.parquet")
    master = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    panel = pd.read_parquet(config.CACHE_DIR / "panel.parquet")
    bonds = dl.build_bonds(master, panel, config.FILTERS)
    return params, zero, par, master, panel, bonds


def _save(fig, name):
    path = config.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  -> {path.name}")


def _nearest(index, date):
    date = pd.Timestamp(date)
    return index[np.argmin(np.abs(index - date))]


# --------------------------------------------------------------------------- #
#  Fig. 2 - maturity structure                                                #
# --------------------------------------------------------------------------- #
def fig_maturity_structure(master, panel):
    fig, ax = plt.subplots(figsize=(9, 5))
    kept = master.query("kept_for_fit")
    for _, b in kept.iterrows():
        s = panel[b["isin"]].dropna()
        if s.empty:
            continue
        first = s.index.min()
        mat = pd.Timestamp(b["maturity"])
        ymat = (mat - first).days / 365.25
        color = "tab:blue" if b["security_type"] == "OAT" else "tab:orange"
        ax.plot([first, mat], [ymat, 0], color=color, lw=0.5, alpha=0.6)
    ax.axvline(pd.Timestamp(EURO), color="k", ls="--", lw=1)
    ax.set(xlabel="Date", ylabel="Remaining time to maturity (years)",
           title="Fig. 2 - Maturity structure of the French OAT/BTAN universe")
    _save(fig, "fig02_maturity_structure.png")


# --------------------------------------------------------------------------- #
#  Fig. 3 - overall fitting error                                             #
# --------------------------------------------------------------------------- #
def fig_fitting_errors(params):
    fig, ax = plt.subplots(figsize=(9, 4))
    euro = params.loc[EURO:]
    ax.plot(euro.index, euro["mae"], lw=0.7, color="tab:blue")
    ax.axvline(DOWN, color="tab:red", ls=":", lw=1, label="S&P downgrade (Jan 2012)")
    ax.axvline(pd.Timestamp(config.PAPER_END), color="k", ls="--", lw=1,
               label="paper end (Apr 2018)")
    ax.set(xlabel="Year", ylabel="Mean absolute error (bp)",
           title="Fig. 3 - Overall fitting error (euro sample, extended to today)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig03_fitting_errors.png")


# --------------------------------------------------------------------------- #
#  Fig. 4 - maturity-specific fitting errors                                  #
# --------------------------------------------------------------------------- #
def fig_fitting_errors_by_bin(params):
    euro = params.loc[EURO:]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, label in zip(axes.ravel(), config.MATURITY_BIN_LABELS):
        col = f"mae_{label}"
        ax.plot(euro.index, euro[col], lw=0.6)
        ax.axvline(pd.Timestamp(config.PAPER_END), color="k", ls="--", lw=0.8)
        ax.set_title(f"{label}", fontsize=9)
        ax.set_ylabel("bp")
    fig.suptitle("Fig. 4 - Maturity-specific fitting errors (euro sample, extended)",
                 y=0.995)
    _save(fig, "fig04_fitting_errors_by_bin.png")


# --------------------------------------------------------------------------- #
#  Fig. 5 / 15 - par-yield curve snapshots                                    #
# --------------------------------------------------------------------------- #
def fig_par_snapshots(params, bonds, panel, dates, fname, title):
    n = len(dates)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.0 * n),
                             gridspec_kw={"width_ratios": [2, 1]})
    axes = np.atleast_2d(axes)
    for i, date in enumerate(dates):
        d = _nearest(params.index, date)
        theta = params.loc[d, pipeline.PARAM_COLS].to_numpy(float)
        table, _ = metrics.cross_section_fit(d, bonds, panel, theta=theta)
        if table is None:
            continue
        # Par yield is evaluated on an integer-year grid: with annual coupons
        # the par-bond structure is only consistent at whole maturities, so a
        # fractional grid would introduce a spurious saw-tooth.
        mmax = max(10.0, table["maturity"].max())
        grid = np.arange(1, int(np.ceil(mmax)) + 1)
        par = svensson.par_yield(grid, theta) * 100

        axL, axR = axes[i, 0], axes[i, 1]
        axL.plot(grid, par, color="k", lw=1.2, label="par yield (fitted)")
        axL.scatter(table["maturity"], table["y_obs"], s=18, facecolors="none",
                    edgecolors="tab:blue", label="observed YTM")
        axL.scatter(table["maturity"], table["y_fit"], s=14, marker="x",
                    color="tab:red", label="predicted YTM")
        axL.set(xlabel="Maturity (years)", ylabel="Yield (%)",
                title=f"{pd.Timestamp(d).date()}")
        if i == 0:
            axL.legend(frameon=False, fontsize=8)
        axR.scatter(table["maturity"], (table["y_obs"] - table["y_fit"]) * 100,
                    s=14, color=plotstyle.FRANCE_BLUE)   # YTM error in bp (paper Fig. 5)
        axR.axhline(0, color="k", lw=0.6)
        axR.set(xlabel="Maturity (years)", ylabel="Fitting error (bp)",
                title=f"Fitting error, {pd.Timestamp(d):%d-%b-%Y}")
    fig.suptitle(title, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _save(fig, fname)


# --------------------------------------------------------------------------- #
#  Zero yields + level / slope / curvature                                    #
# --------------------------------------------------------------------------- #
def fig_zero_yields(zero, span):
    # Mask tenors that are extrapolations of the traded range (e.g. the pre-1993
    # short end), so the plot shows only data-constrained yields.
    z = pipeline.mask_extrapolated(zero[["2y", "5y", "10y", "30y"]], span)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in ["2y", "5y", "10y", "30y"]:
        ax.plot(z.index, z[t] * 100, lw=0.7, label=t)
    ax.axvline(pd.Timestamp(EURO), color="k", ls="--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set(xlabel="Year", ylabel="Zero-coupon yield (%)",
           title="Fitted zero-coupon yields, 1987-2026 (extrapolated tenors masked)")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    _save(fig, "fig_zero_yields.png")


def fig_level_slope_curvature(zero):
    f = analysis.term_structure_factors(zero).loc[EURO:]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, col, ttl in zip(axes, ["level", "slope", "curvature"],
                            ["Level (10y)", "Slope (10y - 2y)",
                             "Curvature (2*5y - 2y - 10y)"]):
        ax.plot(f.index, f[col], lw=0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(pd.Timestamp(config.PAPER_END), color="k", ls="--", lw=0.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("%")
    fig.suptitle("Term-structure factors (euro sample, extended to today)", y=0.995)
    _save(fig, "fig_level_slope_curvature.png")


def fig_noise(params):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(params.index, params["noise"], lw=0.6, color="tab:purple")
    ax.axvline(pd.Timestamp(EURO), color="k", ls="--", lw=0.8, label="euro launch")
    ax.set(xlabel="Year", ylabel="Noise (bp)",
           title="Fig. 16 - HPW noise measure of the French bond market (full sample)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig16_noise_measure.png")


# --------------------------------------------------------------------------- #
#  Tables                                                                     #
# --------------------------------------------------------------------------- #
def table_fit_errors(params):
    periods = analysis.standard_periods(params.index.max())
    summary = analysis.fit_error_summary(params, periods)
    summary.to_csv(config.TABLE_DIR / "table1_fit_errors.csv")
    print(f"  table   -> table1_fit_errors.csv")
    return summary


def table_pca(zero):
    rows = {}
    for label, sl in [("Full extended", zero),
                      ("Euro extended", zero.loc[EURO:]),
                      ("Paper euro (1999-2018)", zero.loc[EURO:config.PAPER_END])]:
        ratio, _ = analysis.pca_decomposition(sl)
        rows[label] = ratio
    out = pd.DataFrame(rows).T
    out.to_csv(config.TABLE_DIR / "table4_pca.csv")
    print(f"  table   -> table4_pca.csv")
    return out


def table_extension_summary(params, zero):
    """Part 1 headline: how the key numbers move once we extend to today."""
    f = analysis.term_structure_factors(zero)
    windows = {
        "Paper euro (1999-2018)": (EURO, config.PAPER_END),
        "Post-paper (2018-today)": (config.PAPER_END, params.index.max()),
        "Euro extended (1999-today)": (EURO, params.index.max()),
    }
    rows = {}
    for label, (a, b) in windows.items():
        p = params.loc[a:b]
        ff = f.loc[a:b]
        rows[label] = {
            "n_days": len(p),
            "MAE_mean_bp": p["mae"].mean(),
            "MAE_max_bp": p["mae"].max(),
            "noise_mean_bp": p["noise"].mean(),
            "level_10y_mean_%": ff["level"].mean(),
            "slope_mean_%": ff["slope"].mean(),
            "n_bonds_mean": p["n_bonds"].mean(),
        }
    out = pd.DataFrame(rows).T
    out.to_csv(config.TABLE_DIR / "table_extension_summary.csv")
    print(f"  table   -> table_extension_summary.csv")
    return out


def main():
    params, zero, par, master, panel, bonds = _load()
    last = params.index.max()
    print(f"Loaded {len(params)} fitted days through {last.date()}\n")

    span = pipeline.maturity_span(bonds, panel, dates=params.index)
    span.to_parquet(config.CACHE_DIR / "maturity_span.parquet")

    print("Figures:")
    fig_maturity_structure(master, panel)
    fig_fitting_errors(params)
    fig_fitting_errors_by_bin(params)
    fig_par_snapshots(params, bonds, panel,
                      ["2003-03-25", "2008-06-10", "2018-04-02", last],
                      "fig05_par_snapshots.png",
                      "Fig. 5 - Par yield curve (euro sample, plus latest date)")
    fig_par_snapshots(params, bonds, panel,
                      ["1988-01-04", "1995-09-20", "1999-01-05"],
                      "fig15_par_snapshots_preeuro.png",
                      "Fig. 15 - Par yield curve (pre-euro sample)")
    fig_zero_yields(zero, span)
    fig_level_slope_curvature(zero)
    fig_noise(params)

    print("\nTables:")
    t1 = table_fit_errors(params)
    t4 = table_pca(zero)
    text = table_extension_summary(params, zero)

    print("\n=== Part 1: extension to today (headline) ===")
    with pd.option_context("display.width", 160, "display.float_format",
                           lambda v: f"{v:,.2f}"):
        print(text.to_string())
    print("\n=== Table 4: PCA explained variance ===")
    print(t4.round(4).to_string())
    print(f"\nAll outputs in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
