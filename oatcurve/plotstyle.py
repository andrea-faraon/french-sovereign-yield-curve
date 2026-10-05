"""Shared plotting style for the thesis figures.

A single import point so every notebook renders with the same look:
the paper's default (sans) font, a light academic grid, and the
ColorBrewer *Dark2* qualitative palette as the categorical colour cycle.

Usage
-----
    from oatcurve import plotstyle as ps
    ps.apply_theme()                      # call once, after importing pyplot
    color = ps.TENOR_COLORS[ps.tenor_bucket(orig_term)]
"""
import matplotlib.pyplot as plt

# --- categorical palette (ColorBrewer Dark2) ---------------------------------
# Used both as the default colour cycle and, explicitly, for the maturity
# buckets of the OAT/BTAN universe (Fig. 2).
DARK2 = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A",
         "#66A61E", "#E6AB02", "#A6761D", "#666666"]

# French-flag blue/red — high-contrast accents for single/two-series figures
FRANCE_BLUE = "#0055A4"
FRANCE_RED = "#EF4135"

# --- original-maturity buckets (Fig. 2) --------------------------------------
TENOR_BUCKETS = ["≤3Y", "5Y", "10Y", "15Y", "20–25Y", "30Y", "50Y"]
TENOR_COLORS = {
    "≤3Y":    "#1B9E77",
    "5Y":     "#D95F02",
    "10Y":    "#7570B3",
    "15Y":    "#E7298A",
    "20–25Y": "#66A61E",
    "30Y":    "#E6AB02",
    "50Y":    "#A6761D",
}


def tenor_bucket(x):
    """Map an original term (years) to its canonical OAT/BTAN tenor bucket."""
    if x < 4:    return "≤3Y"
    if x < 7.5:  return "5Y"
    if x < 12.5: return "10Y"
    if x < 17.5: return "15Y"
    if x < 27.5: return "20–25Y"
    if x < 40:   return "30Y"
    return "50Y"


def apply_theme():
    """Set the shared rcParams. Call once per notebook, after importing pyplot."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200,
        "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=DARK2),
        "legend.frameon": False,
    })
