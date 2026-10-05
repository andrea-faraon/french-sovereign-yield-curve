"""
Generator for the three supplementary-study notebooks (valid nbformat-4 JSON,
no extra dependencies).  Run:  python notebooks/build_studies_notebooks.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def nb_init():
    return []


def md(cells, text):
    cells.append({"cell_type": "markdown", "metadata": {}, "id": f"c{len(cells)}",
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(cells, text):
    cells.append({"cell_type": "code", "metadata": {}, "id": f"c{len(cells)}",
                  "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


def write(cells, name):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.13"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {name} ({len(cells)} cells)")


SETUP = r"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from studies import common
common.apply_style()
pd.set_option("display.width", 170)
"""

# =========================================================================== #
#  STUDY 1 - Political risk & OAT-Bund safety premium                         #
# =========================================================================== #
c = nb_init()
md(c, r"""
# Study 1 — Political risk and the OAT–Bund safety premium

**Extends Section 7.1 of Grishchenko, Moraux & Pakulyak (2020); event-study design after the UK/Brexit paper.**

France's *safety premium* is the spread of the fitted OAT zero-coupon curve over the **German Bund** curve — the euro-area safe benchmark. The Bund curve here is **fitted with the very same Svensson/GSW engine as the OATs** (from the parallel Bloomberg Bund export, 1999–2026), so the spread is computed from two internally consistent curves — the most faithful reading of the paper's Section 7.1. We then ask how the **2024–2026 political instability** (the dissolution of the National Assembly and the succession of governments under President Macron) repriced that premium.

Three lenses:
1. the **OAT–Bund spread** term structure (extends the paper's Fig. 12);
2. a **Brexit-style event study** — cumulative *abnormal* spread changes around political shocks, plus a regression with an ECB risk-aversion control (CISS);
3. the rolling **OAT-vs-Bund beta** — does the OAT still trade as a *core* asset (β≈1, comoving with the Bund) or has it drifted toward a riskier *middle-ground* asset (β falls, the level premium jumps)?

> Data: OAT and **Bund** curves from our own daily Svensson fits (run `scripts/02_fit_curves.py` and `scripts/05_fit_bund_curve.py` first); CISS auto-downloaded from the ECB (cached). The ECB AAA euro-area curve remains available as a cross-check via `benchmark="aaa"`. The 2024–25 event calendar is in `studies/common.FRENCH_POLITICAL_EVENTS` — the data-driven shock detector below flags the actual market-moving days regardless.
""")
code(c, SETUP + r"""
from studies import political_risk as pr
spreads = pr.build_spreads(tenors=(2,5,10,30))   # bp, OAT - Bund (both own Svensson fits)
events  = common.events_frame()
print("Spread sample:", spreads.index.min().date(), "->", spreads.index.max().date())
spreads.tail(3).round(1)
""")

md(c, "## 1. The OAT–Bund safety-premium term structure (extends Fig. 12)")
code(c, r"""
# Plotted from 2002: in the early euro-convergence years the OAT 2y is an
# extrapolation (no short OATs yet), so the 2y spread is unreliable then; the
# German 2y is always pinned by Schätze. The 5y/10y/30y are reliable from 1999.
s_plot = spreads.loc["2002":]
fig, ax = plt.subplots(figsize=(10,4.5))
for t in ["2y","5y","10y","30y"]:
    ax.plot(s_plot.index, s_plot[t], lw=.7, label=t)
ax.axhline(0, color="grey", lw=.5)
ax.set(xlabel="Year", ylabel="OAT - Bund spread (bp)",
       title="French safety premium (OAT minus German Bund), 2002-2026")
ax.legend(ncol=4, frameon=False, fontsize=8)
common.save_fig(fig, "study1_spread_termstructure.png"); plt.show()
print("10y spread: 2019 avg %.1f bp | 2024 avg %.1f bp | latest %.1f bp"
      % (spreads.loc['2019','10y'].mean(), spreads.loc['2024','10y'].mean(), spreads['10y'].iloc[-1]))
""")

md(c, r"""
**Robustness — Bund vs ECB AAA benchmark.** In calm periods and in the current
episode the two benchmarks give a nearly identical 10y premium (a few bp), which
cross-validates both. But during the **2008–2012 sovereign crisis the real
OAT–Bund spread is far wider** than OAT–AAA (peaking ~200 bp vs ~75 bp): flight
to quality compressed *Bund* yields specifically, whereas the AAA basket
(Netherlands, Austria, … alongside Germany) understates the premium to the
ultimate safe asset. So using the exact Bund — not the proxy — matters precisely
in the crisis periods, which is why we adopt it as the benchmark.
""")
code(c, r"""
sp_bund = spreads["10y"]
sp_aaa  = pr.build_spreads(tenors=(10,), benchmark="aaa")["10y"]
cmp = common.align(sp_bund.rename("OAT-Bund (Svensson)"), sp_aaa.rename("OAT-AAA (ECB)"))
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(cmp.index, cmp.iloc[:,0], lw=.8, label="OAT - Bund (own fit)")
ax.plot(cmp.index, cmp.iloc[:,1], lw=.8, label="OAT - AAA (ECB proxy)", alpha=.8)
ax.set(xlabel="Year", ylabel="10y spread (bp)", title="Safety premium: Bund vs AAA benchmark")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study1_bund_vs_aaa.png"); plt.show()
print("Mean abs difference (Bund vs AAA), 10y: %.1f bp" % (cmp.iloc[:,0]-cmp.iloc[:,1]).abs().mean())
""")

md(c, r"""
## 2. Zoom on the political crisis + data-driven shock detection

The detector flags days whose 1-day spread move exceeds 3 rolling standard deviations — the market's own verdict on which days were shocks.
""")
code(c, r"""
shocks = pr.detect_shocks(spreads["10y"], k=3)
recent = spreads.loc["2024-01-01":]
fig, ax = plt.subplots(figsize=(10,4.5))
ax.plot(recent.index, recent["10y"], color="tab:blue", lw=1)
for d, info in events.iterrows():
    if d >= recent.index.min():
        ax.axvline(d, color=("tab:red" if info.tier==1 else "grey"), ls="--", lw=.8)
        ax.text(d, ax.get_ylim()[1], " "+info.label[:22], rotation=90, va="top", ha="left", fontsize=6)
sh = shocks[shocks.index >= recent.index.min()]
ax.scatter(sh.index, sh["spread_bp"], color="tab:red", zorder=5, s=25, label="detected shock (>3 sd)")
ax.set(xlabel="Date", ylabel="10y OAT-Bund spread (bp)", title="10y safety premium and French political shocks (2024-2026)")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study1_10y_political_zoom.png"); plt.show()
print("Top data-driven shock days (10y spread):"); shocks.head(8).round(2)
""")

md(c, r"""
## 3. Event study — cumulative abnormal spread change (CASC)

For each event, the cumulative change in the spread over post-event windows, **net of** the normal drift estimated on the prior 60 trading days; *t*-stats use the estimation-window volatility. A positive, significant CASC means the event widened France's risk premium beyond normal fluctuations.
""")
code(c, r"""
es = pr.event_study(spreads["10y"], windows=((0,1),(0,5),(0,10)))
display(es)
es.to_csv(common.TABLE_DIR / "study1_event_study.csv", index=False)

fig, ax = plt.subplots(figsize=(9,4))
colors = ["tab:red" if t==1 else "tab:grey" for t in es["tier"]]
ax.barh(es["event"].str[:38], es["CASC[0,5]"], color=colors)
ax.axvline(0, color="k", lw=.6)
ax.set(xlabel="Cumulative abnormal spread change over [0,+5] days (bp)",
       title="Event study: 10y OAT-Bund reaction to French political events")
ax.invert_yaxis()
common.save_fig(fig, "study1_event_study_bars.png"); plt.show()
""")

md(c, r"""
## 4. Core or middle-ground? Rolling OAT-vs-Bund beta

A 60-day rolling regression of ΔOAT(10y) on ΔBund(10y). β and correlation near 1 ⇒ the OAT comoves with the safe benchmark (a *core* asset); a decline signals decoupling / risk repricing.
""")
code(c, r"""
bc = pr.rolling_beta_corr(tenor=10, window=60)
fig, ax = plt.subplots(figsize=(10,4.5))
ax.plot(bc.index, bc["beta"], lw=.7, label="rolling beta")
ax.plot(bc.index, bc["corr"], lw=.7, label="rolling corr", color="tab:green")
ax.axhline(1, color="grey", lw=.5, ls=":")
for d, info in events.iterrows():
    if info.tier==1: ax.axvline(d, color="tab:red", ls="--", lw=.7)
ax.set(xlabel="Year", ylabel="beta / correlation", title="OAT-Bund 10y comovement (60-day rolling)")
ax.legend(frameon=False, fontsize=8); ax.set_xlim(pd.Timestamp("2015-01-01"), bc.index.max())
common.save_fig(fig, "study1_rolling_beta.png"); plt.show()
""")

md(c, r"""
## 5. Brexit-style regression and the structural break

`dSpread_t = a + g·Event_t + d·dCISS_t + φ·Spread_{t-1} + e_t` (HAC/Newey-West *t*-stats), and a pre/post comparison around the 9 June 2024 dissolution.
""")
code(c, r"""
out = pr.event_regression(spreads["10y"])
print("Regression (dSpread, 10y):  R2=%.3f  n=%d" % (out["regression"].attrs["r2"], out["regression"].attrs["nobs"]))
display(out["regression"])
print("\nStructural break around 2024-06-09 (dissolution):")
display(out["structural"])
out["regression"].to_csv(common.TABLE_DIR / "study1_regression.csv")
""")

md(c, r"""
### Reading the results

- The **dissolution (9 Jun 2024)** produced the largest, most significant jump in the safety premium (≈ +25 bp on the 10y over the following week, *t* ≈ 12) — the single clearest political shock in the sample.
- The **structural break** is stark: the average 10y OAT–Bund spread **more than doubled** between the pre- and post-dissolution regimes (≈ 29 → ≈ 69 bp). Crucially the OAT's beta to the Bund stays ≈ 1 (it even edges up, ~0.91 → ~1.02): France keeps *comoving* with the core but now carries a persistently higher premium — repriced *in level*, not decoupled. This is the precise sense in which the OAT sits between a pure core asset and a riskier one.
- In the daily regression, **risk aversion (ΔCISS)** is the significant high-frequency driver and the spread mean-reverts; the narrow event dummy is weak because the French repricing was more gradual/anticipated than the one-day Brexit surprise — itself an informative contrast with the UK paper.

*Notes:* the benchmark is now the **German Bund** curve, fitted with the same Svensson engine as the OATs (the ECB AAA proxy, `benchmark="aaa"`, gives a near-identical spread — see the robustness chart).
""")
write(c, "Study1_Political_Risk_OAT_Bund.ipynb")

# =========================================================================== #
#  STUDY 2 - PCA across regimes & the 2022-2024 inversion                     #
# =========================================================================== #
c = nb_init()
md(c, r"""
# Study 2 — PCA of the curve across monetary regimes & the 2022–2024 inversion

**Extends Section 5.3 (Table 4); hardened following Holtz (2023) and Lord & Pelsser (2007).**

The paper runs PCA on the *levels* of 1–10y OAT yields and finds PC1 (level) explains ~97.6% over 1999–2018. But yield **levels** share a strong common trend, so PC1 trivially dominates and the slope's role is hidden. The proper object for factor analysis (Litterman–Scheinkman, 1991) is the **changes**. Our extended sample adds the **2022–2024 inflation shock** and the most aggressive ECB tightening on record, which **inverted** the curve (2y above 10y) for the first time.

**Hypothesis:** the **slope** (PC2) gains importance in 2022–2024. We show it is *invisible on levels but clear on changes*, certify each regime with the **Lord & Pelsser (2007) sign-change test**, and stress-test it for the **short-window unreliability** Holtz (2023) warns about.

> *Method note.* Our Svensson/GSW pipeline is the same family used in recent work; the monetary-policy-shock literature instead PCAs *standardised surprises* (e.g. Karlberg & Åkesson, 2025) — a different statistical object. We run **covariance** PCA on changes (the Litterman–Scheinkman, 1991 convention; see also Luber, 2024 for a PCA decomposition of a related euro curve) and offer standardised/shrinkage variants for robustness.
""")
code(c, SETUP + r"""
from studies import pca_regimes as pca
# Euro sample: before ~1999 there were no short OATs, so the 2y is an
# extrapolation; we restrict the study to where the short end is data-pinned.
zero = common.load_base()["zero"].loc["1999-01-01":]

# Headline: PC2 (slope) share under the LEVELS metric vs the CHANGES metric.
lvc = pca.levels_vs_changes(zero)
display(lvc)
lvc.to_csv(common.TABLE_DIR / "study2_levels_vs_changes.csv")
""")

md(c, r"""
## 1. The slope is invisible on levels, clear on changes

On **levels** the slope share is a couple of percent everywhere — the hypothesis does not show. On **changes** it jumps to ~24% in 2022–2024, far above the 7–10% of every other regime.
""")
code(c, r"""
fig, ax = plt.subplots(figsize=(10,4.5))
x = np.arange(len(lvc)); w=0.38
ax.bar(x-w/2, lvc["PC2 levels (%)"].astype(float), w, label="PC2 on levels", color="tab:grey")
ax.bar(x+w/2, lvc["PC2 changes (%)"].astype(float), w, label="PC2 on changes", color="tab:orange")
ax.set_xticks(x); ax.set_xticklabels([s.split(" (")[0] for s in lvc.index], fontsize=7, rotation=12)
ax.set(ylabel="PC2 (slope) share of variance (%)", title="Slope factor: levels vs changes, by regime")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study2_levels_vs_changes.png"); plt.show()
""")

md(c, r"""
## 2. Lord & Pelsser (2007) sign-change certification

The level/slope/curvature reading is valid only if PC1/PC2/PC3 have **0, 1, 2 sign changes**. The change-PCA passes on each homogeneous regime (incl. 2013–2021 and 2022–2024, our comparison) — **but fails on the full multi-regime sample** (PC1 itself crosses zero twice), which is precisely why a *regime-by-regime* analysis is the right approach rather than one global PCA.
""")
code(c, r"""
cert = pca.pca_by_regime(zero, on="changes")[["PC1 (level)","PC2 (slope)","PC3 (curv.)","years","lord_signs","lord_valid"]]
display(cert)
cert.to_csv(common.TABLE_DIR / "study2_lord_certification.csv")
Xfull = pca._prepare(zero, "changes", None)
_, Vfull = pca._pca(Xfull.to_numpy()); cnt, ok = pca.lord_test(Vfull)
print(f"Full-period (1999-2026) change-PCA sign changes = {cnt}  ->  Lord valid: {ok}")
""")

md(c, r"""
## 3. The inversion (slope = 10y − 2y)

`pct_days_inverted` is the share of trading days with the curve inverted (10y below 2y). The 2022–24 euro-area inversion — its depth and information content — is analysed by Fonseca, McQuade, Van Robays & Vladu (ECB Economic Bulletin, 2023) and by the Banque de France (2024); both document the steepest inversion in decades and discuss why it differs from past episodes.
""")
code(c, r"""
inv = pca.inversion_stats(zero=zero); display(inv)
inv.to_csv(common.TABLE_DIR / "study2_inversion_stats.csv")
slope = (zero["10y"]-zero["2y"])*1e4
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(slope.index, slope, lw=.6)
ax.axhline(0, color="k", lw=.7)
ax.fill_between(slope.index, slope, 0, where=(slope<0), color="tab:red", alpha=.5, label="inverted (10y<2y)")
ax.set(xlabel="Year", ylabel="10y - 2y slope (bp)", title="French curve slope and inversion episodes")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study2_slope_inversion.png"); plt.show()
""")

md(c, r"""
## 4. Factor shapes (PC loadings, on changes) in the 2022–2024 regime

The loadings should look like a flat **level**, a monotone **slope**, and a hump-shaped **curvature** — and satisfy the 0/1/2 sign-change rule.
""")
code(c, r"""
load = pca.pca_loadings(start="2022-01-01", end="2024-12-31", on="changes")
cnt, ok = pca.lord_test(load.to_numpy())
fig, ax = plt.subplots(figsize=(8,4))
mats = [int(i[:-1]) for i in load.index]
for pc,lab in zip(["PC1","PC2","PC3"],["PC1 ~ level","PC2 ~ slope","PC3 ~ curvature"]):
    ax.plot(mats, load[pc], marker="o", ms=4, label=lab)
ax.axhline(0, color="grey", lw=.5)
ax.set(xlabel="Tenor (years)", ylabel="loading",
       title=f"PC loadings on changes, 2022-2024  (sign changes {cnt}, Lord valid={ok})")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study2_loadings.png"); plt.show()
""")

md(c, r"""
## 5. Rolling PC2 share + Lord validity (Holtz's short-window warning)

A 1-year rolling change-PCA. Holtz (2023): short windows lack enough yield variation to estimate loadings reliably, so we **shade the periods where the rolling decomposition fails the Lord test** — read the PC2 line with caution there.
""")
code(c, r"""
roll = pca.rolling_pc_shares(zero=zero, window=252, on="changes")
invalid = roll[~roll["lord_valid"].astype(bool)]
fig, ax = plt.subplots(figsize=(10,4.5))
ax.plot(roll.index, roll["PC1"], lw=.7, label="PC1 (level)")
ax.plot(roll.index, roll["PC2"], lw=.8, label="PC2 (slope)", color="tab:orange")
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"), color="tab:orange", alpha=.10, label="2022-2024")
# mark Lord-invalid windows
for d in invalid.index:
    ax.axvline(d, color="tab:red", alpha=.06, lw=1)
ax.set(xlabel="Year", ylabel="share of variance (1y rolling, changes)",
       title="Rolling slope share; red band = Lord test fails (decomposition unreliable)")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study2_rolling_pc.png"); plt.show()
print("Rolling windows failing Lord: %.0f%% of the sample" % (100*(~roll['lord_valid'].astype(bool)).mean()))
""")

md(c, r"""
## 6. Robustness — sampling frequency (Holtz) and covariance estimator

Holtz: it is the **length of time**, not the sample count, that matters. Re-estimating the slope share with **weekly** (not daily) data over the *same* regimes shows the 2022–2024 rise is robust in *direction* (still the highest), though smaller in *magnitude* — and flags the ~1.5-year 2025 window as too short to trust. The **constant-correlation shrinkage** estimator Holtz recommends leaves the result essentially unchanged (δ≈0.03), i.e. it is not an artefact of the sample covariance.
""")
code(c, r"""
rob = pca.window_length_robustness(zero)
display(rob)
rob.to_csv(common.TABLE_DIR / "study2_window_robustness.csv")
# estimator robustness on the key regime
import numpy as np
X22 = pca._prepare(zero.loc["2022-01-01":"2024-12-31"], "changes", None).to_numpy()
pc2_sample = pca._pca(X22, "sample")[0][1]
pc2_shrink = pca._pca(X22, "cc_shrink")[0][1]
print("2022-2024 PC2 (changes):  sample=%.2f%%   const-corr shrinkage=%.2f%%" % (pc2_sample*100, pc2_shrink*100))
""")

md(c, r"""
### Reading the results

- **Hypothesis confirmed — on changes.** The slope's variance share rises from **14.9% (1999–2018) to 20.4% (1999–2026)**, and reaches **~24% in 2022–2024 vs 7–10% in every other regime**. On *levels* it is ~2% throughout (even lower for the extended sample), so the result is only visible on the correct (changes) object — a concrete methodological point for the thesis.
- **Certified and self-justifying.** The Lord (2007) test passes on 2013–2021 and 2022–2024 (the comparison), so the slope reading is valid there; it *fails* on the full 1999–2026 change-PCA, which formally motivates the regime-by-regime design.
- **Robust but bounded.** The rise is the largest at daily frequency (≈24%) and still the highest of any regime at weekly frequency (≈12%); the 2025 window (1.4y) is too short to trust (Holtz) and is flagged; shrinkage doesn't move the result.
- **Inversion.** 2022–2024 is the only regime with sustained inversion (~30% of days, slope to about −30 bp) — unprecedented in the 1999–2026 OAT sample.

*Caveat on interpretation (ECB).* The 2022–2024 slope is not driven by conventional-policy expectations alone: Fonseca, McQuade, Van Robays & Vladu (ECB Economic Bulletin, 2023) note that **APP/PEPP asset purchases compressed the slope** (so the risk-free slope would have stayed positive longer absent QE) and that the **negative nominal slope is driven mainly by the real curve**, not inflation compensation. So "the slope factor grew" embeds balance-sheet (QT) and real-rate/inflation components — it is a statement about the term structure's shape, not a clean read of policy expectations.
""")
md(c, r"""
## References

- Litterman, R. & Scheinkman, J. (1991). *Common Factors Affecting Bond Returns.* Journal of Fixed Income, 1(1), 54–61.
- Lord, R. & Pelsser, A. (2007). *Level-Slope-Curvature — Fact or Artefact?* Applied Mathematical Finance, 14(2), 105–130.
- Holtes, G. (2023). *Decoding Yield Curves: The Covariance Matrix and its Ripple Effect on PCA-based Decomposition.* (grantholtes.medium.com, 26 Nov 2023.)
- Fonseca, L., McQuade, P., Van Robays, I. & Vladu, A. L. (2023). *The inversion of the yield curve and its information content in the euro area and the United States.* ECB Economic Bulletin.
- Banque de France (2024). *What does an inversion of the yield curve tell us?* Bulletin de la Banque de France, 250/3, Jan–Feb 2024.
- Karlberg, T. & Åkesson, N. (2025). *Transmission of Monetary Policy Shocks in a Small Open Economy: A Sector-Level PCA-VAR Approach.* MSc thesis, Lund University.
- Luber, S. (2024). *PCA on Inflation Swaps.* (Medium, 7 Aug 2024.)
- Grishchenko, O. V., Moraux, F. & Pakulyak, O. (2020). *Fuel up with OATmeals!* The Journal of Finance and Data Science, 6, 49–85.
""")
write(c, "Study2_PCA_Curve_Regimes.ipynb")

# =========================================================================== #
#  STUDY 3 - March 2020 dislocation vs past crises                            #
# =========================================================================== #
c = nb_init()
md(c, r"""
# Study 3 — Market dislocation: March 2020 vs past crises

**Extends Section 8.2; hardened following Hu–Pan–Wang (2012), the Banque de France free-float Bulletin and the FSB (2022) report.**

The paper reads the **MAE** and the **HPW noise measure** as proxies for illiquidity / the (un)availability of arbitrage capital, spiking at Lehman (2008) and the 2011–12 sovereign crisis. We add **COVID-19**. But a raw overall comparison (≈11 bp vs ≈6 bp) is **not valid**: the cross-section differs across crises (number and maturity mix of bonds), and our OAT cross-section is small (~30–50 bonds vs HPW's >100), so its *composition* sways the overall RMSE. HPW themselves exclude <1y and note the short end is structurally noisier — so we compare **within maturity bins** (the paper's eq. 15), which is principled, not ad hoc.

Reframed questions:
1. Was March 2020 the most severe dislocation, and **in which part of the curve**?
2. Did **PEPP** (18 Mar 2020) restore pricing efficiency, and how fast — distinct from the *structural* rise of the noise floor in 2023–25?

*Related work.* The March-2020 "dash for cash" and the central-bank response are documented by the Brookings Institution (2020) for the US and, for euro-area sovereigns, by Corradin, Grimm & Schwaab (2021), who show PEPP compressed euro-area sovereign risk premia; the PEPP itself is described in the ECB's programme documentation (ECB, 2020).
""")
code(c, SETUP + r"""
from studies import crisis_dislocation as cd

# Cross-section-normalised comparison: peak fitting error WITHIN each maturity bin.
perbin = cd.per_bin_severity(bins=("2-5yr","5-10yr","10-20yr"))
display(perbin)
perbin.to_csv(common.TABLE_DIR / "study3_per_bin_severity.csv")
""")

md(c, r"""
## 1. Severity by maturity bin — the valid comparison

Comparing **within** a maturity bin controls for the differing cross-sections. The picture sharpens the thesis: COVID is the clear outlier **only in the 5–10y belly**, is middle-of-pack at 2–5y, and is the *mildest* of all crises at 10–20y.
""")
code(c, r"""
fig, ax = plt.subplots(figsize=(9,4.5))
bins = list(perbin.columns); x = np.arange(len(perbin)); w = 0.26
for i,b in enumerate(bins):
    ax.bar(x+(i-1)*w, perbin[b].astype(float), w, label=b.split(" (")[0])
ax.set_xticks(x); ax.set_xticklabels([s.split(" (")[0] for s in perbin.index], fontsize=7, rotation=10)
ax.set(ylabel="peak MAE in bin (bp)", title="Crisis severity by maturity bin (cross-section-normalised)")
ax.legend(title="maturity bin", frameon=False, fontsize=8)
common.save_fig(fig, "study3_per_bin_severity.png"); plt.show()
""")

md(c, r"""
## 2. Overall noise: peak and recovery (1-year window)

Shown for completeness, but read it through the per-bin lens above (it is not cross-section-comparable). With a ~1-year window the recovery is captured: **NaN now means "not recovered within 250 trading days"**, not "never".
""")
code(c, r"""
summ = cd.crisis_summary("noise"); display(summ)
summ.to_csv(common.TABLE_DIR / "study3_crisis_noise.csv")
cd.crisis_summary("mae").to_csv(common.TABLE_DIR / "study3_crisis_mae.csv")
""")

md(c, "## 3. Recovery speed — noise in event time (days from each crisis's onset)")
code(c, r"""
ov = cd.event_time_overlay("noise", pre=20, post=250)
fig, ax = plt.subplots(figsize=(10,4.5))
for col in ov.columns:
    ax.plot(ov.index, ov[col], lw=1, label=col)
ax.axvline(0, color="k", lw=.6, ls=":")
ax.set(xlabel="Trading days from each crisis's onset", ylabel="noise (bp)",
       title="Dislocation and recovery, aligned at crisis onset (1 year)")
ax.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study3_event_time_overlay.png"); plt.show()
""")

md(c, r"""
## 4. The COVID episode and PEPP — corrected timeline

The noise spikes in mid-March, **PEPP** is announced 18 Mar, the noise **peaks in August 2020** and halves ~**17 trading days** later, normalising through **2021**. This is distinct from the **structural rise of the noise floor in 2023–25** (a larger, more heterogeneous post-pandemic OAT universe plus 2022/2024 stress), shown on the right.
""")
code(c, r"""
det = cd.march2020_detail()
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12,4.2))
axL.plot(det.index, det["noise"], color="tab:purple", lw=1, label="noise")
axL.plot(det.index, det["mae"], color="tab:blue", lw=1, label="MAE")
axL.axvline(pd.Timestamp("2020-03-18"), color="tab:green", ls="--", lw=1.2, label="ECB PEPP (18 Mar)")
axL.axvline(pd.Timestamp("2020-08-05"), color="tab:red", ls=":", lw=1.2, label="noise peak (Aug)")
axL.set(ylabel="bp", title="COVID episode and recovery (2020-2021)"); axL.legend(frameon=False, fontsize=8)
nz = common.load_base()["params"]["noise"].loc["2018":]
axR.plot(nz.index, nz, color="tab:purple", lw=.6)
axR.axvspan(pd.Timestamp("2023-01-01"), nz.index.max(), color="tab:red", alpha=.10, label="2023-25 structural floor")
axR.axhline(nz.loc["2015":"2019"].median(), color="tab:green", ls="--", lw=.8, label="pre-COVID median")
axR.set(title="Noise floor: 2021 normalisation vs 2023-25 rise"); axR.legend(frameon=False, fontsize=8)
common.save_fig(fig, "study3_march2020_zoom.png"); plt.show()
print("noise: 2021 mean %.2f bp | 2023-25 mean %.2f bp" % (nz.loc['2021'].mean(), nz.loc['2023':'2025'].mean()))
""")

md(c, r"""
## 5. Is the COVID belly dislocation driven by a few squeezed bonds? (HPW / free-float)

HPW exclude bonds >4σ from the curve so one or two securities cannot drive the measure; our daily fit already applies a stronger robust screen (drops |error|>75 bp). As a direct check, we recompute the 5–10y MAE on the peak days with and without the single worst-fit bond.
""")
code(c, r"""
base = common.load_base(with_bonds=True); bonds, panel = base["bonds"], base["panel"]
peak_days = {"COVID acute (Mar-20)":"2020-03-23", "COVID peak (Aug-20)":"2020-08-05",
             "GFC (Nov-08)":"2008-11-13", "Inflation (Oct-22)":"2022-10-25"}
loo = pd.DataFrame({k: cd.leave_one_out_bin_mae(d, 5, 10, bonds, panel) for k,d in peak_days.items()}).T
display(loo[["n_in_bin","MAE_full_bp","MAE_drop_worst_bp","worst_bond_contribution_bp"]])
loo.to_csv(common.TABLE_DIR / "study3_leave_one_out.csv")
""")

md(c, r"""
### Reading the results

- **Reframed verdict (thesis 1).** Comparing *within* maturity bins, March 2020 is the most severe dislocation **only in the 5–10y belly** (~11 bp peak, ~2× any other crisis there); at 2–5y it is comparable to 2008 and **below** the 2022 shock, and at 10–20y it is the *mildest* of all crises. COVID was a **belly-specific** dislocation, not a uniform one.
- **PEPP timeline (thesis 2), corrected.** The noise peaks in **August 2020** and halves within **~17 trading days**, normalising through **2021**. PEPP (18 Mar) helped stabilise the market, but cross-sectional pricing efficiency took months, not days, to recover. The **elevated noise of 2023–25 is a separate structural shift** (larger, more heterogeneous post-pandemic OAT universe plus 2022/2024 stress) — not COVID persistence.
- **The free-float / collateral confound (BdF, FSB).** The leave-one-out check shows COVID's 5–10y dislocation is genuine (still ~7 bp dropping the worst bond — the highest of any crisis) **but carries a sizeable single-issue component** (~3 bp on the peak day, ~50% in the acute March phase), whereas 2008/2022 are broad-based (<1 bp). This matches the **Banque de France** finding that the falling *free float* (51.1%→34.8% since 2015, via Eurosystem holdings) made low-free-float securities dislocate more in March 2020, and the **FSB (2022)** euro-area "dash for collateral". Treat these as one mechanism — *relative scarcity of specific paper* — a third confound alongside cross-section size and genuine arbitrage-capital scarcity. *Counter-nuance (FSB):* French primary markets stayed resilient (large OAT/BTF issuance in March 2020), so the long-run free-float decline is not the whole story for the acute shock.
- **Why the noise isn't just bad curve-fitting (HPW).** HPW show term-structure variables explain only ~1.5–5.6% of the noise, and the measure is an RMSE *averaged* across bonds (÷N), not a cumulative sum — so the cross-section-size concern is about *composition* (addressed by the per-bin comparison), not a mechanical artefact.
""")
md(c, r"""
## References

- Hu, G. X., Pan, J. & Wang, J. (2013). *Noise as Information for Illiquidity.* The Journal of Finance, 68(6), 2341–2382. (Working-paper version, July 2012.)
- Banque de France (2023). *French sovereign debt liquidity: main factors, recent developments and resilience during the Covid crisis.* Bulletin de la Banque de France, 246/1, May–June 2023.
- Financial Stability Board (2022). *Liquidity in Core Government Bond Markets.* FSB, 20 October 2022.
- Corradin, S., Grimm, N. & Schwaab, B. (2021). *Euro area sovereign bond risk premia during the Covid-19 pandemic.* ECB Working Paper No. 2561.
- European Central Bank (2020). *Pandemic Emergency Purchase Programme (PEPP).* ECB programme documentation.
- Brookings Institution (2020). *What did the Fed do in response to the COVID-19 crisis?* (Cheng, Powell, Skidmore et al.)
- Panigrahi, A. K., Sharma, A. & Sarda, V. (2026). *Liquidity Recovery Dynamics Following Volatility Shocks.* Journal of Risk and Financial Management, 19. (Emerging-equity evidence; cited for the recovery-dynamics framing.)
- Grishchenko, O. V., Moraux, F. & Pakulyak, O. (2020). *Fuel up with OATmeals!* The Journal of Finance and Data Science, 6, 49–85.
""")
write(c, "Study3_Crisis_Dislocation_2020.ipynb")
print("\nAll three study notebooks generated.")
