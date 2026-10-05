# Supplementary studies — the post-2018 years

Self-contained extensions built **on top of** the (untouched) OATmeals
replication, focused on the years the original paper never saw (2018–2026).
Each study reuses the cached daily Svensson fit produced by the base project and
adds only what it needs; none of them modifies `oatcurve/` or its outputs.

| Study | Question | Paper section extended | Notebook |
|---|---|---|---|
| **1. Political risk** | How did the 2024–26 French political instability reprice the OAT–Bund safety premium? | §7.1 + UK/Brexit event study | `notebooks/Study1_Political_Risk_OAT_Bund.ipynb` |
| **1.2 Political premium decomposition** | How much of the spread is idiosyncratic French vs common-euro, across the curve, and does it reach corporate credit? | §7.1 + INSEE two-stage Focus | `notebooks/Study1.2_Political_Premium_Decomposition.ipynb` |
| **2. PCA regimes** | Did the slope factor gain importance as the 2022–24 ECB tightening inverted the curve? | §5.3 (Table 4) | `notebooks/Study2_PCA_Curve_Regimes.ipynb` |
| **3. Crisis dislocation** | Was March 2020 as severe as 2008, and did PEPP restore pricing efficiency? | §8.2 (MAE / noise) | `notebooks/Study3_Crisis_Dislocation_2020.ipynb` |
| **3.1 Regime detection** | Can the COVID / 2022 / 2024 regime shifts be formally dated and certified? | §8.2 + structural-break econometrics | `notebooks/Study3.1_Regime_Detection.ipynb` |

## Code

```
studies/
├── bund_curve.py      # German Bund curve, fitted with the same Svensson engine as the OATs
├── external_data.py   # ECB curves (AAA + all-bonds), CISS, balance sheet, Euribor, MIR; auto-download/cached
├── common.py          # load base results, plotting, French political-event calendar
├── political_risk.py  # Study 1: spread, shock detection, event study, rolling beta, regression
├── political_premium.py   # Study 1.2: common-euro vs idiosyncratic decomposition, term structure, INSEE Stage-1/2
├── pca_regimes.py     # Study 2: PCA on levels & changes, Lord (2007) sign-change test, window-length & shrinkage robustness
├── crisis_dislocation.py  # Study 3: per-bin severity (cross-section-normalised), recovery, leave-one-out robustness
└── regime_detection.py    # Study 3.1: Markov-switching (Hamilton) + Bai-Perron breaks + CUSUM, CISS/event corroboration
```

Study 3.1 needs `statsmodels` and `ruptures` (`pip install statsmodels ruptures`).

## How to run

```bash
# 1) the base OAT fit must exist first (one-off):
python scripts/01_build_panel.py
python scripts/02_fit_curves.py
# 2) the German Bund curve for Study 1 (one-off, ~4 min):
python scripts/05_fit_bund_curve.py
# 2b) Study 1.2 also needs the Bund curve:  python scripts/05_fit_bund_curve.py
# 3) (re)build the study notebooks from source:
python notebooks/build_studies_notebooks.py      # Studies 1, 2, 3
python notebooks/build_study12_notebook.py       # Study 1.2 (standalone)
python notebooks/build_study31_notebook.py       # Study 3.1 (standalone)
# 4) execute them in-place so they open already populated with figures/tables:
python scripts/06_execute_notebooks.py     # needs: pip install nbconvert ipykernel
```

> The builders write notebooks **without** outputs (clean source for version
> control). If you open one straight after building it shows code + text but no
> rendered figures — run step 4 (or *Run All* in Jupyter) to populate it. The
> rendered figures are also always saved as PNGs in `output/studies_figures/`.

The studies download ECB series (CISS, euro-area curves, balance sheet,
Euribor, MFI lending rates) from the ECB Data Portal on first use and cache
them under `output/studies_cache/`; afterwards everything runs offline. Behind
a TLS-intercepting proxy or antivirus, set `ECB_SSL_VERIFY=0`.

## Data choices (documented)

* **Safe benchmark = German Bund curve**, fitted with the *same* Svensson/GSW
  engine as the OATs from the Bloomberg Bund export (`Analisi Bunds *.csv`,
  1999–2026), covering Schätze (~2y), Bobls (~5y) and Bunds (10–30y). The
  OAT–Bund spread is therefore computed from two internally consistent curves —
  the most faithful reading of Section 7.1. The **ECB AAA euro-area curve**
  (`YC … G_N_A`, same method, from 2004) remains available as a cross-check via
  `benchmark="aaa"`: it matches the Bund spread to a few bp in calm periods and
  recently, but is materially narrower during the 2008–2012 crisis (the AAA
  basket understates the premium to the ultimate safe asset), so the exact Bund
  is preferred. A user CSV can still be passed via `benchmark_csv=`.
* **Risk-aversion control = ECB CISS** (Composite Indicator of Systemic Stress),
  the euro-area analogue of the UK paper's "regional risk aversion".
* **Event calendar**: the 2024–25 sequence (dissolution → elections → Barnier →
  no-confidence → Bayrou → Lecornu) is hard-coded in
  `common.FRENCH_POLITICAL_EVENTS`. The data-driven shock detector
  (`political_risk.detect_shocks`) flags the market-moving days independently,
  so the analysis does not hinge on the calendar being exhaustive.

## Headline findings (with the current data, to 5 Jun 2026)

* **Study 1** — the 9 Jun 2024 dissolution triggered the largest, most
  significant jump in the safety premium (+24.6 bp on the 10y OAT–Bund over the
  following week, *t* ≈ 12 with the exact Bund curve); the average 10y spread
  **more than doubled** between the pre- and post-dissolution regimes
  (28.6 → 69.4 bp). The OAT's beta to the Bund stays ≈ 1 (~0.91 → ~1.02): France
  keeps comoving with the core but carries a persistently higher premium —
  repriced in level, not decoupled.
* **Study 1.2** — decomposing the spread, common-euro factors (periphery
  dispersion + CISS) explain only ~15–20% of the daily variation; the
  idiosyncratic French premium rises from ~0 (2019) to ~40 bp now, and the
  post-2024 widening is almost entirely French (the common-euro component is
  flat) — it is *not* European risk-off. The dissolution moved the 5–30y spread
  ~2× more than the 2y (a structural-fiscal, not rollover, premium). Reduced
  INSEE Stage-2: the spread passes through to French NFC loan rates at ~0.67,
  implying a **+0.2 pp** political-premium cost of corporate credit — matching
  INSEE. Validation: our 3y widening (+18.6 bp, May-24→Jan-25) ≈ INSEE's +18 bp.
* **Study 2** — the slope's importance is invisible on yield *levels* (PC2 ≈ 2 %
  everywhere) but clear on *changes*: PC2 rises 14.9 % (1999–2018) → 20.4 %
  (1999–2026) and reaches **~24 % in 2022–2024 vs 7–10 % in every other regime**.
  The Lord & Pelsser (2007) test certifies the level/slope/curvature reading on
  each sub-regime but **fails on the full multi-regime change-PCA**, which
  formally justifies the regime-by-regime design. The rise is robust in
  direction at weekly frequency and to constant-correlation shrinkage (Holtz,
  2023), though the ~1.5-year 2025 window is flagged as too short. 2022–2024 is
  also the only regime with sustained inversion (~30 % of days). *Caveat (ECB):*
  the slope embeds APP/PEPP balance-sheet compression and a real-rate component,
  not just conventional-policy expectations.
* **Study 3** — crisis episodes (2008, 2011, 2020, 2022) are compared *within
  maturity bins* (the paper's eq. 15), because a raw overall comparison mixes
  cross-sections of different size and composition. Recovery half-lives and a
  leave-one-out check separate broad-based stress from single-issue effects.
  The pricing-noise floor of 2023–25 (≈ 9.7 bp, against ≈ 4.4 bp in 2021) is a
  separate structural shift.
* **Study 3.1** — the regime shifts are dated formally with Markov-switching
  and Bai–Perron: a **structural break in pricing noise on 1 Jul 2022**, a
  flat/inverted-slope regime in 2022–24 (probability 0.83, against 0.39 before
  2022) and a high political-premium regime from 2024 (probability 0.90 in
  2024–26, against 0.00 in 2019). CUSUM rejects parameter stability on all three
  series. In the spirit of Yi et al. (2026) the regimes are corroborated with
  independent signals: the premium regime matches the event calendar (1.00 on
  event weeks vs 0.15 otherwise), while the noise regime does *not* follow euro
  systemic stress (CISS; correlation ≈ −0.06), which points to a structural
  rather than a stress-driven floor.
