# Replication notes: the OATmeals paper, extended to 2026

A full, reproducible implementation of

> Grishchenko, O. V., Moraux, F., & Pakulyak, O. (2020).
> **"Fuel up with OATmeals! The case of the French nominal yield curve."**
> *The Journal of Finance and Data Science*, 6, 49–85.

The French nominal zero-coupon yield curve is estimated **every trading day**
from the cross-section of fixed-coupon OAT/BTAN bid prices, using the
**Svensson (1994)** parametric forward-rate curve fitted in the style of
**Gürkaynak, Sack & Wright (GSW, 2007)**.

The original paper runs from **22 Oct 1987 to 10 Apr 2018**. This project
reproduces that work *and extends it to the latest available quote
(5 Jun 2026)* — this is **"Part 1"**: documenting how the fit quality, the
shape and dynamics of the curve, and the market-functioning diagnostics change
once the methodology is carried all the way to today.

---

## 1. Methodology → code map

| Paper | Equation(s) | Module / function |
|---|---|---|
| Zero-coupon price ↔ yield | (1)–(2) | `svensson.discount_factor`, `bonds.continuously_compounded` |
| Coupon-bond pricing (no-arbitrage) | (3) | `estimation.model_dirty_prices` |
| Yield-to-maturity | (4) | `bonds.ytm_from_dirty` |
| Par yield | (5) | `svensson.par_yield` |
| Forward rate / integration | (6)–(8) | `svensson.forward_rate`, `svensson.zero_yield` |
| Macaulay / modified duration | (9)–(10) | `bonds.macaulay_duration`, `bonds.modified_duration` |
| Svensson forward curve | (11) | `svensson.forward_rate` |
| Svensson zero curve | (12) | `svensson.zero_yield` |
| Filters | §4.3 | `data_loading.build_bonds` (static) + `estimation.prepare_day` (dynamic) |
| Weighted-least-squares fit | (13)–(14) | `estimation.fit_day` |
| MAE (overall / by bin) | (15)–(16) | `metrics.day_metrics` |
| On-the-run premium | (17) | `otr.on_the_run_premium` |
| HPW noise measure | (18) | `metrics.day_metrics["noise"]` |
| Level / slope / curvature, PCA | §5.2 | `analysis.term_structure_factors`, `analysis.pca_decomposition` |

### The Svensson curve (eqs. 11–12)

Forward rate at horizon `m` with parameters `θ = (β0, β1, β2, β3, τ1, τ2)`:

```
f(m) = β0 + β1·e^(−m/τ1) + β2·(m/τ1)·e^(−m/τ1) + β3·(m/τ2)·e^(−m/τ2)
```

Continuously-compounded zero yield (integral of `f` over `[0, m]`):

```
y(m) = β0 + β1·G(m/τ1) + β2·[G(m/τ1) − e^(−m/τ1)] + β3·[G(m/τ2) − e^(−m/τ2)]
       with  G(x) = (1 − e^(−x)) / x.
```

### The daily estimation (eq. 14)

For each day `t`, the six parameters minimise the **duration-weighted sum of
squared price errors**:

```
θ̂_t = argmin_θ  Σ_k [ (P_obs(k) − P_model(k; θ)) / D_k ]²
```

where `P_obs` is the **dirty** bid price (clean bid + ACT/ACT accrued interest),
`P_model` discounts the bond's cash flows on the Svensson zero curve, and `D_k`
is the bond's **modified duration**. Constraints follow the paper:
`τ1, τ2, β0 > 0`, while `β0 + β1` is left free so the **short rate can be
negative** (relevant for 2014–2022).

---

## 2. Data (Section 3)

Two Bloomberg exports, placed in `data/` (layout in [`data/README.md`](../data/README.md)):

* **`Analisi OATs 1.csv`** — the security master (ISIN, coupon, issue/maturity
  dates, currency, optionality).
* **`Analisi OATs (Bid-Ask Data).csv`** — a wide panel: two rows per bond
  (Ask, Bid) × 10,077 daily date columns from 22 Oct 1987 to 5 Jun 2026. In
  line with GSW we use the **bid** rows.

After filtering, **155 securities** are kept (152 OATs + 3 legacy BTANs), with
**6 to 156 bonds quoted per day** — exactly matching the paper's statement that
from 22 Oct 1987 onwards there are always ≥ 6 quotes (the minimum needed to
identify the 6 Svensson parameters).

### Filters implemented (Section 4.3)

1. **Regular bonds only** — drop special-feature securities (the 1 `CALL/SINK`
   and 1 `PUTABLE` in the data); floaters, linkers and STRIPS are not present.
   Currencies at issuance are FRF, XEU (ECU) and EUR — all admissible.
2. **No BTFs** — short-term bills are not in this dataset.
3. **≥ 12 months to maturity** — applied dynamically each day.
4. **On-the-run kept** — unlike GSW, the most-recent and first-off-the-run
   bonds are *not* excluded from the baseline fit (the paper finds no premium).
5. **Abnormal quotes** — the two hand-checked ISINs flagged in the paper
   (`FR0000041410`, `FR0000570509`) are blanked on the exact dates listed.

---

## 3. How to run

```bash
pip install -r requirements.txt

python scripts/01_build_panel.py     # parse + filter + cache   (seconds)
python scripts/02_fit_curves.py      # fit every day 1987-2026  (~10-15 min, one-off)
python scripts/03_make_outputs.py    # figures + tables         (seconds)
python scripts/04_on_the_run.py      # on-the-run premium       (optional, ~1-2 min)
```

Outputs land in `output/figures/` and `output/tables/`; intermediate parquet
caches in `output/cache/`. The Jupyter notebook
`notebooks/French_OAT_Yield_Curve.ipynb` loads the caches and walks through the
whole story with inline figures.

All modelling choices live in **`oatcurve/config.py`** (sample windows, filter
toggles, maturity bins, optimiser bounds), so the entire study can be
re-run or stress-tested by editing one file.

---

## 4. Validation against the paper

| Quantity | Paper | This replication |
|---|---|---|
| Overall MAE, euro sample (mean) | ≤ ~3 bp | ~2 bp (1999–2018) |
| Overall MAE, euro sample (max) | 13 bp | same order |
| Worst-fit bin, euro sample | 10–20 yr | 10–20 yr |
| Pre-euro fit | poor / volatile | poor / volatile (as expected) |
| PCA: PC1 share (euro) | 0.9755 | ~0.97–0.98 |
| On-the-run premium | negligible | negligible |
| 2-Apr-2018 curve | 10y ≈ 0.7% | 10y ≈ 0.70% |

Economic sanity checks across regimes are all reproduced: negative short rates
under ECB NIRP (2016, 2020), the inverted curve of 2023, and the
post-tightening normalisation of 2025–2026.

---

## 5. Notes, conventions and judgment calls

* **Settlement** is taken as the quote date (no T+2 adjustment); horizons use
  Actual/365.25, accrued interest uses Actual/Actual (ICMA) — the OAT
  convention. These are standard in the GSW-style literature and the effect of
  the T+2 simplification on fitted yields is negligible.
* **OAT/BTAN labelling** is economic (BTAN = issued before 2013 with original
  term ≤ 7 yr), since Bloomberg's `Serie` field is not a reliable tag here;
  both are priced identically as fixed-coupon bullets, so this is descriptive
  only.
* **Retail "France-Physiques" OATs** — the 17 retail-tranche securities carry
  *no price data* in this Bloomberg extract, so they never enter the fit; the
  `include_retail_oat` toggle is provided for completeness.
* **Short-end pre-euro** — before ~1999 there were no short-maturity OATs, so
  the curve's extrapolated short end (< ~2 yr) is unreliable on those days
  (a known Svensson limitation). The long end (≥ 5 yr) is robust throughout,
  and the paper itself separates the pre-euro period for this reason.
* **Warm starts** — each day is seeded from the previous day's solution (as in
  GSW), with a multi-start fallback when the fit is poor; this yields smooth
  parameter paths and keeps the full-sample run to a few minutes.
