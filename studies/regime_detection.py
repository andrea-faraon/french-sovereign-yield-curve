"""
Study 3.1 - Formally detecting and dating the crisis/regime shifts.

Study 3 showed *descriptively* that the OAT pricing-noise floor stepped up after
2022 and that COVID was a distinct dislocation.  Here we put dates and
confidence on those shifts with **peer-reviewed structural-change methods on our
own structured series** (no unstructured/text data):

  * Markov-switching (Hamilton, 1989) - `statsmodels`: estimates latent
    calm/stressed regimes and returns the smoothed probability of the stressed
    regime through time, with transition probabilities and expected durations.
  * Bai-Perron multiple structural breaks - `ruptures` (exact dynamic
    programming, mean shifts, number of breaks chosen by BIC): dates the level
    shifts of the series.
  * CUSUM-of-OLS-residuals stability test (Ploberger-Kramer) - `statsmodels`:
    a formal test that the series is *not* structurally stable.

The same toolkit is applied to the **noise** (Study 3 dislocation), the
**slope** 10y-2y (Study 2 inversion) and the **idiosyncratic OAT-Bund premium**
(Study 1.2 political regime), so the three post-2018 episodes are dated with one
consistent, certified procedure.  In the spirit of Yi, Mehra, Chen & Cartlidge
(2026) - who *augment* regime detection with an extra information source - we
corroborate the price-based regimes against an independent market-stress signal
(the ECB CISS) and the political-event calendar, instead of reproducing their
(text-based, US-Treasury, non-peer-reviewed) pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from . import common, external_data, political_premium

DEFAULT_START = "2005-01-01"   # focus on the modern regimes (GFC onward); the
                               # 1999-2003 early-euro noise is a separate transition
FREQ = "W-FRI"                 # weekly: regimes are macro-scale; removes daily microstructure


# --------------------------------------------------------------------------- #
#  Target series                                                              #
# --------------------------------------------------------------------------- #
def target_series(name: str, start: str = DEFAULT_START, freq: str = FREQ) -> pd.Series:
    """Weekly series to analyse: 'noise', 'slope' (10y-2y, bp) or 'premium'
    (idiosyncratic OAT-Bund, bp)."""
    base = common.load_base()
    if name == "noise":
        s = base["params"]["noise"]
    elif name == "slope":
        s = (base["zero"]["10y"] - base["zero"]["2y"]) * 1e4
    elif name == "premium":
        s = political_premium.idiosyncratic_premium(10)["idiosyncratic_FR"]
    else:
        raise ValueError(name)
    return s.loc[start:].resample(freq).mean().dropna().rename(name)


# --------------------------------------------------------------------------- #
#  Markov-switching (Hamilton)                                                #
# --------------------------------------------------------------------------- #
def _regime_means(series: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Probability-weighted data mean of each regime (robust regime labelling)."""
    return np.array([(series * smoothed[:, i]).sum() / smoothed[:, i].sum()
                     for i in range(smoothed.shape[1])])


def markov_switching(name: str, start: str = DEFAULT_START, ks=(2, 3),
                     stress_is_high: bool = None) -> dict:
    """
    Fit a Markov-switching model (switching mean and variance) for several regime
    counts, pick the one preferred by BIC, and return the smoothed regime
    probabilities plus a regime summary.  ``stress_is_high`` marks whether the
    stressed regime is the high-mean one (noise, premium) or the low-mean one
    (slope, i.e. inverted); inferred from ``name`` if not given.

    We cap at 3 regimes (calm / elevated / crisis): 4+ marginally improves BIC
    but the EM step becomes unstable, so 3 is the robust, interpretable choice.
    Multiple random restarts (``search_reps``) guard against local optima.
    """
    if stress_is_high is None:
        stress_is_high = name != "slope"
    s = target_series(name, start)
    x = s.to_numpy()

    fits = {}
    for k in ks:
        res = MarkovRegression(x, k_regimes=k, trend="c",
                               switching_variance=True).fit(search_reps=20)
        fits[k] = res
    bic = {k: res.bic for k, res in fits.items()}
    best_k = min(bic, key=bic.get)
    res = fits[best_k]

    sm = res.smoothed_marginal_probabilities
    means = _regime_means(x, sm)
    order = np.argsort(means)                       # calm -> stressed by mean
    stress_reg = int(order[-1] if stress_is_high else order[0])

    P = np.asarray(res.regime_transition)
    P = P[..., 0] if P.ndim == 3 else P
    dur = 1.0 / (1.0 - np.clip(np.diag(P), None, 0.9999))

    probs = pd.DataFrame(sm, index=s.index, columns=[f"regime{i}" for i in range(best_k)])
    table = pd.DataFrame({
        "mean_bp": np.round(means, 2),
        "vol_bp": np.round([np.sqrt((x - means[i]) ** 2 @ sm[:, i] / sm[:, i].sum())
                            for i in range(best_k)], 2),
        "avg_duration_wks": np.round(dur, 1),
        "time_share": np.round(sm.mean(axis=0), 3),
    }).sort_values("mean_bp")
    return {"series": s, "best_k": best_k, "bic": bic, "res": res,
            "probs": probs, "table": table,
            "stress_prob": probs[f"regime{stress_reg}"].rename("stress_prob")}


# --------------------------------------------------------------------------- #
#  Bai-Perron multiple structural breaks (BIC-selected)                       #
# --------------------------------------------------------------------------- #
def bai_perron(name: str, start: str = DEFAULT_START, max_k: int = 6,
               min_size: int = 12) -> dict:
    """
    Exact multiple-mean-shift detection (Bai-Perron / dynamic programming) with
    the number of breaks selected by BIC.  Returns the break dates and the mean
    of each segment.
    """
    s = target_series(name, start)
    x = s.to_numpy()
    n = len(x)
    algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(x)

    def rss(bkps):
        prev, r = 0, 0.0
        for b in bkps:
            seg = x[prev:b]
            r += ((seg - seg.mean()) ** 2).sum()
            prev = b
        return r

    scan = {}
    for k in range(0, max_k + 1):
        bkps = algo.predict(n_bkps=k) if k > 0 else [n]
        scan[k] = (bkps, n * np.log(rss(bkps) / n) + (k + 1) * np.log(n))
    best_k = min(scan, key=lambda k: scan[k][1])
    bkps = scan[best_k][0]

    dates = [s.index[b - 1] for b in bkps[:-1]]
    seg_bounds = [0] + bkps
    segments = pd.DataFrame({
        "start": [s.index[a] for a in seg_bounds[:-1]],
        "end": [s.index[b - 1] for b in seg_bounds[1:]],
        "mean_bp": [round(x[a:b].mean(), 1) for a, b in zip(seg_bounds[:-1], seg_bounds[1:])],
    })
    return {"series": s, "break_dates": dates, "best_k": best_k,
            "bic_by_k": {k: round(v[1], 0) for k, v in scan.items()},
            "segments": segments}


# --------------------------------------------------------------------------- #
#  CUSUM stability test                                                       #
# --------------------------------------------------------------------------- #
def cusum_stability(name: str, start: str = DEFAULT_START) -> dict:
    """CUSUM-of-OLS-residuals structural-stability test (series on a constant).
    A small p-value rejects stability, i.e. confirms a structural break exists."""
    s = target_series(name, start)
    resid = (s - s.mean()).to_numpy()
    stat, pval, _ = breaks_cusumolsresid(resid)
    return {"stat": round(float(stat), 3), "pvalue": round(float(pval), 4),
            "stable": bool(pval > 0.05)}


# --------------------------------------------------------------------------- #
#  Corroboration in the spirit of Yi et al. (2026): an independent signal     #
# --------------------------------------------------------------------------- #
def ciss_stress_prob(start: str = DEFAULT_START) -> pd.Series:
    """Smoothed probability of the high-stress regime of a 2-state Markov model
    on the ECB CISS - an independent, market-derived stress signal."""
    ciss = external_data.fetch_ciss().loc[start:].resample(FREQ).mean().dropna()
    res = MarkovRegression(ciss.to_numpy(), k_regimes=2, trend="c",
                           switching_variance=True).fit()
    sm = res.smoothed_marginal_probabilities
    hi = int(np.argmax(_regime_means(ciss.to_numpy(), sm)))
    return pd.Series(sm[:, hi], index=ciss.index, name="ciss_stress_prob")


def regime_agreement(prob_a: pd.Series, prob_b: pd.Series) -> dict:
    """How well two smoothed stress-probability series agree (correlation and the
    share of weeks both call the same state at a 0.5 threshold)."""
    df = common.align(prob_a.rename("a"), prob_b.rename("b"))
    same = ((df["a"] > 0.5) == (df["b"] > 0.5)).mean()
    return {"correlation": round(df["a"].corr(df["b"]), 3),
            "agreement_share": round(float(same), 3), "n": len(df)}


def event_stress_alignment(stress_prob: pd.Series, events: dict = None,
                           tier: int = 1) -> dict:
    """
    Corroborate a stressed-regime probability against the political-event
    calendar (an independent, hand-coded signal, in the spirit of Yi et al.):
    the mean stressed-regime probability on the tier-<=``tier`` event weeks vs
    on all other weeks.  A large gap means the dated regime coincides with the
    political events.
    """
    ev = common.events_frame(events)
    ev = ev[ev["tier"] <= tier]
    idx = stress_prob.index
    event_weeks = {common.nearest_trading_day(idx, d) for d in ev.index}
    on = stress_prob[stress_prob.index.isin(event_weeks)]
    off = stress_prob[~stress_prob.index.isin(event_weeks)]
    return {"prob_on_event_weeks": round(float(on.mean()), 3),
            "prob_other_weeks": round(float(off.mean()), 3),
            "n_events_in_window": int(len(on))}
