"""
Step 2 - Estimate the Svensson curve on every trading day of the full sample
(eq. 14) and cache the parameters, fit diagnostics, MAE/noise metrics and the
reconstructed zero / par / forward curves.

Run:  python scripts/02_fit_curves.py            (whole sample, ~10-15 min, one-off)
      python scripts/02_fit_curves.py 2015 2026   (a date range, for quick tests)
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from oatcurve import config, data_loading as dl, pipeline


def rebuild_curves(params):
    """Reconstruct and cache zero/par/forward curves from fitted parameters
    (cheap: no re-fitting).  The zero curve covers every tenor needed by the
    term-structure factors and the PCA (so includes 1-10y integers)."""
    tenors = sorted(set(config.STANDARD_TENORS) | set(config.PCA_TENORS))
    pipeline.zero_curve(params, tenors).to_parquet(config.CACHE_DIR / "zero_curve.parquet")
    pipeline.par_curve(params, config.STANDARD_TENORS).to_parquet(config.CACHE_DIR / "par_curve.parquet")
    pipeline.forward_curve(params, config.STANDARD_TENORS).to_parquet(config.CACHE_DIR / "forward_curve.parquet")


def main(start=None, end=None):
    if (config.CACHE_DIR / "panel.parquet").exists():
        panel = pd.read_parquet(config.CACHE_DIR / "panel.parquet")
        master = pd.read_parquet(config.CACHE_DIR / "master.parquet")
        bonds = dl.build_bonds(master, panel, config.FILTERS)
    else:
        master, panel, bonds = dl.load_all()

    if start or end:
        panel = panel.loc[str(start or "") or None:str(end or "") or None]

    print(f"Fitting Svensson curve on {len(panel)} trading days, "
          f"{len(bonds)} candidate bonds ...")
    t0 = time.time()
    params = pipeline.run_fit(bonds, panel, compute_metrics=True, progress=True)
    print(f"Done: {len(params)} curves fitted in {time.time() - t0:.0f}s")

    params.to_parquet(config.CACHE_DIR / "fit_params.parquet")
    rebuild_curves(params)

    euro = params.loc[config.EURO_START:]
    print(f"\nEuro-sample overall MAE : mean={euro['mae'].mean():.2f} bp, "
          f"max={euro['mae'].max():.2f} bp")
    print(f"Cached parameters + curves to {config.CACHE_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args + [None, None])[:2])
