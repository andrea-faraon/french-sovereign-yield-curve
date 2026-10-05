"""
Step 4 - On-the-run premium (Section 6, eq. 17).

Re-fits the curve every day on the cross-section that excludes the on-the-run
and first off-the-run bonds, then measures the 5- and 10-year premiums.  By
default it runs on the euro sample at weekly frequency (the daily second fit is
expensive and the series is smooth); pass ``--daily`` for every trading day.

Run:  python scripts/04_on_the_run.py [--daily]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oatcurve import config, data_loading as dl, otr


def main(daily=False):
    panel = pd.read_parquet(config.CACHE_DIR / "panel.parquet")
    master = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    bonds = dl.build_bonds(master, panel, config.FILTERS)

    dates = panel.loc[config.EURO_START:].index
    if not daily:
        dates = dates[::5]                      # ~weekly
    print(f"Computing on-the-run premium on {len(dates)} dates "
          f"({'daily' if daily else 'weekly'}) ...")

    prem = otr.on_the_run_premium(bonds, panel, dates=dates, target_tenors=(5, 10))
    prem.to_parquet(config.CACHE_DIR / "otr_premium.parquet")
    prem.to_csv(config.TABLE_DIR / "otr_premium.csv")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, t in zip(axes, (5, 10)):
        col = f"otr_{t}y"
        ax.plot(prem.index, prem[col], lw=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(prem[col].mean(), color="tab:red", ls="--", lw=0.8,
                   label=f"mean = {prem[col].mean():.2f} bp")
        ax.set(title=f"Panel: {t}-year on-the-run premium", ylabel="bp")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig. 10 - On-the-run premium on the French OAT market", y=0.995)
    fig.savefig(config.FIG_DIR / "fig10_on_the_run_premium.png", bbox_inches="tight")
    plt.close(fig)

    print("\nOn-the-run premium summary (bp):")
    for t in (5, 10):
        s = prem[f"otr_{t}y"].dropna()
        print(f"  {t:>2}y: mean={s.mean():6.2f}  median={s.median():6.2f}  "
              f"std={s.std():5.2f}  [min={s.min():.1f}, max={s.max():.1f}]")
    print(f"\nSaved figure + table. (Paper finding: the premium is negligible.)")


if __name__ == "__main__":
    main(daily="--daily" in sys.argv)
