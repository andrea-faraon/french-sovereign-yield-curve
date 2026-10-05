"""
Step 1 - Load the raw Bloomberg exports, apply the static filters of
Section 4.3 and cache a clean security master and daily bid-price panel.

Run:  python scripts/01_build_panel.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from oatcurve import config, data_loading as dl


def main():
    print("Loading security master and daily bid panel ...")
    master = dl.load_security_master()
    panel = dl.drop_abnormal_quotes(dl.load_bid_panel(), config.FILTERS.abnormal_quotes)
    bonds = dl.build_bonds(master, panel, config.FILTERS)

    # Restrict the master cache to the bonds that survive the static filters
    # and actually have prices, and tag those kept for fitting.
    master = master.copy()
    master["in_panel"] = master["isin"].isin(panel.columns)
    master["kept_for_fit"] = master["isin"].isin(bonds.keys())

    panel.to_parquet(config.CACHE_DIR / "panel.parquet")
    master.to_parquet(config.CACHE_DIR / "master.parquet")

    print(f"\nSecurities in master              : {len(master)}")
    print(f"  - with price data               : {int(master['in_panel'].sum())}")
    print(f"  - kept for fitting (post-filter): {len(bonds)}")
    print(f"    OAT / BTAN                     : "
          f"{(master.query('kept_for_fit')['security_type'] == 'OAT').sum()} / "
          f"{(master.query('kept_for_fit')['security_type'] == 'BTAN').sum()}")
    print(f"    retail France-Physiques        : "
          f"{int(master.query('kept_for_fit')['is_retail'].sum())} "
          f"(included={config.FILTERS.include_retail_oat})")
    print(f"\nPanel: {panel.shape[0]} trading days x {panel.shape[1]} securities")
    print(f"  {panel.index.min().date()}  ->  {panel.index.max().date()}")
    daily_n = panel.notna().sum(axis=1)
    print(f"  bonds/day: min={daily_n.min()}, median={int(daily_n.median())}, "
          f"max={daily_n.max()}")
    print(f"\nCached to {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
