"""
Step 5 (for Study 1) - Fit the German Bund zero-coupon curve with the same
Svensson engine as the OATs, and cache it.  Enables an exact OAT-Bund safety
premium instead of the ECB AAA proxy.

Run:  python scripts/05_fit_bund_curve.py        (~5-8 min, one-off, cached)
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from studies import bund_curve


def main():
    t0 = time.time()
    params, zero = bund_curve.fit_and_cache(force=True, progress=True)
    print(f"\nFitted {len(params)} Bund curves in {time.time() - t0:.0f}s")
    euro = params  # whole Bund sample is euro-era (1999+)
    print(f"Bund overall MAE: mean={euro['mae'].mean():.2f} bp, max={euro['mae'].max():.2f} bp")
    print(f"Bund 10y: {zero['10y'].iloc[-1]*100:.2f}%  (latest {zero.index.max().date()})")
    print("Cached bund_params.parquet + bund_zero.parquet to output/studies_cache/")


if __name__ == "__main__":
    main()
