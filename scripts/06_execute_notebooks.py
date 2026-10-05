"""
Step 6 - Execute every notebook in-place so it opens already populated with its
figures and tables.

The notebook *builders* (notebooks/build_*.py) regenerate the notebooks WITHOUT
outputs (clean source under version control); run this afterwards to render
them.  Needs ``nbconvert`` + ``ipykernel`` (``pip install nbconvert ipykernel``;
the kernel is registered with ``python -m ipykernel install --user``).

Run:  python scripts/06_execute_notebooks.py
"""
import pathlib
import subprocess
import sys

NB_DIR = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = [
    "French_OAT_Yield_Curve.ipynb",
    "Study1_Political_Risk_OAT_Bund.ipynb",
    "Study2_PCA_Curve_Regimes.ipynb",
    "Study3_Crisis_Dislocation_2020.ipynb",
    "Study1.1_Political_Premium_Decomposition.ipynb",
    "Study3.1_Regime_Detection.ipynb",
]


def main():
    for name in NOTEBOOKS:
        path = NB_DIR / name
        if not path.exists():
            print(f"skip (missing): {name}")
            continue
        print(f"executing {name} ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", "--ExecutePreprocessor.timeout=900",
             "--ExecutePreprocessor.kernel_name=python3", str(path)],
            check=True,
        )
    print("done - all notebooks executed and populated with figures/tables.")


if __name__ == "__main__":
    main()
