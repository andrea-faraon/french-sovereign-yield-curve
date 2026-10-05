"""
Supplementary studies built on top of the (untouched) OATmeals replication.

Three self-contained extensions to the most recent, post-paper years:

    political_risk      - Study 1: OAT-Bund safety premium & political-shock
                          event study (extends Section 7.1; UK/Brexit model).
    pca_regimes         - Study 2: PCA of the curve across monetary regimes and
                          the 2022-2024 inversion (extends Section 5.3).
    crisis_dislocation  - Study 3: March-2020 dislocation vs 2008/2011 via the
                          MAE / noise measures (extends Section 8.2).

These modules import the base ``oatcurve`` package read-only and reuse its
cached daily fit; external benchmark/risk data come from the ECB (see
``external_data``).  They never modify the base project or its outputs.
"""
from . import (bund_curve, common, crisis_dislocation, external_data,  # noqa: F401
               pca_regimes, political_premium, political_risk, regime_detection)

__all__ = ["common", "external_data", "bund_curve", "political_risk",
           "political_premium", "pca_regimes", "crisis_dislocation",
           "regime_detection"]
