"""Mixed models on the relevance gain, the lambda = 0 control, POR, and the
input-axis centroids under the alternative flooring of R^2.

Relevance gain per (mouse, area) pair is dI = delta_baseline - delta_shuf
(bias-corrected change, validation_deltas.csv). The mixed model is
dI ~ 1 + random intercept per mouse; the paper reports the maximum-likelihood
fit (reml=False), the REML fit is printed for comparison. The lambda = 0
gain is corrected with the lambda = 1 shuffle null, because the null is a
property of the data and the folds, not of the supervision weight.

Runs from the shipped caches.
"""
import argparse
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from ib.paths import results_dir

# statsmodels convergence/boundary warnings on n = 21 pairs; estimates are
# printed and checked against the paper.
warnings.filterwarnings("ignore")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")
    vd = pd.read_csv(res / "validation_deltas.csv")
    vd["dI"] = vd.delta_baseline - vd.delta_shuf
    vd["dI0"] = vd.delta_lambda0
    vd["dI0c"] = vd.delta_lambda0 - vd.delta_shuf

    por = vd[vd.area == "POR"]
    print(f"POR: n={len(por)} pairs, mean dI {por.dI.mean():+.3f}, pos {int((por.dI > 0).sum())}/"
          f"{len(por)}, Wilcoxon one-sided p="
          f"{stats.wilcoxon(por.dI, alternative='greater').pvalue:.3f}; "
          f"values {por.set_index('mouse').dI.round(3).to_dict()}")

    print(f"lambda=1: mean dI {vd.dI.mean():+.4f}, pos {int((vd.dI > 0).sum())}/{len(vd)}; "
          f"per-mouse means {vd.groupby('mouse').dI.mean().round(3).to_dict()}")
    for col, label in [("dI", "lambda=1, corrected"),
                       ("dI0", "lambda=0, uncorrected"),
                       ("dI0c", "lambda=0, corrected by lambda=1 null")]:
        for reml in (False, True):
            m = smf.mixedlm(f"{col} ~ 1", vd, groups=vd["mouse"]).fit(reml=reml)
            fit = "REML" if reml else "ML"
            print(f"LMM {label:<38} {fit:>4}: intercept {m.params['Intercept']:+.4f}, "
                  f"p={m.pvalues['Intercept']:.2e}")
    for col, label in [("dI0", "lambda=0 raw"), ("dI0c", "lambda=0 corrected")]:
        pos_m = int((vd.groupby("mouse")[col].mean() > 0).sum())
        print(f"{label}: mean {vd[col].mean():+.3f}, pos {int((vd[col] > 0).sum())}/{len(vd)}, "
              f"per-mouse pos {pos_m}/5")

    T = pd.read_csv(res / "cortex_pairs_endpoints.csv")
    print("Centroid dR2 by flooring convention:\n" +
          T.groupby("group")[["dR2_clip", "dR2_clip_after_mean", "dR2_raw"]].mean()
          .round(3).to_string())
    print("Feature-cluster pairs:\n" +
          T[T.group == "feature (AL,RL)"][["mouse", "area", "R2_raw_n", "R2_raw_t", "dR2_clip",
                                            "dR2_clip_after_mean", "dR2_raw"]]
          .round(3).to_string())


if __name__ == "__main__":
    main()
