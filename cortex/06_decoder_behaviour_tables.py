"""Decoder-versus-behaviour tables of the SI (per pair, per mouse, mixed model).

Two session sets, both from cortex_sessions_with_stage.csv:
  category_stage  all K = 10 sessions except F04 AL 20160212, a session from
                  the baseline period that has K = 10 only because 10 of its 12
                  stimuli occur among its 27 valid trials (reported in the paper)
  learned_stage   sessions labelled "Learned Category Task" only (check)

Several sessions of one pair are averaged; per-mouse behaviour is the mean
over that mouse's pairs. The mixed model (gap ~ 1 + random intercept per
mouse) is fitted by maximum likelihood.

Writes results/cortex/decoder_behaviour_table_<set>.tex (table rows) and
prints the summary numbers. Runs from the shipped caches.
"""
import argparse
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from ib.paths import results_dir

# The statsmodels fit warns about the boundary (zero between-mouse variance)
# on the category-stage set; that estimate is the reported result.
warnings.filterwarnings("ignore")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")
    S = pd.read_csv(res / "cortex_sessions_with_stage.csv", dtype={"date": str})
    sets = {
        "category_stage": S[(S.K == 10) & ~((S.mouse == "F04") & (S.area == "AL")
                                            & (S.date == "20160212"))],
        "learned_stage": S[S.stage == "Learned Category Task"],
    }
    vd = pd.read_csv(res / "validation_deltas.csv")
    gain = (vd.delta_baseline - vd.delta_shuf).groupby(vd.mouse).mean()

    for k, sub in sets.items():
        pp = (sub.groupby(["mouse", "area"])
              .agg(dec=("acc_dec", "mean"), beh=("beh", "mean"), n=("date", "size"))
              .reset_index())
        pp["gap"] = (pp.dec - pp.beh) * 100
        rows = "\n".join(f"{r.mouse} & {r.area:<3} & {r.dec:.3f} & {r.beh:.3f} & {r.n} & "
                         f"${r.gap:+.1f}$ \\\\" for r in pp.itertuples())
        pm = pp.groupby("mouse").agg(na=("area", "size"), dec=("dec", "mean"))
        pm["beh"] = pp.groupby("mouse").beh.mean()
        pm["gap"] = (pm.dec - pm.beh) * 100
        rows2 = "\n".join(f"{m} & {int(r.na)} & {r.dec:.3f} & {r.beh:.3f} & ${r.gap:+.1f}$ \\\\"
                          for m, r in pm.iterrows())
        lmm = smf.mixedlm("gap ~ 1", pp, groups=pp["mouse"]).fit(reml=False)
        w = stats.wilcoxon(pm.gap, alternative="greater").pvalue
        print(f"[{k}] pair mean gap {pp.gap.mean():+.2f} pp; positive "
              f"{int((pp.gap > 0).sum())}/{len(pp)}; range {pp.gap.min():+.1f}..{pp.gap.max():+.1f}; "
              f"per-mouse mean {pm.gap.mean():+.2f}, range {pm.gap.min():+.1f}..{pm.gap.max():+.1f}, "
              f"all positive {bool((pm.gap > 0).all())}, Wilcoxon one-sided p={w:.3f}; "
              f"LMM (ML) intercept {lmm.params['Intercept']:+.2f} pp "
              f"p={lmm.pvalues['Intercept']:.2g}; group var {float(lmm.cov_re.iloc[0, 0]):.3f}; "
              f"excluding F02: {pm.drop('F02').gap.mean():+.2f}")
        (res / f"decoder_behaviour_table_{k}.tex").write_text(
            "% per-pair rows\n" + rows + "\n% per-mouse rows\n" + rows2 + "\n")

        beh = pp.groupby("mouse").beh.mean()
        j = pd.concat([beh, gain], axis=1)
        r = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]
        print(f"[{k}] Spearman(behaviour, corrected relevance gain) = {r:+.2f}, n={len(j)}; "
              f"behaviour {beh.round(3).to_dict()}")


if __name__ == "__main__":
    main()
