"""Cortex numbers of the paper that no other step prints, next to the paper value.

All from the shipped caches:
  headline     cohort relevance gain dI = delta_baseline - delta_shuf over the
               21 pairs, one-sided Wilcoxon; the semantic-leaning cluster;
               V1 per mouse, leave-one-out range, one-sided binomial sign test
  clusters     one-sided Mann-Whitney C2 > C1 on dI under the STRICT and the
               stream-based cluster definitions
  input axis   change of the input-axis shuffle null R2_shuf between the
               Figure 3 endpoints; Gaussian re-expression -1/2 log2(1 - R^2)
               of the endpoint R^2, averaged per pair within each group
  cohort       mice, pairs, areas and neuron counts of the analysed sessions

With --data-dir pointing at the deposit's data/chronicrecordings folder (the
files are only listed, not read), it also counts mice and recordings in the
whole dataset.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from goltstein import PAPER_MICE
from scipy import stats

from ib.paths import results_dir

C2_STRICT = {"V1", "PM", "P", "POR"}
C2_STREAM = C2_STRICT | {"LM", "LI"}
C1 = {"AL", "RL"}


def show(label, paper, computed):
    print(f"  {label:<58} paper {paper:<16} computed {computed}")


def gaussian_mi(r2):
    # R^2 is capped below 1 so the transform stays finite
    return -0.5 * np.log2(1 - np.clip(r2, 0, 0.999))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="deposit data/chronicrecordings folder, only listed (optional)")
    args = p.parse_args()
    res = results_dir("cortex")

    vd = pd.read_csv(res / "validation_deltas.csv")
    vd["dI"] = vd.delta_baseline - vd.delta_shuf
    T = pd.read_csv(res / "cortex_pairs_endpoints.csv", dtype={"date_n": str, "date_t": str})
    S = pd.read_csv(res / "cortex_sessions_with_stage.csv", dtype={"date": str})
    xs = pd.read_csv(res / "xaxis_shuffle_per_session.csv", dtype={"date": str})

    print("Headline relevance gain (main text, cortex results)")
    d = vd.dI
    show("cohort mean dI (bits)", "+0.255", f"{d.mean():+.4f}")
    show("pairs positive", "16/21", f"{int((d > 0).sum())}/{len(d)}")
    show("one-sided Wilcoxon p", "0.0006",
         f"{stats.wilcoxon(d, alternative='greater').pvalue:.4f}")
    c2 = vd[vd.area.isin(C2_STREAM)].dI
    show("semantic-leaning cluster mean dI", "+0.266", f"{c2.mean():+.4f}")
    show("semantic-leaning cluster positive", "13/16", f"{int((c2 > 0).sum())}/{len(c2)}")
    # The main text gives p < 0.001 (one-sided); the SI cluster table gives
    # two-sided p, as the Methods state for cluster comparisons.
    show("semantic-leaning cluster Wilcoxon p one-sided; two-sided", "< 0.001; SI 0.001",
         f"{stats.wilcoxon(c2, alternative='greater').pvalue:.4f}; "
         f"{stats.wilcoxon(c2).pvalue:.4f}")
    c1 = vd[vd.area.isin(C1)].dI
    show("feature-leaning cluster mean dI, positive", "+0.22, 3/5",
         f"{c1.mean():+.4f}, {int((c1 > 0).sum())}/{len(c1)}")
    show("feature-leaning cluster Wilcoxon p one-sided; two-sided", "> 0.1; SI 0.44",
         f"{stats.wilcoxon(c1, alternative='greater').pvalue:.4f}; "
         f"{stats.wilcoxon(c1).pvalue:.4f}")

    print("\nV1 per mouse (SI)")
    v1 = vd[vd.area == "V1"].set_index("mouse").dI
    show("per-mouse dI F02 F03 F04 K01 K02", "+.289 +.706 +.525 +.549 +.429",
         " ".join(f"{v:+.3f}" for v in v1.loc[list(PAPER_MICE)]))
    show("mean, s.d., range", "+0.500, 0.154, +0.289..+0.706",
         f"{v1.mean():+.3f}, {v1.std():.3f}, {v1.min():+.3f}..{v1.max():+.3f}")
    loo = [v1.drop(m).mean() for m in v1.index]
    loo_pos = [int((v1.drop(m) > 0).sum()) for m in v1.index]
    show("leave-one-out mean range; positive of 4", "+0.45..+0.55; 4/4",
         f"{min(loo):+.3f}..{max(loo):+.3f}; {min(loo_pos)}/4")
    show("one-sided Wilcoxon p", "1/32 = 0.031",
         f"{stats.wilcoxon(v1, alternative='greater').pvalue:.4f}")
    show("one-sided binomial sign test p", "(1/2)^5 = 0.031",
         f"{stats.binomtest(int((v1 > 0).sum()), len(v1), 0.5, alternative='greater').pvalue:.4f}")

    print("\nCluster contrast, one-sided Mann-Whitney C2 > C1 on dI (SI)")
    for label, c2_areas, paper in (("STRICT C2 {V1,PM,P,POR} vs {AL,RL}", C2_STRICT, "0.42"),
                                   ("stream-based C2 (+LM,LI) vs {AL,RL}", C2_STREAM, "0.45")):
        a = vd[vd.area.isin(c2_areas)].dI.values
        b = vd[vd.area.isin(C1)].dI.values
        pv = stats.mannwhitneyu(a, b, alternative="greater").pvalue
        show(f"{label} (n={len(a)} vs {len(b)})", paper, f"{pv:.4f}")
    gap = [vd[vd.area.isin(c)].dI.mean() - c1.mean() for c in (C2_STRICT, C2_STREAM)]
    show("magnitude difference C2 - C1 (bits)", "0.05-0.08",
         f"{min(gap):.3f}-{max(gap):.3f}")

    print("\nInput-axis shuffle null between the Figure 3 endpoints (SI)")
    nv = xs[xs.K == 12].sort_values("date").groupby(["mouse", "area"]).first()
    tr = xs[xs.K == 10].sort_values("date").groupby(["mouse", "area"]).last()
    J = nv.join(tr, lsuffix="_n", rsuffix="_t", how="inner")
    dsh = J.R2_shuf_mean_t - J.R2_shuf_mean_n
    show(f"mean change of R2_shuf over {len(J)} pairs (s.d.)", "+0.0011 (0.0019)",
         f"{dsh.mean():+.4f} ({dsh.std():.4f})")
    show("per-session R2_shuf median, range (69 sessions)", "0.000, [0.000, 0.009]",
         f"{xs.R2_shuf_mean.median():.3f}, [{xs.R2_shuf_mean.min():.3f}, "
         f"{xs.R2_shuf_mean.max():.3f}] (n={len(xs)})")

    print("\nGaussian re-expression -1/2 log2(1 - R^2), mean per-pair change (SI)")
    T["dMI"] = gaussian_mi(T.R2_clip_t) - gaussian_mi(T.R2_clip_n)
    paper_mi = {"V1": "+0.35", "semantic ex-V1": "0.00", "feature (AL,RL)": "-0.04"}
    for g, val in T.groupby("group").dMI.mean().items():
        show(f"{g} (n={int((T.group == g).sum())})", paper_mi[g], f"{val:+.3f}")

    print("\nCohort (methods and Table 1)")
    show("analysed mice", "5 (F02 F03 F04 K01 K02)",
         f"{S.mouse.nunique()} ({' '.join(sorted(S.mouse.unique()))})")
    show("mouse-area recordings; areas", "21; 8",
         f"{S.groupby(['mouse', 'area']).ngroups}; {S.area.nunique()}")
    show("neurons per session (first imaging plane)", "157-308",
         f"{S.n_neurons.min()}-{S.n_neurons.max()} over {len(S)} sessions")

    if args.data_dir:
        root = Path(args.data_dir)
        recs = sorted({(f.parent.parent.name, f.parent.name) for f in root.glob("*/*/*.mat")})
        mice = sorted({m for _, m in recs})
        ours = [r for r in recs if r[1] in PAPER_MICE]
        print("\nWhole dataset (file listing of the deposit)")
        show("mice in dataset", "10", f"{len(mice)} ({' '.join(mice)})")
        show("chronic recordings (area, mouse) in dataset", "39", f"{len(recs)}")
        show("recordings of the analysed mice; not analysed", "21; 18",
             f"{len(ours)}; {len(recs) - len(ours)}")


if __name__ == "__main__":
    main()
