"""SI figure: relevance gain against behavioural accuracy, one point per mouse.

Behaviour: category-stage (K = 10) sessions without the baseline-period
session F04 AL 20160212, averaged within each (mouse, area) pair and then over
the mouse's pairs (as in 06_decoder_behaviour_tables.py). Relevance gain:
bias-corrected dI = delta_baseline - delta_shuf, mean over the mouse's pairs.
The exact two-sided p enumerates all 5! = 120 orderings.

Reads results/cortex; writes figures_dir()/fig_supp_cortex_outcome.png.
"""
import argparse
from itertools import permutations
from math import factorial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from ib.paths import figures_dir, results_dir

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")
    S = pd.read_csv(res / "cortex_sessions_with_stage.csv", dtype={"date": str})
    sub = S[(S.K == 10) & ~((S.mouse == "F04") & (S.area == "AL") & (S.date == "20160212"))]
    beh = sub.groupby(["mouse", "area"]).beh.mean().groupby("mouse").mean()
    vd = pd.read_csv(res / "validation_deltas.csv")
    gain = (vd.delta_baseline - vd.delta_shuf).groupby(vd.mouse).mean()
    j = pd.concat([beh.rename("beh"), gain.rename("g")], axis=1)
    r = stats.spearmanr(j.beh, j.g)[0]
    c = sum(abs(stats.spearmanr(j.beh, j.g.values[list(p)])[0]) >= abs(r) - 1e-12
            for p in permutations(range(len(j))))
    p_exact = c / factorial(len(j))
    print(j.round(4).to_string())
    print(f"Spearman {r:+.2f}, exact two-sided p = {p_exact:.3f}, n = {len(j)}")

    fig, ax = plt.subplots(figsize=(2.75, 2.1), layout="constrained")
    ax.scatter(j.beh * 100, j.g, s=14, color="k")
    for m, row in j.iterrows():
        ax.annotate(m, (row.beh * 100, row.g), xytext=(3, 2), textcoords="offset points",
                    fontsize=6)
    ax.set_xlabel("behaviour, category stage (% correct)")
    ax.set_ylabel(r"relevance gain $\Delta I(Z;Y)$ (bits)")
    ax.set_title(f"Spearman {r:.2f}, exact p = {p_exact:.2f}, n = {len(j)} mice")
    ax.set_xlim(50, 85)
    ax.set_ylim(-0.02, 0.6)
    out = figures_dir() / "fig_supp_cortex_outcome.png"
    fig.savefig(out, dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
