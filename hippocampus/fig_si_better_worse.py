"""Supplementary figure: better and worse learners (61 sessions).

a. retention per day for mice above (blue) and below (red) the median of day-4
relevance; b. relevance for the same groups (day 4 shaded: the ranking variable);
c. day-4 retention against day-4 relevance, one point per mouse.

Inputs: results/hippocampus/ib_plane_lambda1.csv
Output: <figures>/si_bestworst_paper61.png
Usage:  python hippocampus/fig_si_better_worse.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from stats_better_worse import DAYS, pivots, split_by_relevance

from ib.paths import figures_dir


def main():
    ret, rel = pivots()
    hi, lo = split_by_relevance(rel)
    both = pd.concat([ret[4], rel[4]], axis=1, keys=["ret", "rel"]).dropna()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.95), constrained_layout=True)
    col = {"better": "#1b6ca8", "worse": "#c0392b"}
    for a, piv, title in ((ax[0], ret, r"Retention $I(X_\mathrm{ego};Z)$"),
                          (ax[1], rel, r"Relevance $I(Z;Y_\mathrm{alloc})$")):
        for g, lab in ((hi, "better"), (lo, "worse")):
            M = piv.loc[g, DAYS]
            for _, row in M.iterrows():
                a.plot(DAYS, row.values, color=col[lab], alpha=0.18, lw=0.6)
            a.errorbar(DAYS, M.mean().values, yerr=M.sem().values, color=col[lab], marker="o",
                       ms=3, lw=1.3, capsize=2, elinewidth=0.8, label=f"{lab} learners (n = {len(g)})")
        a.set_xticks(DAYS)
        a.set_xlabel("day")
        a.set_title(title)
    ax[0].set_ylabel("bias-corrected MI (bits)")
    ax[1].set_ylabel("bias-corrected MI (bits)")
    ax[1].axvspan(3.6, 4.4, color="0.88", zorder=0)
    ax[1].text(4, 0.97, "ranking\nday", transform=ax[1].get_xaxis_transform(),
               ha="center", va="top", fontsize=6)
    ax[0].legend(frameon=False, loc="lower left", handlelength=1.4, borderaxespad=0.2)
    s = stats.spearmanr(both.ret, both.rel)
    for m, row in both.iterrows():
        ax[2].scatter(row.rel, row.ret, color=col["better"] if m in hi else col["worse"], s=10)
    ax[2].set_xlabel("day-4 relevance (bits)")
    ax[2].set_ylabel("day-4 retention (bits)")
    ax[2].set_title(f"Spearman {s.correlation:.2f} (n = {len(both)})")
    for a, letter in zip(ax, "abc"):
        a.text(-0.02, 1.0, letter, transform=a.transAxes, fontweight="bold", fontsize=8,
               ha="right", va="bottom")
    out = figures_dir() / "si_bestworst_paper61.png"
    fig.savefig(out, dpi=300)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
