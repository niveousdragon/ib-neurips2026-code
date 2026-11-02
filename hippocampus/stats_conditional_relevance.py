"""Summary of the target-conditioned relevance I(Z; Y_alloc | X_ego) (Supplementary
"Target-conditioned relevance"; paper: 59/61 sessions > 0, mean 0.24 bits, day 1 0.20 ->
day 4 0.30, rising in 11/16 mice, one-sided Wilcoxon p = 0.072).

Input:  results/hippocampus/conditional_relevance.csv (+ ib_plane_lambda1.csv as a check
        that rel_corr reproduces the relevance axis)
Output: results/hippocampus/stats/conditional_relevance_summary.txt
Usage:  python hippocampus/stats_conditional_relevance.py
"""
import pandas as pd
from hippo_common import per_mouse_slopes, stats_dir, table_path
from scipy import stats
from sessions import cohort


def main():
    d0 = pd.read_csv(table_path("conditional_relevance.csv"))
    a1 = pd.read_csv(table_path("ib_plane_lambda1.csv"))
    L = []
    m = d0.merge(a1[["mouse", "day", "I_zy_bin_corr"]], on=["mouse", "day"])
    L.append(f"check: rel_corr vs relevance axis I_zy_bin_corr Spearman "
             f"{stats.spearmanr(m.rel_corr, m.I_zy_bin_corr).correlation:.3f}, "
             f"mean |diff| {(m.rel_corr - m.I_zy_bin_corr).abs().mean():.3f}")
    for name, label in (("drop3", "61"), ("all64", "64")):
        d = cohort(d0, name)
        L.append(f"=== {label} sessions ({len(d)}) ===")
        L.append(f"I(X;Y) decoder mean {d.I_x_y.mean():.3f}; rel_corr mean {d.rel_corr.mean():.3f}; "
                 f"CMI_corr mean {d.cmi_corr.mean():.3f} median {d.cmi_corr.median():.3f}; "
                 f"CMI>0 in {(d.cmi_corr > 0).sum()}/{len(d)}; "
                 f"ratio CMI/rel (means) {d.cmi_corr.mean() / d.rel_corr.mean():.2f}")
        L.append("by day CMI: " + str(d.groupby("day").cmi_corr.mean().round(3).to_dict())
                 + " rel: " + str(d.groupby("day").rel_corr.mean().round(3).to_dict()))
        for c in ("cmi_corr", "rel_corr", "I_x_y"):
            sl = per_mouse_slopes(d, c).dropna()
            L.append(f"slope {c}: median {sl.median():+.3f}, up {(sl > 0).sum()}/{len(sl)}, "
                     f"p_one(greater) {stats.wilcoxon(sl, alternative='greater').pvalue:.3f}, "
                     f"p_two {stats.wilcoxon(sl).pvalue:.3f}")
    out = stats_dir() / "conditional_relevance_summary.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
