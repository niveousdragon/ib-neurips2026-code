"""Supplementary robustness tables of the hippocampus information plane, per cohort.

  (a) input axis: day-1 / day-4 cohort mean, mice with declining slope, one-sided Wilcoxon;
  (b) input-axis lambda = 0 vs lambda = 1 contrast: per-day mean +- SEM and the paired
      day 4 - day 1 change (Supplementary "Input-axis lambda = 0 vs lambda = 1 contrast");
  (c) per-mouse direction grid: sign of each axis' slope with a +-0.05 bits/day
      near-zero band (Supplementary "Per-mouse direction summary").

Inputs: results/hippocampus/ib_plane_lambda{0,1}.csv
Outputs: results/hippocampus/stats/si_input_axis.csv, si_lambda_contrast_per_day.csv,
         si_lambda_contrast_paired.csv, si_direction_grid.csv
Usage:  python hippocampus/stats_si_cohort.py
"""
import numpy as np
import pandas as pd
from hippo_common import per_mouse_slopes, stats_dir, table_path
from scipy import stats
from sessions import cohort

COHORTS = ("all64", "drop3", "drop5")


def ib(regime):
    return pd.read_csv(table_path(f"ib_plane_{regime}.csv"))


def input_axis():
    rows = []
    for c in COHORTS:
        d = cohort(ib("lambda1"), c)
        m = d.groupby("day").I_xz_corr.mean()
        s = per_mouse_slopes(d, "I_xz_corr")
        rows.append({"cohort": c, "day1_mean": m[1], "day4_mean": m[4],
                     "n_declining": int((s < 0).sum()), "n_mice": len(s),
                     "p_less": stats.wilcoxon(s, alternative="less").pvalue})
    return pd.DataFrame(rows)


def lambda_contrast():
    per_day, paired = [], []
    for lam in ("lambda0", "lambda1"):
        for c in COHORTS:
            d = cohort(ib(lam), c)
            for day, g in d.groupby("day"):
                v = g["I_xz_corr"].values
                per_day.append({"lambda": lam, "cohort": c, "day": day, "n": len(v),
                                "mean": v.mean(), "sem": v.std(ddof=1) / np.sqrt(len(v)),
                                "sd": v.std(ddof=1)})
            p = d.pivot(index="mouse", columns="day", values="I_xz_corr")
            delta = (p[4] - p[1]).dropna()
            paired.append({"lambda": lam, "cohort": c, "n": len(delta),
                           "mean_delta": delta.mean(), "n_negative": int((delta < 0).sum()),
                           "p_two_sided": stats.wilcoxon(delta).pvalue})
    return pd.DataFrame(per_day), pd.DataFrame(paired)


def classify(s, thr):
    return "inc" if s > thr else ("dec" if s < -thr else "near")


def direction_grid(thr=0.05):
    rows = []
    for lam in ("lambda1", "lambda0"):
        for c in COHORTS:
            d = cohort(ib(lam), c)
            sx, sy = per_mouse_slopes(d, "I_xz_corr"), per_mouse_slopes(d, "I_zy_bin_corr")
            cells = pd.crosstab(sy.map(lambda v: classify(v, thr)), sx.map(lambda v: classify(v, thr)))
            for ylab in ("inc", "near", "dec"):
                r = {"lambda": lam, "cohort": c, "threshold": thr, "Y": ylab}
                for xlab in ("dec", "near", "inc"):
                    r[f"X_{xlab}"] = int(cells.loc[ylab, xlab]) if (
                        ylab in cells.index and xlab in cells.columns) else 0
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    out = stats_dir()
    a = input_axis()
    a.to_csv(out / "si_input_axis.csv", index=False)
    print("[input axis]\n", a.round(4).to_string(index=False))
    b1, b2 = lambda_contrast()
    b1.to_csv(out / "si_lambda_contrast_per_day.csv", index=False)
    b2.to_csv(out / "si_lambda_contrast_paired.csv", index=False)
    print("[lambda contrast, per day: mean +- SEM]\n", b1.round(3).to_string(index=False))
    print("[lambda contrast, paired day4 - day1]\n", b2.round(4).to_string(index=False))
    c = direction_grid()
    c.to_csv(out / "si_direction_grid.csv", index=False)
    print("[direction grid: rows Y direction, columns X direction]\n", c.to_string(index=False))


if __name__ == "__main__":
    main()
