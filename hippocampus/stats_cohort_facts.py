"""Descriptive facts of the hippocampus cohort quoted in the paper.

  - matched neurons per mouse (Methods "Hippocampus": mean 200, median 158, range 29-435);
  - occupied 10 x 10 bins in the test fold per day, 61 sessions (Supplementary "Coverage
    shrinkage": ~74 / 64 / 63 / 58);
  - session-exclusion table: the five sessions with the lowest distance travelled, their
    cohort percentile and the effective dimensionality of X_ego (Supplementary
    "Session exclusion").

Inputs: results/hippocampus/session_summary.csv, behavior.csv
Output: printed; results/hippocampus/stats/cohort_facts.txt
Usage:  python hippocampus/stats_cohort_facts.py
"""
import pandas as pd
from hippo_common import stats_dir, table_path
from sessions import DROP3, cohort


def main():
    ss = pd.read_csv(table_path("session_summary.csv"))
    beh = pd.read_csv(table_path("behavior.csv"))
    L = []

    per_mouse = ss.groupby("mouse").n_matched.agg(["min", "max"])
    pm = per_mouse["min"]
    L.append(f"matched neurons constant across days within each mouse: "
             f"{bool((per_mouse['min'] == per_mouse['max']).all())}")
    L.append(f"matched neurons per mouse ({len(pm)}): mean {pm.mean():.1f}, median {pm.median():.1f}, "
             f"range {pm.min()}-{pm.max()}")

    v = cohort(ss, "drop3").groupby("day").n_bins_visited_test.agg(["count", "mean", "median"])
    L.append("occupied 10x10 bins in the test fold, drop3 (n, mean, median per day):")
    for day, r in v.iterrows():
        L.append(f"  day {day}: n={int(r['count'])}  mean {r['mean']:.1f}  median {r['median']:.0f}")

    j = beh.merge(ss[["mouse", "day", "X_dim_kept"]], on=["mouse", "day"]).sort_values("distance_traveled")
    j["rank"] = range(1, len(j) + 1)
    j["percentile"] = 100 * j["rank"] / len(j)
    L.append(f"lowest distance travelled ({len(j)} sessions):")
    for _, r in j.head(6).iterrows():
        tag = "excluded" if (r.mouse, int(r.day)) in DROP3 else "kept"
        L.append(f"  {r.mouse} day {int(r.day)}: distance {r.distance_traveled:.2f}, "
                 f"percentile {r.percentile:.1f}%, X_ego effective dim {int(r.X_dim_kept)}  [{tag}]")
    L.append("sessions with X_ego effective dim <= 3: "
             + ", ".join(f"{r.mouse} day {int(r.day)} ({int(r.X_dim_kept)})"
                         for _, r in j[j.X_dim_kept <= 3].iterrows()))
    txt = "\n".join(L)
    (stats_dir() / "cohort_facts.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
