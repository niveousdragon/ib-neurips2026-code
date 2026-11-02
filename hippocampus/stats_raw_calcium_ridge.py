"""Raw-calcium Ridge table: median test R^2 per target over the 64 sessions (and the 61).

Input:  results/hippocampus/raw_calcium_ridge.csv (legacy_03_raw_calcium_ridge.py)
Output: printed; results/hippocampus/stats/raw_calcium_ridge.txt
Usage:  python hippocampus/stats_raw_calcium_ridge.py
"""
import pandas as pd
from hippo_common import stats_dir, table_path
from sessions import cohort


def main():
    sk = pd.read_csv(table_path("raw_calcium_ridge.csv"))
    L = []
    for name, label in (("all64", "64 sessions"), ("drop3", "61 sessions")):
        g = cohort(sk, name).groupby("var", sort=False).R2.agg(["count", "median", "mean"])
        L.append(f"[{label}] Ridge test R^2 per target\n" + g.round(3).to_string())
    txt = "\n".join(L)
    (stats_dir() / "raw_calcium_ridge.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
