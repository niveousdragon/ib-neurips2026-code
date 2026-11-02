"""GLM table: what the probe latent encodes (Supplementary "Per-session GLM").

Delta r^2 = r^2(real latent) - r^2(mean of the shuffle latents) per session, latent rank
and predictor (from 09_glm_latents.py). Per cohort and regime: mean and max over the 16
ranks of the cohort-mean Delta r^2, and the number of ranks with one-sided Wilcoxon
(real > shuffle, over sessions) p < 0.001 (and also mean Delta r^2 > 0.05).

Inputs: results/hippocampus/glm_r2_lambda{1,0}.csv
Output: results/hippocampus/stats/glm_table.csv
Usage:  python hippocampus/stats_glm.py
"""
import pandas as pd
from hippo_common import stats_dir, table_path
from scipy import stats
from sessions import cohort


def table(df, regime, name):
    d = cohort(df, name).copy()
    d["delta"] = d.r2_real - d.r2_shuf
    n_sessions = d[["mouse", "day"]].drop_duplicates().shape[0]
    rows = []
    for p, g in d.groupby("predictor", sort=False):
        dm = g.groupby("rank").delta.mean()
        pv = g.groupby("rank").delta.apply(lambda x: stats.wilcoxon(x, alternative="greater").pvalue)
        rows.append(dict(regime=regime, cohort=name, n_sessions=n_sessions, predictor=p,
                         mean_dr2=dm.mean(), max_dr2=dm.max(),
                         n_dims_p001=int((pv < 1e-3).sum()),
                         n_dims_p001_and_gt005=int(((pv < 1e-3) & (dm > 0.05)).sum())))
    return rows


def main():
    rows = []
    for regime in ("lambda1", "lambda0"):
        df = pd.read_csv(table_path(f"glm_r2_{regime}.csv"))
        for name in ("all64", "drop3"):
            rows += table(df, regime, name)
    res = pd.DataFrame(rows)
    out = stats_dir() / "glm_table.csv"
    res.to_csv(out, index=False)
    print(res.round(3).to_string(index=False))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
