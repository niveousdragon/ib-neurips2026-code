"""Better and worse learners (Supplementary "Better and worse learners").

Ranking: day-4 relevance I(Z; Y_alloc) of the 14 mice with a retained day-4 session,
median split 7 vs 7 (H32 and H06 have no retained day 4 and are not ranked).
  - retention per day, better vs worse, two-sided Mann-Whitney, Holm over the four days;
    day-1 relevance; day 4 - day 1 retention change and four-day retention slope;
  - day-4 retention vs day-4 relevance (Spearman; partial Spearman with the number of
    matched neurons regressed out of the ranks) and vs day-4 arena-coverage entropy;
  - joint states: how many mice whose retention fell had relevance rise / fall, per
    regime and cohort, with no split, the relevance split and a coverage-entropy split.

Inputs (results/hippocampus/): ib_plane_lambda{0,1}.csv, behavior.csv, session_summary.csv
Outputs (results/hippocampus/stats/): better_worse_tests.csv, better_worse_states.csv
Usage:  python hippocampus/stats_better_worse.py
"""
import numpy as np
import pandas as pd
from hippo_common import stats_dir, table_path
from scipy import stats
from sessions import COHORTS, cohort

DAYS = [1, 2, 3, 4]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (m - i) * p[j]))
        adj[j] = run
    return adj


def pivots(regime="lambda1", name="drop3"):
    df = cohort(pd.read_csv(table_path(f"ib_plane_{regime}.csv")), name)
    return (df.pivot(index="mouse", columns="day", values="I_xz_corr"),
            df.pivot(index="mouse", columns="day", values="I_zy_bin_corr"))


def split_by_relevance(rel):
    """Better / worse halves by day-4 relevance (mice without day 4 are not ranked)."""
    perf = rel[4].dropna().sort_values(ascending=False)
    h = len(perf) // 2
    return list(perf.index[:h]), list(perf.index[h:])


def slopes(piv):
    out = {}
    for m, row in piv.iterrows():
        y = row.dropna()
        out[m] = stats.linregress(y.index.values.astype(float), y.values).slope if len(y) >= 2 else np.nan
    return pd.Series(out)


def coverage_day4():
    beh = pd.read_csv(table_path("behavior.csv"))
    return beh[beh.day == 4].set_index("mouse")["coverage_entropy_y_bin"]


def group_tests(ret, rel, hi, lo):
    mw = lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    rows, ps = [], []
    for d in DAYS:
        a, b = ret.loc[hi, d].dropna(), ret.loc[lo, d].dropna()
        ps.append(mw(a, b))
        rows.append(dict(test=f"retention day{d} better vs worse", n_hi=len(a), n_lo=len(b),
                         med_hi=a.median(), med_lo=b.median(), p=ps[-1]))
    for d, pa in zip(DAYS, holm(ps)):
        rows.append(dict(test=f"Holm retention day{d}", p=pa))
    a, b = rel.loc[hi, 1].dropna(), rel.loc[lo, 1].dropna()
    rows.append(dict(test="relevance day1 better vs worse", n_hi=len(a), n_lo=len(b),
                     med_hi=a.median(), med_lo=b.median(), p=mw(a, b)))
    delta = ret[4] - ret[1]
    a, b = delta.loc[hi].dropna(), delta.loc[lo].dropna()
    rows.append(dict(test="retention change day4-day1 better vs worse", n_hi=len(a), n_lo=len(b),
                     med_hi=a.median(), med_lo=b.median(), p=mw(a, b)))
    sl = slopes(ret)
    a, b = sl.loc[hi].dropna(), sl.loc[lo].dropna()
    rows.append(dict(test="retention slope better vs worse", n_hi=len(a), n_lo=len(b),
                     med_hi=a.median(), med_lo=b.median(), p=mw(a, b)))

    both = pd.concat([ret[4], rel[4]], axis=1, keys=["ret", "rel"]).dropna()
    r = stats.spearmanr(both.ret, both.rel)
    rows.append(dict(test="Spearman day4 retention vs day4 relevance", n_hi=len(both),
                     rho=r.correlation, p=r.pvalue))
    nm = pd.read_csv(table_path("session_summary.csv")).groupby("mouse").n_matched.first()
    rk = pd.concat([both, nm], axis=1, join="inner").rename(columns={"n_matched": "nm"}).rank()

    def resid(a, b):
        return rk[a] - np.polyval(np.polyfit(rk[b], rk[a], 1), rk[b])

    pr = stats.pearsonr(resid("ret", "nm"), resid("rel", "nm"))
    rows.append(dict(test="partial Spearman day4 retention vs relevance | n_matched",
                     n_hi=len(rk), rho=pr[0], p=pr[1]))
    j = pd.concat([ret[4], coverage_day4()], axis=1, keys=["ret", "cov"]).dropna()
    r = stats.spearmanr(j["ret"], j["cov"])
    rows.append(dict(test="Spearman day4 retention vs day4 coverage entropy", n_hi=len(j),
                     rho=r.correlation, p=r.pvalue))
    return pd.DataFrame(rows)


def joint_states():
    """Day 4 - day 1 change on both axes, counted per mouse."""
    rows = []
    cov4 = coverage_day4()
    for regime in ("lambda1", "lambda0"):
        for name in COHORTS:
            ret, rel = pivots(regime, name)
            dx, dy = ret[4] - ret[1], rel[4] - rel[1]
            ok = dx.notna() & dy.notna()
            rows.append(dict(regime=regime, cohort=name, split="none", group="all", n=int(ok.sum()),
                             ret_fall=int((ok & (dx < 0)).sum()),
                             ret_fall_rel_rise=int((ok & (dx < 0) & (dy > 0)).sum()),
                             ret_fall_rel_fall=int((ok & (dx < 0) & (dy < 0)).sum())))
            for sname, key in (("relevance_day4", rel[4]), ("coverage_day4", cov4.loc[rel[4].dropna().index])):
                perf = key.dropna().sort_values(ascending=False)
                h = len(perf) // 2
                for gname, g in (("better", list(perf.index[:h])), ("worse", list(perf.index[h:]))):
                    a, b = dx.loc[g].dropna(), dy.loc[g].dropna()
                    rows.append(dict(regime=regime, cohort=name, split=sname, group=gname, n=len(a),
                                     ret_fall=int((a < 0).sum()), ret_med=a.median(),
                                     rel_rise=int((b > 0).sum()), rel_fall=int((b < 0).sum()),
                                     rel_med=b.median(), mice=" ".join(g)))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220, "display.max_columns", 20)
    out = stats_dir()
    ret, rel = pivots()
    hi, lo = split_by_relevance(rel)
    print("better:", hi, "\nworse: ", lo)
    t = group_tests(ret, rel, hi, lo)
    t.to_csv(out / "better_worse_tests.csv", index=False)
    print(t.round(4).to_string(index=False))
    s = joint_states()
    s.to_csv(out / "better_worse_states.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
