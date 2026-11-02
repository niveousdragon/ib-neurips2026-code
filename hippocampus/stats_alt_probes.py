"""Alternative-probe paragraph and the KSG vs GCMI numbers (Supplementary
"Alternative probes of the same activity", "Decoder family for the relevance axis").

Per probe (AE lambda=1, AE lambda=0, PCA-4/8/16/30, Isomap-16) and axis: cohort mean,
day-1 / day-4 means, median per-mouse slope, mice in the predicted direction
(relevance up, retention down), one-sided and two-sided Wilcoxon; pairwise Spearman
agreement of per-session values. KSG vs GCMI: means over the 16 day-1 sessions.

Inputs (results/hippocampus/): alt_probes.csv, ib_plane_lambda{0,1}.csv, ksg_vs_gcmi.csv
Outputs (results/hippocampus/stats/): alt_probes_summary.csv, alt_probes.txt
Usage:  python hippocampus/stats_alt_probes.py
"""
import itertools

import pandas as pd
from hippo_common import per_mouse_slopes, stats_dir, table_path
from scipy import stats
from sessions import cohort

PROBES = ["ae1", "ae0", "pca4", "pca8", "pca16", "pca30", "iso16"]
NAMES = {"ae1": "AE l=1", "ae0": "AE l=0", "pca4": "PCA-4", "pca8": "PCA-8",
         "pca16": "PCA-16", "pca30": "PCA-30", "iso16": "Isomap-16"}


def probe_table():
    """Per-session values of every probe on both axes (columns rel_<probe>, ret_<probe>)."""
    df = pd.read_csv(table_path("alt_probes.csv"))
    for regime, tag in (("lambda1", "ae1"), ("lambda0", "ae0")):
        ae = pd.read_csv(table_path(f"ib_plane_{regime}.csv"))
        df = df.merge(ae[["mouse", "day", "I_zy_bin_corr", "I_xz_corr"]].rename(
            columns={"I_zy_bin_corr": f"rel_{tag}", "I_xz_corr": f"ret_{tag}"}), on=["mouse", "day"])
    return df


def summary(df):
    rows, L = [], []
    for name, label in (("drop3", "61"), ("all64", "64")):
        d = cohort(df, name)
        L.append(f"\n=== cohort {label}: {len(d)} sessions ===")
        for ax, pred in (("rel", "greater"), ("ret", "less")):
            for p in PROBES:
                c = f"{ax}_{p}"
                sl = per_mouse_slopes(d, c).dropna()
                n_dir = int((sl > 0).sum() if pred == "greater" else (sl < 0).sum())
                r = dict(cohort=label, axis=ax, probe=p, mean_value=d[c].mean(),
                         day1=d[d.day == 1][c].mean(), day4=d[d.day == 4][c].mean(),
                         median_slope=sl.median(), n_pred_dir=n_dir, n_mice=len(sl),
                         p_one=stats.wilcoxon(sl, alternative=pred).pvalue,
                         p_two=stats.wilcoxon(sl).pvalue)
                rows.append(r)
                L.append(f"{ax} {NAMES[p]:9s} mean {r['mean_value']:.3f} d1 {r['day1']:.3f} "
                         f"d4 {r['day4']:.3f} median slope {r['median_slope']:+.3f} "
                         f"dir {n_dir}/{len(sl)} p1 {r['p_one']:.3f} p2 {r['p_two']:.3f}")
        for ax in ("rel", "ret"):
            for group in (["ae1", "ae0", "pca16", "iso16"], ["pca4", "pca8", "pca16", "pca30"]):
                rs = [stats.spearmanr(d[f"{ax}_{a}"], d[f"{ax}_{b}"]).correlation
                      for a, b in itertools.combinations(group, 2)]
                L.append(f"{ax}: pairwise Spearman of session values {group}: {min(rs):.2f}-{max(rs):.2f}")
    return pd.DataFrame(rows), L


def ksg_summary():
    k = pd.read_csv(table_path("ksg_vs_gcmi.csv"))
    L = [f"\n=== KSG vs GCMI, I(X_ego;Z), {len(k)} day-1 sessions, test frames {k.n.min()}-{k.n.max()} ==="]
    for c in ("gcmi", "gcmi_perm", "ksg_k5", "ksg_k5_perm", "ksg_k23", "ksg_k23_perm"):
        L.append(f"{c:13s} mean {k[c].mean():.3f} median {k[c].median():.3f} "
                 f"min {k[c].min():.3f} max {k[c].max():.3f}")
    return L


def main():
    s, L = summary(probe_table())
    L += ksg_summary()
    out = stats_dir()
    s.to_csv(out / "alt_probes_summary.csv", index=False)
    (out / "alt_probes.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
