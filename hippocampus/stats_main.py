"""Cohort statistics of the hippocampus section on the 61-session cohort (and 64 / 59 for
the sensitivity analysis), from the shipped per-session tables.

  1. Per-mouse slope tests of both axes, both regimes, cohorts all64 / drop3 / drop5
     (main text: relevance 11/16 p=.037, lambda=0 13/16 p=.019, input 10/16 p=.047;
     Supplementary "Aggregate session-inclusion sensitivity").
  2. Linear mixed-effects model of relevance on day with behavioural covariates and a
     per-mouse random intercept (Supplementary "Relevance-axis day-coefficient ...").
  3. Habituation: cohort means of the behavioural covariates per day.
  4. Coverage and decoder tables per day and the per-mouse cross-check of decoder
     accuracy against corrected relevance (Supplementary "Hippocampus bias correction
     mechanism"), and the day-1 / day-4 MI trajectory.

Inputs (results/hippocampus/): ib_plane_lambda1.csv, ib_plane_lambda0.csv,
behavior.csv, decoder_accuracy_lambda1.csv.
Outputs (results/hippocampus/stats/): slope_tests.csv, lmm_relevance.csv,
habituation.csv, coverage_decoder_tables.csv.
Usage:  python hippocampus/stats_main.py
"""
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from hippo_common import per_mouse_slopes, stats_dir, table_path
from scipy import stats
from sessions import cohort

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def load_lmm_table():
    ip = pd.read_csv(table_path("ib_plane_lambda1.csv"))
    beh = pd.read_csv(table_path("behavior.csv"))
    df = ip.merge(beh, on=["mouse", "day"], how="inner")
    return df.rename(columns={"I_zy_bin_corr": "I_corr", "coverage_entropy_y_bin": "H_Y"})


LMM_SPECS = [
    ("day_only", "I_corr ~ day"),
    ("full", "I_corr ~ day + H_Y + mean_speed + locomotion_fraction + rest_freeze_fraction"
             " + rear_fraction + object_fraction + distance_traveled"),
    ("day_plus_HY", "I_corr ~ day + H_Y"),
    # H(Y_bin) of the test frames instead of the whole-session coverage entropy.
    ("day_plus_HYtest", "I_corr ~ day + H_Y_bin"),
]


def fit_lmm(df, label):
    rows = []
    for model, formula in LMM_SPECS:
        md = smf.mixedlm(formula, data=df, groups=df["mouse"]).fit()
        for name in md.params.index:
            if "Var" in name or "Group" in name:
                continue
            rows.append({"cohort": label, "n_sessions": len(df), "n_mice": df.mouse.nunique(),
                         "model": model, "term": name, "estimate": float(md.params[name]),
                         "se": float(md.bse[name]), "z": float(md.tvalues[name]),
                         "p": float(md.pvalues[name])})
    return rows


def slope_row(slopes, label, axis, regime):
    w = lambda alt: float(stats.wilcoxon(slopes, alternative=alt).pvalue)
    return {"cohort": label, "regime": regime, "axis": axis, "n_mice": len(slopes),
            "n_positive": int((slopes > 0).sum()), "n_negative": int((slopes < 0).sum()),
            "mean_slope": float(slopes.mean()), "p_one_sided_greater": w("greater"),
            "p_one_sided_less": w("less"), "p_two_sided": w("two-sided")}


def main():
    out = stats_dir()

    # ---- 1. slope tests
    sens = []
    for regime in ("lambda1", "lambda0"):
        ip = pd.read_csv(table_path(f"ib_plane_{regime}.csv"))
        for c in ("all64", "drop3", "drop5"):
            d = cohort(ip, c)
            for axis, col in (("relevance", "I_zy_bin_corr"), ("input", "I_xz_corr")):
                r = slope_row(per_mouse_slopes(d, col).values, c, axis, regime)
                m = d.groupby("day")[col].mean()
                r.update({"day1_mean": m[1], "day4_mean": m[4]})
                sens.append(r)
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "slope_tests.csv", index=False)
    print("[slopes] per-mouse OLS slope over days, Wilcoxon over 16 mice")
    print(sens.round(4).to_string(index=False))

    # ---- 2. mixed-effects model
    df = load_lmm_table()
    rows = []
    for c in ("all64", "drop3"):
        rows += fit_lmm(cohort(df, c), c)
    lmm = pd.DataFrame(rows)
    lmm.to_csv(out / "lmm_relevance.csv", index=False)
    print("\n[LMM] day coefficient:")
    print(lmm[lmm.term == "day"].round(4).to_string(index=False))
    print("[LMM] full model, drop3:")
    print(lmm[(lmm.cohort == "drop3") & (lmm.model == "full")].round(4).to_string(index=False))

    # ---- 3. habituation
    hab = []
    for c in ("all64", "drop3"):
        d = cohort(df, c)
        for col in ("mean_speed", "distance_traveled", "rest_freeze_fraction",
                    "locomotion_fraction", "H_Y", "H_Y_bin"):
            m = d.groupby("day")[col].mean()
            hab.append({"cohort": c, "variable": col, "day1": m[1], "day2": m[2], "day3": m[3],
                        "day4": m[4], "delta_4_1": m[4] - m[1],
                        "rel_change_pct": 100 * (m[4] - m[1]) / m[1],
                        "n_day1": int((d.day == 1).sum()), "n_day4": int((d.day == 4).sum())})
    hab = pd.DataFrame(hab)
    hab.to_csv(out / "habituation.csv", index=False)
    print("\n[habituation]")
    print(hab.round(3).to_string(index=False))

    # ---- 4. coverage / decoder tables
    dec = pd.read_csv(table_path("decoder_accuracy_lambda1.csv"))
    tab = []
    for c in ("all64", "drop3"):
        d, dd = cohort(df, c), cohort(dec, c)
        for day in (1, 2, 3, 4):
            s, t = d[d.day == day], dd[dd.day == day]
            tab.append({"cohort": c, "day": day, "n": len(s), "H_Y_bin_test": s.H_Y_bin.mean(),
                        "H_Y_coverage_session": s.H_Y.mean(),
                        "distance_m": s.distance_traveled.mean(),
                        "acc_logreg": t.acc_logreg.mean(), "acc_chance": t.acc_chance.mean(),
                        "acc_minus_chance": (t.acc_logreg - t.acc_chance).mean(),
                        "I_raw": s.I_zy_bin_real.mean(), "I_shuf": s.I_zy_bin_shuf_mean.mean(),
                        "I_corr": s.I_corr.mean()})
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "coverage_decoder_tables.csv", index=False)
    print("\n[coverage / decoder / MI per day]")
    print(tab.round(3).to_string(index=False))
    t = tab[tab.cohort == "drop3"].set_index("day")
    print("[MI / H(Y_bin), drop3] I_raw/H day1 {:.2f} day4 {:.2f}; I_shuf/H day1 {:.2f} day4 {:.2f}".format(
        t.I_raw[1] / t.H_Y_bin_test[1], t.I_raw[4] / t.H_Y_bin_test[4],
        t.I_shuf[1] / t.H_Y_bin_test[1], t.I_shuf[4] / t.H_Y_bin_test[4]))

    for c in ("all64", "drop3"):
        dd = cohort(dec, c)
        s = []
        for _, g in dd.groupby("mouse"):
            s.append({"acc_mlp_above": stats.linregress(g.day, g.acc_mlp - g.acc_chance).slope,
                      "acc_raw": stats.linregress(g.day, g.acc_logreg).slope,
                      "I": stats.linregress(g.day, g.I_zy_bin_corr).slope})
        s = pd.DataFrame(s)
        r1 = stats.spearmanr(s.acc_mlp_above, s.I)
        r2 = stats.spearmanr(s.acc_raw, s.I)
        print(f"[decoder cross-check slopes] {c}: above-chance MLP accuracy vs corrected I "
              f"Spearman r={r1.correlation:+.3f} p={r1.pvalue:.4f} n={len(s)}; "
              f"raw logreg accuracy r={r2.correlation:+.3f}")
    print(f"\n[OK] tables in {out}")


if __name__ == "__main__":
    main()
