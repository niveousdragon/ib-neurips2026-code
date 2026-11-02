"""Per-component slopes of I(v; Z) (main text input-axis paragraph; Supplementary
"Per-component slopes"), both regimes, cohorts all64 / drop3 / drop5.

Per mouse with >= 3 retained days: OLS slope of the bias-corrected I(v; Z) on day;
cohort: mean slope, mice declining, one-sided Wilcoxon (less and greater).

Inputs: results/hippocampus/per_component_mi_lambda{0,1}.csv
Output: results/hippocampus/stats/per_component_slopes.csv
Usage:  python hippocampus/stats_per_component.py
"""
import pandas as pd
from hippo_common import per_mouse_slopes, stats_dir, table_path
from scipy.stats import wilcoxon
from sessions import cohort

VARS = ["speed", "omega", "head_direction", "d_obj1", "d_obj2", "d_obj3", "d_obj4",
        "time_since_start"]


def summarize(df, regime):
    out = []
    for c in ("all64", "drop3", "drop5"):
        d = cohort(df, c)
        for v in VARS:
            col = f"I_{v}_corr"
            s = per_mouse_slopes(d, col, min_days=3).values
            out.append({"regime": regime, "cohort": c, "n_sessions": len(d), "variable": v,
                        "day1_mean_I": d[d.day == 1][col].mean(), "n_mice": len(s),
                        "mean_slope": s.mean(), "n_declining": int((s < 0).sum()),
                        "p_less": wilcoxon(s, alternative="less").pvalue,
                        "p_greater": wilcoxon(s, alternative="greater").pvalue})
    return pd.DataFrame(out)


def main():
    res = pd.concat([summarize(pd.read_csv(table_path(f"per_component_mi_{r}.csv")), r)
                     for r in ("lambda1", "lambda0")], ignore_index=True)
    out = stats_dir() / "per_component_slopes.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    for r in ("lambda1", "lambda0"):
        d = res[(res.regime == r) & (res.cohort == "drop3")]
        print(f"[{r} drop3] smallest one-sided p for a rise: "
              f"{d.p_greater.min():.3f} ({d.loc[d.p_greater.idxmin(), 'variable']})")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
