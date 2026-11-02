"""Collect the per-mouse control runs into two tables.

Reads results/cortex/ib_goltstein_validation_<mouse>.npz (02_run_validation.py)
and writes
  validation_per_session.csv  one row per session (90): I(Z;Y) for baseline,
                              lambda0, blocked CV, shuffle-null mean and max
  validation_deltas.csv       one row per (mouse, area) pair (21): change from
                              the earliest K = 12 session (naive, baseline 3)
                              to the latest K = 10 session (trained) for each
                              variant
and prints cohort summaries. Runs from the shipped caches.
"""
import argparse

import numpy as np
import pandas as pd
from goltstein import PAPER_MICE
from scipy.stats import wilcoxon

from ib.paths import results_dir


def load_all(res):
    rows = []
    for m in PAPER_MICE:
        f = res / f"ib_goltstein_validation_{m}.npz"
        if not f.exists():
            print(f"  MISSING: {f.name}")
            continue
        d = np.load(f, allow_pickle=True)
        for r in d["results"]:
            r = r.item() if hasattr(r, "item") else r
            rows.append({
                "mouse": str(r["mouse_id"]),
                "area": str(r["area"]),
                "date": str(r["date"]),
                "K": int(r["K"]),
                "n_trials": int(r["n_trials"]),
                "n_neurons": int(r["n_neurons"]),
                "behavior_acc": float(r["behavior_acc"]),
                "ice_baseline": float(r["ice_baseline"]),
                "ice_lambda0": float(r["ice_lambda0"]),
                "ice_blocked": float(r["ice_blocked"]),
                "ice_shuf_mean": float(r["ice_shuf_mean"]),
                "ice_shuf_max": float(r["ice_shuf_max"]),
            })
    return pd.DataFrame(rows)


def compute_deltas(df):
    """Per (mouse, area): latest K = 10 minus earliest K = 12, per variant."""
    rows = []
    for ma, sub in df.groupby(["mouse", "area"]):
        sub = sub.sort_values("date")
        k12 = sub[sub.K == 12]
        k10 = sub[sub.K == 10]
        if k12.empty or k10.empty:
            continue
        f12 = k12.iloc[0]
        l10 = k10.iloc[-1]
        rows.append({
            "mouse": ma[0], "area": ma[1],
            "delta_baseline": l10["ice_baseline"] - f12["ice_baseline"],
            "delta_lambda0": l10["ice_lambda0"] - f12["ice_lambda0"],
            "delta_blocked": l10["ice_blocked"] - f12["ice_blocked"],
            "delta_shuf": l10["ice_shuf_mean"] - f12["ice_shuf_mean"],
            "ice_baseline_last": l10["ice_baseline"],
            "ice_lambda0_last": l10["ice_lambda0"],
            "ice_blocked_last": l10["ice_blocked"],
            "ice_shuf_last": l10["ice_shuf_mean"],
            "ice_baseline_first": f12["ice_baseline"],
            "ice_lambda0_first": f12["ice_lambda0"],
            "ice_blocked_first": f12["ice_blocked"],
            "ice_shuf_first": f12["ice_shuf_mean"],
        })
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")
    df = load_all(res)
    if df.empty:
        print("No data loaded")
        return
    print(f"Loaded {len(df)} session-rows from {df.mouse.nunique()} mice")
    df.to_csv(res / "validation_per_session.csv", index=False)

    print("\nReal signal above the shift-shuffle null:")
    n_total = len(df)
    print(f"  baseline > shuf_max:   {int((df.ice_baseline > df.ice_shuf_max).sum())}/{n_total}")
    print(f"  baseline > shuf_mean:  {int((df.ice_baseline > df.ice_shuf_mean).sum())}/{n_total}")
    print(f"  lambda0  > shuf_mean:  {int((df.ice_lambda0 > df.ice_shuf_mean).sum())}/{n_total}")
    n_blk = int((~df.ice_blocked.isna()).sum())
    n_blk_above = int(((df.ice_blocked > df.ice_shuf_mean) & ~df.ice_blocked.isna()).sum())
    print(f"  blocked  > shuf_mean:  {n_blk_above}/{n_blk}")

    deltas = compute_deltas(df)
    deltas.to_csv(res / "validation_deltas.csv", index=False)

    dec = deltas.dropna(subset=["delta_blocked"])
    print(f"\nCohort change (latest K=10 - earliest K=12), n={len(dec)} pairs (raw, not corrected):")
    for col, label in [
        ("delta_baseline", "baseline (lambda=1, random 5-fold)"),
        ("delta_lambda0", "lambda=0 (reconstruction-only AE)"),
        ("delta_blocked", "blocked CV (lambda=1)"),
        ("delta_shuf", "shift-shuffle null"),
    ]:
        d = dec[col].values
        p = wilcoxon(d, alternative="greater").pvalue
        print(f"  {label:<38} mean={np.mean(d):+.3f} median={np.median(d):+.3f} "
              f"n_pos={int((d > 0).sum())}/{len(d)}  one-sided p={p:.4f}")

    sub = df.dropna(subset=["ice_blocked"])
    r = np.corrcoef(sub.ice_baseline, sub.ice_blocked)[0, 1]
    print(f"\nPearson r(random-fold, blocked) per session: {r:.3f} (n={len(sub)})")
    r = np.corrcoef(df.ice_baseline, df.ice_lambda0)[0, 1]
    print(f"Pearson r(lambda=1, lambda=0) per session: {r:.3f} (n={len(df)})")


if __name__ == "__main__":
    main()
