"""Per-area and per-cluster changes on both axes (main text and SI cluster table).

One convention throughout:
  dI   bias-corrected relevance gain, delta_baseline - delta_shuf
       (validation_deltas.csv)
  dR2  R^2(latest K = 10 session) - R^2(earliest K = 12 session), as in Figure 3

Cluster definitions: STRICT takes the semantic-leaning cluster of the text of
Goltstein et al. (2021) {V1, PM, P, POR}; STREAM adds the ventral-stream areas
LM and LI. The feature-leaning cluster is {AL, RL} in both (AM is not in the
cohort). Runs from the shipped caches.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from ib.paths import results_dir

CLUSTER_DEFS = {
    "STRICT C2 (V1+PM+P+POR)": {"V1", "PM", "P", "POR"},
    "STREAM C2 (V1+PM+P+POR+LM+LI)": {"V1", "PM", "P", "POR", "LM", "LI"},
    "C1 (AL+RL)": {"AL", "RL"},
    "STREAM ex-V1 (PM+P+POR+LM+LI)": {"PM", "P", "POR", "LM", "LI"},
    "V1 only": {"V1"},
}


def load_per_session_R2(res) -> pd.DataFrame:
    rows = []
    for npz_path in sorted(res.glob("ib_goltstein_*_kf5.npz")):
        d = np.load(npz_path, allow_pickle=True)
        for rec in d["results_post_stim"]:
            if not isinstance(rec, dict):
                continue
            rows.append({"mouse": rec["mouse_id"], "area": rec["area"],
                         "date": str(rec["date"]), "K": int(rec["K"]),
                         "R2_zx": float(rec["R2_zx"])})
    return pd.DataFrame(rows)


def per_pair_dR2(sess: pd.DataFrame) -> pd.DataFrame:
    sub = sess.copy()
    sub["date_int"] = pd.to_numeric(sub.date, errors="coerce")
    naive = (sub[sub.K == 12].sort_values("date_int")
             .groupby(["mouse", "area"], as_index=False).first()
             [["mouse", "area", "R2_zx"]].rename(columns={"R2_zx": "R2_naive"}))
    trained = (sub[sub.K == 10].sort_values("date_int")
               .groupby(["mouse", "area"], as_index=False).last()
               [["mouse", "area", "R2_zx"]].rename(columns={"R2_zx": "R2_trained"}))
    p = naive.merge(trained, on=["mouse", "area"], how="inner")
    p["dR2"] = p.R2_trained - p.R2_naive
    return p


def cluster_stats(values: np.ndarray) -> dict:
    n = len(values)
    out = {"n": n, "n_pos": int((values > 0).sum()), "mean": float(values.mean())}
    for key, alt in (("p_two", "two-sided"), ("p_greater", "greater"), ("p_less", "less")):
        try:
            out[key] = float(stats.wilcoxon(values, alternative=alt).pvalue) if n >= 2 else np.nan
        except Exception:  # scipy refuses all-zero or too-short inputs
            out[key] = float("nan")
    return out


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")
    pairs_R2 = per_pair_dR2(load_per_session_R2(res))
    vd = pd.read_csv(res / "validation_deltas.csv")
    pairs_I = vd[["mouse", "area"]].copy()
    pairs_I["dI"] = vd.delta_baseline - vd.delta_shuf

    print("Per-area means:")
    pa = (pairs_R2.merge(pairs_I, on=["mouse", "area"]).groupby("area")
          .agg(n=("mouse", "count"), mean_dR2=("dR2", "mean"), mean_dI=("dI", "mean"))
          .reset_index())
    print(pa.round(4).to_string(index=False))

    print(f"\n{'cluster':<32} {'axis':<4} {'n':>3} {'n_pos':>5} {'mean':>8} "
          f"{'p two':>8} {'p >0':>8} {'p <0':>8}")
    for label, areas in CLUSTER_DEFS.items():
        for axis, df, col in (("dR2", pairs_R2, "dR2"), ("dI", pairs_I, "dI")):
            s = cluster_stats(df[df.area.isin(areas)][col].values)
            print(f"{label:<32} {axis:<4} {s['n']:>3} {s['n_pos']:>5} {s['mean']:>+8.4f} "
                  f"{s['p_two']:>8.4f} {s['p_greater']:>8.4f} {s['p_less']:>8.4f}")


if __name__ == "__main__":
    main()
