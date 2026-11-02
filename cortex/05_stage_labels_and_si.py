"""Session table with training-stage labels, naive/trained endpoints, SI numbers.

Reads the cached per-session results (no autoencoder training):
  ib_goltstein_<mouse>_kf5.npz       R2_zx (floored per fold), R2_zx_raw, I_ce_zy,
                                     decoder and behavioural accuracy
  validation_per_session.csv         shuffle-null mean, blocked-CV reading
  validation_deltas.csv              per-pair changes of every variant
  xaxis_shuffle_per_session.csv      input-axis shuffle null

Writes to results/cortex/
  cortex_sessions_with_stage.csv     90 sessions, I_corr = I_ce - shuffle mean,
                                     stage label
  cortex_pairs_endpoints.csv         21 (mouse, area) pairs: earliest K = 12
                                     (naive, baseline 3) and latest K = 10
                                     (trained) session side by side, suffixes _n/_t
and prints the numbers of the cortex robustness section of the SI.

Stage labels follow the rule of the dataset authors' analysis code
(CArec.timepoint in github.com/pgoltstein/category-learning-visual-areas):
the task stage is set by the number of stimuli and the date relative to the
per-mouse learned dates below. Here K is counted from the valid trials, so a
short session can get a stage different from the one in the file; that
happens once (F04 AL 20160212, 27 valid trials, labelled K = 10).
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from ib.paths import results_dir

MICE = ['F02', 'F03', 'F04', '21a', '21b', 'K01', 'K02', 'K03', 'K04', 'K06', 'K07', 'K08']
PROTnl = [315, 314, 304, 523, 516, 520, 608, 602, 607, 602, 610, 625]
CATnl = [421, 419, 423, 605, 528, 606, 704, 628, 615, 629, 705, 728]

# Goltstein et al. (2021) clusters; LM and LI join the semantic-leaning
# cluster by their ventral-stream membership.
C2X = {"PM", "P", "POR", "LM", "LI"}
C1 = {"AL", "RL"}


def stage(mouse, date, K):
    i = MICE.index(mouse)
    md = int(str(date)[-3:])
    if K == 12:
        return "Baseline Task"
    if K == 2:
        return "Not-learned Prototype Task" if md <= PROTnl[i] else "Learned Prototype Task"
    if K == 10:
        return "Not-learned Category Task" if md <= CATnl[i] else "Learned Category Task"
    return "?"


def grp(a):
    return "V1" if a == "V1" else ("semantic ex-V1" if a in C2X else "feature (AL,RL)")


def session_table(res):
    rows = []
    for f in sorted(res.glob("ib_goltstein_*_kf5.npz")):
        d = np.load(f, allow_pickle=True)
        for r in d["results_post_stim"]:
            if not isinstance(r, dict):
                continue
            pr = r.get("per_run")
            has_folds = (pr is not None and len(pr) and isinstance(pr[0], dict)
                         and "R2_zx_raw" in pr[0])
            fold_raw = [float(x["R2_zx_raw"]) for x in pr] if has_folds else []
            rows.append(dict(
                mouse=r["mouse_id"], area=r["area"], date=str(r["date"]), K=int(r["K"]),
                n_neurons=int(r["n_neurons"]), n_trials=int(r["n_trials"]),
                R2_clip=float(r["R2_zx"]), R2_raw=float(r["R2_zx_raw"]),
                R2_clip_after_mean=(max(0.0, float(r["R2_zx_raw"]))
                                    if np.isfinite(r["R2_zx_raw"]) else np.nan),
                n_folds_neg=int(np.sum(np.array(fold_raw) < 0)) if fold_raw else -1,
                n_folds=len(fold_raw), I_ce=float(r["I_ce_zy"]),
                acc_dec=float(r.get("acc_logreg_y", np.nan)),
                beh=float(r.get("behavior_acc", np.nan))))
    S = pd.DataFrame(rows)
    pers = pd.read_csv(res / "validation_per_session.csv")
    pers["date"] = pers.date.astype(str)
    S = S.merge(pers[["mouse", "area", "date", "K", "ice_shuf_mean", "ice_blocked",
                      "ice_baseline"]], on=["mouse", "area", "date", "K"], how="left")
    S["I_corr"] = S.I_ce - S.ice_shuf_mean
    S["stage"] = [stage(m, d, k) for m, d, k in zip(S.mouse, S.date, S.K)]
    return S.sort_values(["mouse", "area", "date"])


def summ(x, name, alt="two-sided"):
    x = np.asarray(x, float)
    w = stats.wilcoxon(x, alternative=alt)
    print(f"  {name}: mean {x.mean():+.4f}, median {np.median(x):+.4f}, "
          f"up {int((x > 0).sum())}/{len(x)}, down {int((x < 0).sum())}, "
          f"zero {int((x == 0).sum())}, Wilcoxon {alt} p={w.pvalue:.4f}")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")

    S = session_table(res)
    S.to_csv(res / "cortex_sessions_with_stage.csv", index=False)
    print("=== Sessions and stage labels ===")
    print("sessions:", len(S), " mice:", S.mouse.nunique(),
          " pairs:", S.groupby(["mouse", "area"]).ngroups, " areas:", sorted(S.area.unique()))
    print(pd.crosstab(S.stage, S.K).to_string())
    early10 = S[(S.K == 10) & (S.date.str[-3:].astype(int) < 300)]
    print("K=10 sessions dated in the baseline period:",
          early10[["mouse", "area", "date", "n_trials"]].to_dict("records"))

    # Endpoints as in Figure 3: earliest K = 12 (naive) and latest K = 10 (trained)
    nv = S[S.K == 12].sort_values("date").groupby(["mouse", "area"]).first()
    tr = S[S.K == 10].sort_values("date").groupby(["mouse", "area"]).last()
    T = nv.join(tr, lsuffix="_n", rsuffix="_t", how="inner").reset_index()
    print("\n=== Endpoints ===")
    print("pairs:", len(T), "; naive stage:", T.stage_n.value_counts().to_dict(),
          "; trained stage:", T.stage_t.value_counts().to_dict())

    T["dR2_clip"] = T.R2_clip_t - T.R2_clip_n
    T["dR2_raw"] = T.R2_raw_t - T.R2_raw_n
    T["dR2_clip_after_mean"] = T.R2_clip_after_mean_t - T.R2_clip_after_mean_n
    T["group"] = T.area.map(grp)
    T["dI_fig"] = T.I_corr_t - T.I_corr_n
    T.to_csv(res / "cortex_pairs_endpoints.csv", index=False)

    print("\n=== Input axis R^2(Z -> X_phys), naive -> trained, 21 pairs (SI) ===")
    summ(T.dR2_clip, "dR2 (per-fold floor at 0, as reported)")
    summ(T.dR2_clip_after_mean, "dR2 (floor after averaging folds)")

    print("\n=== Cluster centroids (Figure 3) ===")
    cen = T.groupby("group").agg(n=("dR2_clip", "size"), dR2=("dR2_clip", "mean"),
                                 dI=("dI_fig", "mean"), R2_n=("R2_clip_n", "mean"),
                                 R2_t=("R2_clip_t", "mean"))
    print(cen.round(4).to_string())
    fc = T[T.group == "feature (AL,RL)"].dR2_clip
    print(f"feature cluster dR2: {int((fc < 0).sum())}/5 negative, "
          f"Wilcoxon two-sided p={stats.wilcoxon(fc).pvalue:.4f}")

    vd = pd.read_csv(res / "validation_deltas.csv")
    vd["dI_text"] = vd.delta_baseline - vd.delta_shuf
    M = T.merge(vd[["mouse", "area", "dI_text"]], on=["mouse", "area"])
    print(f"per-pair dI from validation_deltas vs from the session table: max |diff| = "
          f"{np.abs(M.dI_text - M.dI_fig).max():.4f}")

    print("\n=== Population sizes ===")
    print(f"  all {len(S)} sessions: {S.n_neurons.min()}-{S.n_neurons.max()} neurons "
          f"(median {S.n_neurons.median():.0f})")
    E = pd.concat([T.n_neurons_n, T.n_neurons_t])
    print(f"  {len(E)} endpoint sessions: {E.min()}-{E.max()} (median {E.median():.0f})")
    print(f"  trials per session: {S.n_trials.min()}-{S.n_trials.max()} "
          f"(median {S.n_trials.median():.0f})")

    print("\n=== R^2 flooring ===")
    ok = S[S.K != 2]
    print(f"  sessions with R^2 defined: {len(ok)}; mean-over-folds raw R^2 < 0 in "
          f"{int((ok.R2_raw < 0).sum())}; sessions with >=1 negative fold: "
          f"{int((ok.n_folds_neg > 0).sum())}")

    print("\n=== Shift-shuffle null on the relevance axis (SI) ===")
    x = vd.delta_shuf.values
    print(f"  dI_shuffle mean {x.mean():+.4f}, median {np.median(x):+.4f}, "
          f"negative {int((x < 0).sum())}/{len(x)}, "
          f"two-sided Wilcoxon p={stats.wilcoxon(x).pvalue:.4g}")
    print(f"  corrected dI mean {vd.dI_text.mean():+.4f}")

    print("\n=== Blocked-CV agreement (SI) ===")
    sb = S.dropna(subset=["ice_blocked"])
    print(f"  per-session I, random vs blocked folds, n={len(sb)}: "
          f"r={stats.pearsonr(sb.ice_baseline, sb.ice_blocked)[0]:.3f}")
    print(f"  per-pair dI, random vs blocked, n={len(vd)}: "
          f"r={stats.pearsonr(vd.delta_baseline, vd.delta_blocked)[0]:.3f}")

    print("\n=== Input-axis shuffle null (SI) ===")
    xs = pd.read_csv(res / "xaxis_shuffle_per_session.csv")
    print(f"  sessions: {len(xs)}; R2_shuf median {xs.R2_shuf_mean.median():.4f}, "
          f"range [{xs.R2_shuf_mean.min():.4f}, {xs.R2_shuf_mean.max():.4f}]")


if __name__ == "__main__":
    main()
