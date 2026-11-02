"""Figure 3: cortex information-plane trajectories and decoder-behaviour gap.

Left: per-(mouse, area) arrow from the naive session (earliest K = 12,
baseline 3) to the trained session (latest K = 10) on (R^2(Z -> X_phys),
I(Z;Y) - shuffle mean); thick arrows are cluster centroids. The input axis is
plotted without subtraction although its label says bias-corrected: its
shuffle null is ~0 in every session (xaxis_shuffle_per_session.csv, at most
0.009), so subtracting it would not move the points visibly.
Right: decoder accuracy from Z against behavioural accuracy, averaged over the
category-stage (K = 10) sessions of each pair, without F04 AL 20160212 (a
baseline-period session with K = 10 among its 27 valid trials; it is not an
endpoint of the left panel).

Reads results/cortex (ib_goltstein_<mouse>_kf5.npz, validation_per_session.csv);
writes figures_dir()/fig3_cortex.png.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ib.paths import figures_dir, results_dir

EXCLUDE_K10 = {("F04", "AL", "20160212")}

# Goltstein et al. (2021) Fig. 4f stream-based clusters: ventral {LM, LI, P,
# POR} with V1 and PM in cluster 2 (learning-recruited); dorsal {AL, RL, AM}
# in cluster 1 (AM is not in the cohort).
CLUSTER2 = {"V1", "PM", "P", "POR", "LM", "LI"}
CLUSTER1 = {"AL", "RL", "AM"}

COL_V1 = "#9b1d20"
COL_C2 = "#e6843b"
COL_C1 = "#888888"
COL_DOT = "black"

TITLE_FS = 14
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 11
ANNOT_FS = 11


def colour_of(area: str) -> str:
    if area == "V1":
        return COL_V1
    if area in CLUSTER2:
        return COL_C2
    if area in CLUSTER1:
        return COL_C1
    return COL_DOT


def alpha_of(area: str) -> float:
    # individual arrows stay visible but under the centroids
    if area == "V1":
        return 0.65
    if area in CLUSTER2:
        return 0.55
    return 0.45


def load_per_session(res) -> pd.DataFrame:
    rows = []
    for npz_path in sorted(res.glob("ib_goltstein_*_kf5.npz")):
        d = np.load(npz_path, allow_pickle=True)
        for rec in d["results_post_stim"]:
            if not isinstance(rec, dict):
                continue
            rows.append({
                "mouse": rec["mouse_id"],
                "area": rec["area"],
                "date": rec["date"],
                "K": int(rec["K"]),
                "R2_zx": float(rec["R2_zx"]),
                "I_ce_zy": float(rec["I_ce_zy"]),
                "acc_logreg_y": float(rec.get("acc_logreg_y", np.nan)),
                "behavior_acc": float(rec.get("behavior_acc", np.nan)),
            })
    return pd.DataFrame(rows)


def panel_trajectories(ax, traj: pd.DataFrame):
    # cluster 1 drawn first, V1 last so it lands on top
    for tier_areas in (CLUSTER1, CLUSTER2 - {"V1"}, {"V1"}):
        sub = traj[traj.area.isin(tier_areas)]
        for _, row in sub.iterrows():
            col = colour_of(row.area)
            alpha = alpha_of(row.area)
            lw = 1.0
            mscale = 14 if row.area == "V1" else 9
            ax.annotate("", xy=(row.R2_trained, row.I_trained),
                        xytext=(row.R2_naive, row.I_naive),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                        alpha=alpha, mutation_scale=mscale))
            ax.plot([row.R2_naive], [row.I_naive], "o",
                    ms=4 if row.area == "V1" else 2.8,
                    mfc="white", mec=col, mew=lw * 0.55, alpha=alpha)

    def grp_label(area):
        if area == "V1":
            return "V1"
        if area in CLUSTER2:
            return "C2"
        if area in CLUSTER1:
            return "C1"
        return None

    traj = traj.copy()
    traj["grp"] = traj.area.apply(grp_label)
    centroids = (traj.groupby("grp")[["R2_naive", "R2_trained", "I_naive", "I_trained"]]
                 .mean().reset_index())
    grp_col = {"V1": COL_V1, "C2": COL_C2, "C1": COL_C1}
    for _, c in centroids.iterrows():
        col = grp_col[c.grp]
        ax.annotate("", xy=(c.R2_trained, c.I_trained), xytext=(c.R2_naive, c.I_naive),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=4.5, alpha=1.0,
                                    mutation_scale=22, joinstyle="round"),
                    zorder=10)
        ax.plot([c.R2_naive], [c.I_naive], "o", ms=9, mfc=col, mec="black", mew=1.0, zorder=11)

    # autoscale ignores annotate arrows, so limits come from the endpoints
    xs = pd.concat([traj.R2_naive, traj.R2_trained]).values
    ys = pd.concat([traj.I_naive, traj.I_trained]).values
    xpad = 0.05 * (xs.max() - xs.min())
    ypad = 0.05 * (ys.max() - ys.min())
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad + 0.10)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    ax.axhline(0, c="0.5", lw=0.6, ls=":")
    ax.axvline(0, c="0.5", lw=0.6, ls=":")
    ax.set_xlabel(r"$R^{2}(Z \to X_{\mathrm{phys}})$, bias-corrected", fontsize=LABEL_FS)
    ax.set_ylabel(r"$I(Z;Y)$, bias-corrected (bits)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    legend_handles = [
        plt.Line2D([0], [0], color=COL_V1, lw=2.6, marker=">", ms=9,
                   markerfacecolor=COL_V1, markeredgecolor=COL_V1, linestyle="-"),
        plt.Line2D([0], [0], color=COL_C2, lw=1.4, marker=">", ms=7,
                   markerfacecolor=COL_C2, markeredgecolor=COL_C2, linestyle="-"),
        plt.Line2D([0], [0], color=COL_C1, lw=1.4, marker=">", ms=7,
                   markerfacecolor=COL_C1, markeredgecolor=COL_C1, linestyle="-", alpha=0.55),
    ]
    ax.legend(legend_handles, ["V1", "cluster 2 ex-V1", "cluster 1"],
              loc="lower right", frameon=False, fontsize=LEGEND_FS,
              handlelength=1.8, borderpad=0.4, labelspacing=0.4)
    ax.set_title(r"Per-(mouse, area) IB trajectory: naive $\to$ trained",
                 fontsize=TITLE_FS, loc="left")
    ax.grid(alpha=0.22, lw=0.5)


def panel_decoder_vs_behaviour(ax, decoder_df: pd.DataFrame):
    ax.plot([0.4, 1.0], [0.4, 1.0], c="0.4", lw=1.0, ls="--", zorder=1,
            label=r"identity ($y = x$)")
    ax.axhline(0.5, c="0.7", lw=0.6, ls=":", zorder=0)
    ax.axvline(0.5, c="0.7", lw=0.6, ls=":", zorder=0)

    for areas, label in ((CLUSTER1, "cluster 1"), (CLUSTER2 - {"V1"}, "cluster 2 ex-V1"),
                         ({"V1"}, "V1")):
        sub = decoder_df[decoder_df.area.isin(areas)]
        if sub.empty:
            continue
        is_v1 = areas == {"V1"}
        col = COL_V1 if is_v1 else colour_of(next(iter(areas)))
        ax.scatter(sub.beh, sub.dec, s=72 if is_v1 else 50, color=col,
                   edgecolor="black", lw=0.6, alpha=0.95 if is_v1 else 0.75,
                   zorder=4 if is_v1 else 3, label=label)

    gap = (decoder_df.dec - decoder_df.beh).mean() * 100
    ax.text(0.04, 0.96, f"mean gap = +{gap:.1f} pp", transform=ax.transAxes,
            ha="left", va="top", fontsize=ANNOT_FS,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", lw=0.7))
    ax.set_xlabel("behavioural accuracy (category stage)", fontsize=LABEL_FS)
    ax.set_ylabel("decoder accuracy from cortical $Z$\n(LogReg, category stage)",
                  fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(0.42, 1.02)
    ax.set_ylim(0.42, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Decoder vs. behavioural accuracy", fontsize=TITLE_FS, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=LEGEND_FS,
              handlelength=1.0, borderpad=0.4, labelspacing=0.3)
    ax.grid(alpha=0.22, lw=0.5)


def build_traj(sess: pd.DataFrame) -> pd.DataFrame:
    """Earliest K = 12 (naive) and latest K = 10 (trained) session per pair."""
    sub = sess.copy()
    sub["date_int"] = pd.to_numeric(sub.date, errors="coerce")
    naive = (sub[sub.K == 12].sort_values("date_int")
             .groupby(["mouse", "area"], as_index=False).first()
             [["mouse", "area", "R2_corr", "I_corr"]]
             .rename(columns={"R2_corr": "R2_naive", "I_corr": "I_naive"}))
    trained = (sub[sub.K == 10].sort_values("date_int")
               .groupby(["mouse", "area"], as_index=False).last()
               [["mouse", "area", "R2_corr", "I_corr"]]
               .rename(columns={"R2_corr": "R2_trained", "I_corr": "I_trained"}))
    traj = naive.merge(trained, on=["mouse", "area"], how="inner")
    return traj.dropna(subset=["R2_naive", "R2_trained", "I_naive", "I_trained"])


def build_decoder_df(sess: pd.DataFrame) -> pd.DataFrame:
    """Decoder and behavioural accuracy per pair, mean over its K = 10 sessions."""
    drop = [(m, a, str(d)) in EXCLUDE_K10 for m, a, d in zip(sess.mouse, sess.area, sess.date)]
    sub = sess[~np.array(drop, dtype=bool)]
    trained = (sub[sub.K == 10].groupby(["mouse", "area"], as_index=False)
               [["behavior_acc", "acc_logreg_y"]].mean())
    trained = trained.rename(columns={"behavior_acc": "beh", "acc_logreg_y": "dec"})
    return trained.dropna(subset=["beh", "dec"])


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = results_dir("cortex")
    sess = load_per_session(res)
    if sess.empty:
        raise SystemExit(f"no ib_goltstein_*_kf5.npz in {res}")
    pers = pd.read_csv(res / "validation_per_session.csv")
    pers["date"] = pers["date"].astype(str)
    sess["date"] = sess["date"].astype(str)
    sess = sess.merge(pers[["mouse", "area", "date", "K", "ice_shuf_mean", "ice_shuf_max"]],
                      on=["mouse", "area", "date", "K"], how="left")
    sess["I_corr"] = sess.I_ce_zy - sess.ice_shuf_mean.fillna(0)
    # The input-axis shuffle null is ~0 in every session, so R^2 enters as is.
    sess["R2_corr"] = sess.R2_zx

    traj = build_traj(sess)
    decoder_df = build_decoder_df(sess)
    print(f"trajectories: {len(traj)}; decoder pairs: {len(decoder_df)}")
    gap = (decoder_df.dec - decoder_df.beh).mean() * 100
    print(f"decoder-behaviour gap: mean = +{gap:.2f} pp (n={len(decoder_df)})")

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    panel_trajectories(axes[0], traj)
    panel_decoder_vs_behaviour(axes[1], decoder_df)
    fig.tight_layout()
    out = figures_dir() / "fig3_cortex.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
