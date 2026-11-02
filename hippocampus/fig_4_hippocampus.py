"""Figure 4: hippocampal information plane across 4 days (61 sessions).

Left: cohort-mean information plane (bias-corrected), day markers with SEM bars and
the day 1 -> day 4 trajectory. Right: per-component bias-corrected I(X_j; Z) per day
for the six components of X_ego, mean with SEM bands.

Inputs: results/hippocampus/ib_plane_lambda1.csv, per_component_mi_lambda1.csv
Output: <figures>/fig4_hippocampus.png
Usage:  python hippocampus/fig_4_hippocampus.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hippo_common import table_path
from sessions import cohort

from ib.paths import figures_dir

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

TITLE_FS = 14
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 10

COMPONENTS = ["d_obj1", "d_obj2", "d_obj3", "d_obj4", "speed", "omega"]
COMPONENT_LABELS = {
    "d_obj1": r"$d_\mathrm{obj1}$", "d_obj2": r"$d_\mathrm{obj2}$",
    "d_obj3": r"$d_\mathrm{obj3}$", "d_obj4": r"$d_\mathrm{obj4}$",
    "speed": r"speed", "omega": r"$\omega$",
}
COMPONENT_COLORS = {
    "d_obj1": "#b30000", "d_obj2": "#e34a33", "d_obj3": "#fc8d59", "d_obj4": "#fdbb84",
    "speed": "#3182bd", "omega": "#6baed6",
}


def panel_ib_plane(ax, df):
    """Cohort means only; per-mouse trajectories are in the Supplementary."""
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=1, vmax=4)
    days = sorted(df.day.unique())
    x_col, y_col = "I_xz_corr", "I_zy_bin_corr"
    means = df.groupby("day")[[x_col, y_col]].mean().reindex(days)
    sems = df.groupby("day")[[x_col, y_col]].sem().reindex(days)
    ax.errorbar(means[x_col], means[y_col], xerr=sems[x_col], yerr=sems[y_col],
                fmt="none", ecolor="black", capsize=4, lw=1.0, zorder=5)
    for d in days:
        ax.plot(means.loc[d, x_col], means.loc[d, y_col], "o", ms=14,
                mfc=cmap(norm(d)), mec="black", mew=1.2, zorder=6)
    for d, d_next in zip(days[:-1], days[1:]):
        ax.annotate("", xy=(means.loc[d_next, x_col], means.loc[d_next, y_col]),
                    xytext=(means.loc[d, x_col], means.loc[d, y_col]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                    mutation_scale=14, zorder=7))
    # Zoom on the cohort-mean cluster, SEM included.
    x_lo = (means[x_col] - sems[x_col]).min()
    x_hi = (means[x_col] + sems[x_col]).max()
    y_lo = (means[y_col] - sems[y_col]).min()
    y_hi = (means[y_col] + sems[y_col]).max()
    ax.set_xlim(x_lo - 0.30 * (x_hi - x_lo), x_hi + 0.30 * (x_hi - x_lo))
    ax.set_ylim(y_lo - 0.30 * (y_hi - y_lo), y_hi + 0.30 * (y_hi - y_lo))
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$I(X_\mathrm{ego}; Z)$, bias-corrected (bits)", fontsize=LABEL_FS)
    ax.set_ylabel(r"$I(Z; Y_\mathrm{alloc})$, bias-corrected (bits)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title(r"Cohort information plane: day 1 $\to$ day 4 (16 mice)",
                 fontsize=TITLE_FS, loc="center")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(norm(d)),
                          markeredgecolor="black", markersize=10, label=f"day {d}") for d in days]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=LEGEND_FS,
              handletextpad=0.4, labelspacing=0.3)
    ax.grid(alpha=0.22, lw=0.5)


def panel_components(ax, df):
    days = sorted(df.day.unique())
    handles = []
    for comp in COMPONENTS:
        col = f"I_{comp}_corr"
        m = df.groupby("day")[col].mean().reindex(days)
        s = df.groupby("day")[col].sem().reindex(days)
        h, = ax.plot(days, m.values, "o-", color=COMPONENT_COLORS[comp], lw=1.8, ms=6,
                     label=COMPONENT_LABELS[comp], zorder=4)
        ax.fill_between(days, (m - s).values, (m + s).values, color=COMPONENT_COLORS[comp],
                        alpha=0.18, zorder=3)
        handles.append(h)
    ax.axhline(0, c="black", lw=0.4, ls=":")
    ax.set_xticks(days)
    ax.set_xlabel("day", fontsize=LABEL_FS)
    ax.set_ylabel(r"bias-corrected $I(X_j; Z)$ (bits)", fontsize=LABEL_FS)
    ax.set_ylim(-0.02, 0.55)
    ax.set_yticks(np.arange(0.0, 0.51, 0.10))
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title("Per-component bias-corrected MI vs day", fontsize=TITLE_FS, loc="center")
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=LEGEND_FS, ncol=2,
              handletextpad=0.4, labelspacing=0.3, columnspacing=0.8)
    ax.grid(alpha=0.22, lw=0.5)


def main():
    df_a = cohort(pd.read_csv(table_path("ib_plane_lambda1.csv")), "drop3")
    df_b = cohort(pd.read_csv(table_path("per_component_mi_lambda1.csv")), "drop3")
    print(f"left: {len(df_a)} sessions, {df_a.mouse.nunique()} mice; "
          f"right: {len(df_b)} sessions, {df_b.mouse.nunique()} mice")
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.22)
    panel_ib_plane(fig.add_subplot(gs[0, 0]), df_a)
    panel_components(fig.add_subplot(gs[0, 1]), df_b)
    for comp in COMPONENTS:
        g = df_b.groupby("day")[f"I_{comp}_corr"]
        print(f"{comp}: mean " + " / ".join(f"{v:.3f}" for v in g.mean())
              + " | sem " + " / ".join(f"{v:.3f}" for v in g.sem()))
    out = figures_dir() / "fig4_hippocampus.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
