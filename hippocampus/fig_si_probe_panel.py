"""Supplementary figure: alternative probes, hippocampus (61 sessions).

a. relevance per day (mean over sessions) per probe; b. retention per day;
c. mice (of 16) whose per-mouse slope goes in the predicted direction, with the
one-sided Wilcoxon p above the bars.

Inputs: as stats_alt_probes.py
Output: <figures>/si_probe_panel.png
Usage:  python hippocampus/fig_si_probe_panel.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sessions import cohort
from stats_alt_probes import PROBES, probe_table, summary

from ib.paths import figures_dir

NAMES = {"ae1": r"AE $\lambda$=1", "ae0": r"AE $\lambda$=0", "pca4": "PCA-4", "pca8": "PCA-8",
         "pca16": "PCA-16", "pca30": "PCA-30", "iso16": "Isomap-16"}


def main():
    df = probe_table()
    S, _ = summary(df)
    S = S[S.cohort == "61"]
    d61 = cohort(df, "drop3")
    # Built at the printed width (5.5 in) so font sizes are the printed sizes.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.45), layout="constrained",
                            gridspec_kw={"width_ratios": [1, 1, 1.25]})
    cols = plt.cm.tab10(np.arange(len(PROBES)))
    for j, (ax, lab) in enumerate((("rel", r"relevance $I(Z;Y_\mathrm{alloc})$ (bits)"),
                                   ("ret", r"retention $I(X_\mathrm{ego};Z)$ (bits)"))):
        for i, p in enumerate(PROBES):
            m = d61.groupby("day")[f"{ax}_{p}"].mean()
            axs[j].plot(m.index, m.values, "-o", color=cols[i], ms=2, lw=1.0, label=NAMES[p])
        axs[j].set_xlabel("day")
        axs[j].set_ylabel(lab)
        axs[j].set_xticks([1, 2, 3, 4])
    # Legend below panels a-b: inside panel a it would cover the PCA curves.
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower left", ncol=4,
               frameon=False, handlelength=1.4, columnspacing=1.0)
    x = np.arange(len(PROBES))
    w = 0.4
    for j, (ax, lab) in enumerate((("rel", "relevance up"), ("ret", "retention down"))):
        s_ = S[S.axis == ax].set_index("probe").loc[PROBES]
        axs[2].bar(x + (j - 0.5) * w, s_.n_pred_dir, w, color=("#1f77b4", "#ff7f0e")[j], label=lab)
        for xi, (n, pv) in enumerate(zip(s_.n_pred_dir, s_.p_one)):
            axs[2].text(xi + (j - 0.5) * w, n + 0.3, f"{pv:.3f}".lstrip("0"),
                        ha="center", va="bottom", fontsize=5, rotation=90)
    axs[2].axhline(8, color="k", lw=0.6, ls=":")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([NAMES[p] for p in PROBES], rotation=45, ha="right")
    axs[2].set_ylim(0, 20)
    axs[2].set_yticks([0, 4, 8, 12, 16])
    axs[2].set_ylabel("mice in predicted direction (of 16)")
    axs[2].legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.0,
                  borderaxespad=0.1)
    for a, letter in zip(axs, "abc"):
        a.text(-0.02, 1.0, letter, transform=a.transAxes, fontweight="bold", fontsize=8,
               ha="right", va="bottom")
    out = figures_dir() / "si_probe_panel.png"
    fig.savefig(out, dpi=300)
    for ax in ("rel", "ret"):
        s_ = S[S.axis == ax].set_index("probe").loc[PROBES]
        print(ax, "n_pred_dir", s_.n_pred_dir.to_dict(), "p_one", s_.p_one.round(3).to_dict())
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
