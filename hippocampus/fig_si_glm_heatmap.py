"""Supplementary figure: per-session GLM cohort summary of the earlier 20 Hz all-neuron
lambda = 0 pipeline (64 sessions).

Cell colour: cohort mean of Delta r^2 = r^2_real - r^2_shuffle at each (greedy latent
rank, predictor); real and shuffle latents are ordered independently, so the shuffle
also carries the selection bias of the ordering. Star: paired one-sided Wilcoxon
p < 0.001 and mean Delta r^2 >= 0.05.

Input:  results/hippocampus/glm_r2_legacy_20hz.csv (legacy_02_glm_r2.py)
Output: <figures>/glm_latent_heatmap.png
Usage:  python hippocampus/fig_si_glm_heatmap.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hippo_common import table_path
from scipy import stats

from ib.paths import figures_dir

PREDICTORS = ["x", "y", "d_obj1", "d_obj2", "d_obj3", "d_obj4", "speed", "ang_vel"]
PRED_LABELS = [r"$x$", r"$y$", r"$d_\mathrm{obj1}$", r"$d_\mathrm{obj2}$",
               r"$d_\mathrm{obj3}$", r"$d_\mathrm{obj4}$", "speed", r"$\omega$"]
P_THRESHOLD = 1e-3
EFFECT_THRESHOLD = 0.05


def cohort_matrices(df):
    """(n_predictors, n_ranks) mean Delta r^2 and Wilcoxon p."""
    df = df.assign(delta=df.r2_real - df.r2_shuf)
    n_dim = df["rank"].max()
    M = np.full((len(PREDICTORS), n_dim), np.nan)
    P = np.ones_like(M)
    for j, p in enumerate(PREDICTORS):
        for k in range(n_dim):
            d = df[(df.predictor == p) & (df["rank"] == k + 1)].delta.dropna().values
            M[j, k] = d.mean()
            if len(d) >= 5 and not np.allclose(d, 0):
                try:
                    P[j, k] = stats.wilcoxon(d, alternative="greater").pvalue
                except ValueError:
                    pass
    return M, P


def main():
    df = pd.read_csv(table_path("glm_r2_legacy_20hz.csv"))
    n_sess = df[["mouse", "day"]].drop_duplicates().shape[0]
    M, P = cohort_matrices(df)
    n_pred, n_dim = M.shape
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    im = ax.imshow(M, cmap="magma", aspect="auto", vmin=0, vmax=max(0.1, M.max()))
    ax.set_yticks(range(n_pred))
    ax.set_yticklabels(PRED_LABELS, fontsize=12)
    ax.set_xticks(range(n_dim))
    ax.set_xticklabels([f"{i + 1}" for i in range(n_dim)], fontsize=10)
    ax.set_xlabel("greedy-assigned latent rank (per session)", fontsize=12)
    ax.set_ylabel("predictor", fontsize=12)
    ax.set_title(r"Bias-corrected $\Delta r^2 = r^2_\mathrm{real} - r^2_\mathrm{shuffle}$, "
                 f"cohort mean ({n_sess} sessions)", fontsize=12, loc="left")
    n_sig = 0
    for j in range(n_pred):
        for k in range(n_dim):
            val = M[j, k]
            if P[j, k] < P_THRESHOLD and val >= EFFECT_THRESHOLD:
                n_sig += 1
                ax.text(k, j + 0.30, "*", ha="center", va="center", fontsize=14,
                        color="white", fontweight="bold")
            ax.text(k, j, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val < 0.5 * M.max() else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
    cbar.set_label(r"mean $\Delta r^2$", fontsize=11)
    fig.text(0.02, -0.04, rf"* = paired Wilcoxon $p < {P_THRESHOLD:g}$ "
             rf"AND mean $\Delta r^2 \geq {EFFECT_THRESHOLD}$  ({n_sig}/{n_pred * n_dim} cells)",
             ha="left", fontsize=9, color="0.3")
    fig.tight_layout()
    out = figures_dir() / "glm_latent_heatmap.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"significant cells: {n_sig} / {n_pred * n_dim}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
