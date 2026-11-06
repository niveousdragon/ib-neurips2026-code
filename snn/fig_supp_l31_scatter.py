"""Supplementary figure: per-neuron (rho_n, I_n) scatter at L3.0/L3.1/L4.0.

Three panels at training epoch e=700, 64 neurons per panel. Point color
encodes MEI max-activation: blue = low (broadly tuned), red = high
(sharp tuned). OLS fit on the full panel population overlaid as a black
line; Pearson r printed.

Reproduces the L3.1-decoupling result of Supplementary Section
"Per-layer (rho_n, I_n) scatter colored by MEI max-activation": L3.0
and L4.0 slope negative (-0.42, -0.66); L3.1 flattens (~+0.01) due to
a broadly-tuned sub-population in the (low-rho_n, mid-to-high-I_n)
corner of the plane.

Input:
  <IB_RESULTS>/snn/neuron_mi_vs_mei.csv

Output:
  <IB_FIGURES>/fig_supp_l31_scatter.png
"""
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ib.paths import figures_dir, results_dir

PANEL_LAYERS = ["layer3.0.sn1", "layer3.1.sn1", "layer4.0.sn1"]
SHORT = {"layer3.0.sn1": "L3.0", "layer3.1.sn1": "L3.1", "layer4.0.sn1": "L4.0"}
EPOCH = 700


def _ols_line(x, y):
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def main():
    csv = results_dir("snn") / "neuron_mi_vs_mei.csv"
    if not csv.exists():
        raise SystemExit(
            f"Per-neuron MI/MEI table not found: {csv}. "
            "Run `python snn/08_neuron_mi_mei.py` first (or set IB_RESULTS)."
        )
    df = pd.read_csv(csv)
    df = df[df.epoch == EPOCH].copy()

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    for ax, layer in zip(axes, PANEL_LAYERS):
        sub = df[df.layer == layer]
        if len(sub) < 3:
            ax.set_title(f"{SHORT[layer]} (no data)")
            continue
        rp = sub.rel_perpl.values
        mi = sub.mi_neuron.values
        ma = sub.max_activation.values

        ax.scatter(rp, mi, c=ma, cmap=cmap, norm=norm,
                   s=44, edgecolor="black", linewidths=0.4, alpha=0.92)

        if np.std(rp) > 0 and np.std(mi) > 0:
            slope, intercept = _ols_line(rp, mi)
            xs = np.linspace(rp.min(), rp.max(), 64)
            ax.plot(xs, slope * xs + intercept, color="black", lw=1.5)
            r, _ = pearsonr(rp, mi)
        else:
            r = float("nan")

        ax.set_title(f"{SHORT[layer]}, epoch {EPOCH}  ($r = {r:+.2f}$, $n = {len(sub)}$)",
                     fontsize=12)
        ax.set_xlabel(r"$\rho_n$ (MEI relative perplexity)", fontsize=11)
        ax.set_ylabel(r"$I(s_n; Y)$ (bits)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.012)
    cbar.set_label("MEI max-activation", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out = figures_dir() / "fig_supp_l31_scatter.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
