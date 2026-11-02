"""Supplementary figure: SNN information-plane trajectories at d_Z=8.

Mirrors the main paper Figure 1 top row, but with the autoencoder
bottleneck dimension reduced from 16 to 8. Five panels for sublayers
L2.1, L3.0, L3.1, L4.0, L4.1; each point is one training-epoch
checkpoint colored by epoch; trajectory connects checkpoints in
chronological order.

Input axis H_gauss(T) is computed on the spike-count summary T
directly and is therefore identical to the d_Z=16 figure (it does
not pass through the AE).

Inputs:
  <IB_RESULTS>/snn/latent_mi.npz       (H_gauss column)
  <IB_RESULTS>/snn/latent_mi_dim8.npz  (I(Z;Y) at d_Z=8)

Output:
  <IB_FIGURES>/fig_supp_dim8_iplane.png
"""
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ib.paths import figures_dir, results_dir

PANEL_LAYERS = ["layer2.1.sn1", "layer3.0.sn1", "layer3.1.sn1",
                "layer4.0.sn1", "layer4.1.sn1"]
SHORT = {
    "layer2.1.sn1": "L2.1",
    "layer3.0.sn1": "L3.0",
    "layer3.1.sn1": "L3.1",
    "layer4.0.sn1": "L4.0",
    "layer4.1.sn1": "L4.1",
}


def _truncated_cmap(name, lo=0.1, hi=0.9, n=256):
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(lo, hi, n))
    )


def main():
    d_main = np.load(results_dir("snn") / "latent_mi.npz")
    d8 = np.load(results_dir("snn") / "latent_mi_dim8.npz")

    layers16 = list(d_main["layers"])
    epochs = np.array(d_main["epochs"])
    hg = d_main["H_gauss"]

    layers8 = list(d8["layers"])
    iy8 = d8["I_latent"]

    cmap = _truncated_cmap("turbo", lo=0.1, hi=0.9)
    norm = mcolors.Normalize(vmin=epochs.min(), vmax=epochs.max())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
    for ax, layer in zip(axes, PANEL_LAYERS):
        i_main = layers16.index(layer)
        i_d8 = layers8.index(layer)
        x = hg[i_main]
        y = iy8[i_d8]
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        order = np.argsort(epochs[valid])
        xs = x[valid][order]
        ys = y[valid][order]
        es = epochs[valid][order]

        for k in range(len(xs) - 1):
            ax.plot(xs[k:k + 2], ys[k:k + 2],
                    color=cmap(norm(es[k])), lw=1.4, alpha=0.9)
        sc = ax.scatter(xs, ys, c=es, cmap=cmap, norm=norm, s=42,
                        edgecolor="black", linewidths=0.4, zorder=3)

        ax.set_title(f"{SHORT[layer]}", fontsize=14)
        ax.set_xlabel(r"$H_\mathrm{gauss}(T)$ (bits)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    axes[0].set_ylabel(r"$I(Z; Y)$ (bits), $d_Z = 8$", fontsize=12)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label("training epoch", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out = figures_dir() / "fig_supp_dim8_iplane.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
