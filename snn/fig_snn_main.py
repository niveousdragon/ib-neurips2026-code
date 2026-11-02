"""Main-text SNN figures from cached results (Figure 1 and Figure 2 of the paper).

  fig1_snn_information_plane.png (--fig 1; LaTeX source file name fig2.png)
      Top row: information-plane trajectories I(Z;Y) vs H_gauss(T) of the five panel
      sublayers, one point per epoch checkpoint, colored by epoch.
      Bottom row: layer profiles of the intrinsic dimension (left) and of I(Z;Y)
      (right) at eight epochs, with the per-epoch parabolic-fit peak positions.
  fig2_snn_neuron_mei.png (--fig 2; LaTeX source file name fig3_optionA.png)
      Left: per-layer summary at epoch 700, mean per-neuron I(t_n;Y) +/- 95% CI
      (bars) and layer-level I(Z;Y) (line).
      Right: Pearson r(I_n, rho_n) by epoch for the five panel sublayers; inset, the
      same correlation at epoch 700.

Inputs: <IB_RESULTS>/snn/latent_mi.npz, dimensionality.npz, neuron_mi_vs_mei.csv.
Outputs go to <IB_FIGURES>. No recomputation.
"""
import argparse

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ib.paths import figures_dir, results_dir


def _truncated_cmap(name, lo=0.2, hi=1.0, n=256):
    """Return a LinearSegmentedColormap restricted to [lo, hi] of the source.

    Used to skip the very dark and very bright ends of the colormap, where adjacent
    samples become indistinguishable.
    """
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(lo, hi, n))
    )


class EaseOutNorm(mcolors.Normalize):
    """Concave power norm: position = 1 - (1 - linear)**k.

    Distributes data so the colormap step *decreases* with the data value:
    early values spread across the dark/blue end of the colormap, late
    values cluster together near the bright end. k=2 is a quadratic ease-out.
    """

    def __init__(self, vmin=None, vmax=None, k=2.0, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.k = k

    def __call__(self, value, clip=None):
        if self.vmin == self.vmax:
            return np.zeros_like(np.asarray(value, dtype=float))
        x = (np.asarray(value, dtype=float) - self.vmin) / (self.vmax - self.vmin)
        x = np.clip(x, 0.0, 1.0)
        return np.ma.array(1.0 - (1.0 - x) ** self.k)

    def inverse(self, value):
        v = np.asarray(value, dtype=float)
        x = 1.0 - (1.0 - v) ** (1.0 / self.k)
        return self.vmin + x * (self.vmax - self.vmin)


# --- Font sizes (sized for paper-column rendering at \linewidth = 5.5in;
# saved figs are 13-16in wide, so labels need to be large at save time
# to remain legible after the LaTeX scale-down). -------------------------
TITLE_FS  = 18
LABEL_FS  = 16
TICK_FS   = 14
LEGEND_FS = 14
INSET_LABEL_FS = 11
INSET_TICK_FS  = 10

SN1_DEPTH = [
    "sn1", "layer1.0.sn1", "layer1.1.sn1",
    "layer2.0.sn1", "layer2.1.sn1",
    "layer3.0.sn1", "layer3.1.sn1",
    "layer4.0.sn1", "layer4.1.sn1",
]
SHORT = {
    "sn1": "sn1",
    "layer1.0.sn1": "L1.0", "layer1.1.sn1": "L1.1",
    "layer2.0.sn1": "L2.0", "layer2.1.sn1": "L2.1",
    "layer3.0.sn1": "L3.0", "layer3.1.sn1": "L3.1",
    "layer4.0.sn1": "L4.0", "layer4.1.sn1": "L4.1",
}


def _short_full_label(name):
    if name == "sn1":
        return "sn1"
    return name.replace("layer", "L")


# Selected epochs to display as colored lines in bottom-row layer profiles.
PROFILE_EPOCHS = [5, 20, 50, 100, 200, 400, 700, 900]


def fig_information_plane(cmap_name="turbo", out_name="fig1_snn_information_plane.png"):
    z = np.load(results_dir("snn") / "latent_mi.npz")
    layers_lat = list(z["layers"])
    epochs = np.array(z["epochs"])
    iy = z["I_latent"]
    hg = z["H_gauss"]

    dim = np.load(results_dir("snn") / "dimensionality.npz")
    layers_dim = list(dim["layers"])
    id_nn = dim["id_nn"]

    panel_layers = ["layer2.1.sn1", "layer3.0.sn1", "layer3.1.sn1",
                    "layer4.0.sn1", "layer4.1.sn1"]

    fig = plt.figure(figsize=(16.0, 8.5), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.10, hspace=0.075)
    gs = fig.add_gridspec(2, 10, height_ratios=[1.0, 0.95])

    # ----- Top row: IB-plane trajectories (5 panels) -----
    # Truncated turbo [0.1, 0.9] so the very-dark navy and very-bright
    # red endpoints are skipped. EaseOutNorm spreads early epochs across
    # the dark end with larger gaps and compresses late epochs near the
    # bright end.
    cmap = _truncated_cmap(cmap_name, lo=0.1, hi=0.9)
    norm = EaseOutNorm(vmin=epochs.min(), vmax=epochs.max(), k=2.0)
    sc = None
    top_axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        top_axes.append(ax)
    for ax, layer in zip(top_axes, panel_layers):
        if layer not in layers_lat:
            ax.set_visible(False)
            continue
        li = layers_lat.index(layer)
        x = hg[li]
        y = iy[li]
        ax.plot(x, y, "-", color="0.7", lw=0.8, alpha=0.7, zorder=1)
        sc = ax.scatter(x, y, c=epochs, cmap=cmap, norm=norm,
                        s=50, edgecolor="k", lw=0.4, zorder=2)
        ax.set_title(SHORT[layer] + ".sn1", fontsize=TITLE_FS)
        ax.set_xlabel(r"$H_\mathrm{gauss}(T)$, bits", fontsize=LABEL_FS)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=TICK_FS)
    top_axes[0].set_ylabel("I(Z; Y), bits", fontsize=LABEL_FS)

    # Note: colorbar is created later, after bottom_left and bottom_right
    # are defined, so it can span horizontally below all panels.

    # ----- Bottom row: layer-wise profiles, multi-epoch lines -----
    layers_full = layers_lat
    short_full = [_short_full_label(l) for l in layers_full]
    x_layers = np.arange(len(layers_full))

    # Pre-compute per-epoch arrays for ID (sn1+sn2 17 sublayers, NaN if absent)
    id_per_layer_per_epoch = np.full(
        (len(layers_full), len(epochs)), np.nan)
    for k, layer in enumerate(layers_full):
        if layer in layers_dim:
            li = layers_dim.index(layer)
            id_per_layer_per_epoch[k] = id_nn[li]

    bottom_left = fig.add_subplot(gs[1, 0:5])
    bottom_right = fig.add_subplot(gs[1, 5:10])

    # Bottom-row lines reuse the colormap of the top row (one colorbar serves both).
    epochs_list = list(epochs)

    def _parabola_vertex(y, x, half_window=4):
        """Quadratic fit on a window of half_window layers either side of
        argmax(y). Restricted-window fit avoids the asymmetric tail of the
        ID/I(Z;Y) profile (gentle decay through L3-L4) skewing a global
        parabola. Returns (xs, ys) or NaN."""
        m = np.isfinite(y)
        if m.sum() < 4:
            return np.nan
        am = int(np.argmax(np.where(m, y, -np.inf)))
        lo = max(0, am - half_window)
        hi = min(len(x), am + half_window + 1)
        sub_x = x[lo:hi]
        sub_y = y[lo:hi]
        sm = np.isfinite(sub_y)
        if sm.sum() < 4:
            return np.nan
        a, b, c = np.polyfit(sub_x[sm], sub_y[sm], deg=2)
        if a >= 0 or abs(a) < 1e-10:
            return np.nan
        xs = -b / (2 * a)
        if xs < sub_x[sm].min() or xs > sub_x[sm].max():
            return np.nan
        ys = a * xs ** 2 + b * xs + c
        return xs, ys

    id_vertices = []
    iy_vertices = []
    for ep in PROFILE_EPOCHS:
        if ep not in epochs_list:
            continue
        ei = epochs_list.index(ep)
        color = cmap(norm(ep))
        bottom_left.plot(x_layers, id_per_layer_per_epoch[:, ei], "o-",
                         color=color, lw=1.5, ms=4, alpha=0.95,
                         label=f"epoch {ep}")
        bottom_right.plot(x_layers, iy[:, ei], "o-",
                          color=color, lw=1.5, ms=4, alpha=0.95,
                          label=f"epoch {ep}")
        v_id = _parabola_vertex(id_per_layer_per_epoch[:, ei], x_layers)
        v_iy = _parabola_vertex(iy[:, ei], x_layers)
        if v_id is not np.nan and isinstance(v_id, tuple):
            id_vertices.append((ep, v_id[0], v_id[1]))
        if v_iy is not np.nan and isinstance(v_iy, tuple):
            iy_vertices.append((ep, v_iy[0], v_iy[1]))

    # Grey dashed trajectory of parabolic vertex across training epochs
    if id_vertices:
        vx = [v[1] for v in id_vertices]
        vy = [v[2] for v in id_vertices]
        bottom_left.plot(vx, vy, "x", color="0.35", ms=8, mew=1.4, zorder=10)
        bottom_left.plot(vx, vy, "--", color="0.35", lw=1.2, alpha=0.85,
                         zorder=9, label="parabolic peak (Supp. S4)")
    if iy_vertices:
        vx = [v[1] for v in iy_vertices]
        vy = [v[2] for v in iy_vertices]
        bottom_right.plot(vx, vy, "x", color="0.35", ms=8, mew=1.4, zorder=10)
        bottom_right.plot(vx, vy, "--", color="0.35", lw=1.2, alpha=0.85,
                          zorder=9, label="parabolic peak (Supp. S4)")

    for ax_b, ylabel, title in [
        (bottom_left, "ID (2-NN)", "ID across layers, per epoch"),
        (bottom_right, "I(Z; Y), bits",
         "I(Z; Y) across layers, per epoch"),
    ]:
        ax_b.set_xticks(x_layers)
        ax_b.set_xticklabels(short_full, fontsize=TICK_FS - 2,
                             rotation=45, ha="right")
        ax_b.set_xlabel("sublayer (network depth)", fontsize=LABEL_FS)
        ax_b.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax_b.set_title(title, fontsize=TITLE_FS)
        ax_b.grid(alpha=0.3, lw=0.4, axis="y")
        ax_b.tick_params(labelsize=TICK_FS)

    bottom_right.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    bottom_right.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

    # Single horizontal colorbar at the bottom, shared by all 7 panels
    # (5 IB-plane + 2 layer-profiles, same colormap and epoch range).
    cbar = fig.colorbar(sc, ax=top_axes + [bottom_left, bottom_right],
                        location="bottom", shrink=0.6, pad=0.06, aspect=45)
    cbar.set_label("training epoch", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    # Explicit ticks; EaseOutNorm compresses late epochs near the right
    # edge so a uniform 0..900 step would overlap. Pick 5 ticks at
    # positions visually well-spaced after the ease-out transform.
    cbar.set_ticks([0, 100, 300, 600, 900])

    out = figures_dir() / out_name
    # The published files were rendered once before saving; constrained layout is
    # recomputed on each draw, so the extra draw keeps the saved layout identical.
    fig.canvas.draw()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    return out


def fig_neuron_mei(out_name="fig2_snn_neuron_mei.png"):
    """Per-layer summary at epoch 700 (left) and r(I_n, rho_n) by epoch (right)."""
    # Slightly smaller fonts than the information-plane figure: these panels are
    # more text-dense (two y axes on the left, an inset on the right).
    F_TITLE = TITLE_FS - 3
    F_LABEL = LABEL_FS - 3
    F_TICK = TICK_FS - 3
    F_LEGEND = LEGEND_FS - 3
    df = pd.read_csv(results_dir("snn") / "neuron_mi_vs_mei.csv")
    df_e700 = df[df.epoch == 700].copy()

    rows = []
    for layer in SN1_DEPTH:
        sub = df_e700[df_e700.layer == layer]
        mi = sub.mi_neuron.values
        rp = sub.rel_perpl.values
        if len(mi) == 0:
            rows.append({"layer": layer, "mean_mi": np.nan,
                         "ci95": np.nan, "r": np.nan, "n": 0})
            continue
        n = len(mi)
        std = float(mi.std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        ci95 = 2.0 * sem  # 95% CI ~ 2 SEM (precise: 1.96)
        if n >= 3 and np.std(mi) > 0 and np.std(rp) > 0:
            r_e700, _ = pearsonr(mi, rp)
        else:
            r_e700 = float("nan")
        rows.append({"layer": layer,
                     "mean_mi": float(mi.mean()),
                     "ci95": ci95, "r": float(r_e700), "n": n})
    summary = pd.DataFrame(rows)

    z = np.load(results_dir("snn") / "latent_mi.npz")
    layers_lat = list(z["layers"])
    epochs_lat = list(z["epochs"])
    e_idx = epochs_lat.index(700) if 700 in epochs_lat else len(epochs_lat) - 1
    iy_layer = []
    for layer in SN1_DEPTH:
        if layer in layers_lat:
            li = layers_lat.index(layer)
            iy_layer.append(float(z["I_latent"][li, e_idx]))
        else:
            iy_layer.append(np.nan)

    # r(epoch) trajectory for 5 select layers only
    select_layers = ["layer2.1.sn1", "layer3.0.sn1", "layer3.1.sn1",
                     "layer4.0.sn1", "layer4.1.sn1"]
    epochs_avail = sorted(df.epoch.unique())
    r_traj = {layer: [] for layer in select_layers}
    for ep in epochs_avail:
        for layer in select_layers:
            sub = df[(df.epoch == ep) & (df.layer == layer)]
            mi = sub.mi_neuron.values
            rp = sub.rel_perpl.values
            if len(mi) >= 3 and np.std(mi) > 0 and np.std(rp) > 0:
                r, _ = pearsonr(mi, rp)
            else:
                r = np.nan
            r_traj[layer].append(r)

    fig = plt.figure(figsize=(13.0, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axD = fig.add_subplot(gs[0, 1])

    short_labels = [SHORT[l] for l in SN1_DEPTH]
    x = np.arange(len(SN1_DEPTH))

    # ---- Left: bars + I(Z;Y) line, 2 axes ----
    axA.bar(x, summary.mean_mi.values, yerr=summary.ci95.values,
            color="0.7", alpha=0.85, edgecolor="k", capsize=3,
            label=r"per-neuron $I(t_n; Y)$, mean $\pm$ 95% CI")
    axA.set_ylabel(r"per-neuron $I(t_n; Y)$, bits", fontsize=F_LABEL)
    axA.tick_params(axis="y", labelsize=F_TICK)
    axA.set_xticks(x)
    axA.set_xticklabels(short_labels, fontsize=F_TICK, rotation=0)
    axA.set_xlabel("layer", fontsize=F_LABEL)
    axA.grid(alpha=0.3, lw=0.4, axis="y")
    axA.set_title("Per-layer summary at epoch 700", fontsize=F_TITLE)

    axA2 = axA.twinx()
    axA2.plot(x, iy_layer, "o-", color="C3", lw=1.7, ms=6,
              label="layer-level I(Z; Y)")
    axA2.set_ylabel("layer-level I(Z; Y), bits", fontsize=F_LABEL)
    axA2.tick_params(axis="y", labelsize=F_TICK)

    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axA2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2,
               loc="upper center", bbox_to_anchor=(0.5, -0.18),
               fontsize=F_LEGEND, frameon=False, ncol=2)

    # ---- Right: r(epoch) for 5 select layers, turbo by depth ----
    # Truncated turbo [0.1, 0.9] -- skip very-dark and very-bright endpoints.
    cmap_depth = _truncated_cmap("turbo", lo=0.1, hi=0.9)
    n_select = len(select_layers)
    for i, layer in enumerate(select_layers):
        color = cmap_depth(i / max(1, n_select - 1))
        axD.plot(epochs_avail, r_traj[layer], "o-",
                 color=color, lw=1.7, ms=5,
                 label=SHORT[layer])
    axD.axhline(0, c="k", lw=0.5)
    axD.set_xscale("log")
    axD.set_xlabel("training epoch (log scale)", fontsize=F_LABEL)
    axD.set_ylabel(r"Pearson $r(I_n,\ \rho_n)$", fontsize=F_LABEL)
    axD.set_ylim(-1.0, 0.35)  # extend lower bound for inset/legend clearance
    axD.grid(alpha=0.3, lw=0.4)
    axD.set_title(r"$r(I_n,\ \rho_n)$ by epoch", fontsize=F_TITLE)
    # Right-panel legend in 2 rows (ncol=3 for 5 entries -> 3+2 layout).
    axD.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               fontsize=F_LEGEND, frameon=False, ncol=3)
    axD.tick_params(labelsize=F_TICK)

    # ---- Inset: bar chart of per-layer r at e=700 (5 select layers).
    # Small, bottom-centered.
    inset = axD.inset_axes([0.36, 0.065, 0.30, 0.26])
    select_short = [SHORT[l] for l in select_layers]
    r_select = [float(summary[summary.layer == l].r.iloc[0])
                if (summary.layer == l).any() else np.nan
                for l in select_layers]
    x_inset = np.arange(len(select_layers))
    bar_colors_inset = [cmap_depth(i / max(1, len(select_layers) - 1))
                        for i in range(len(select_layers))]
    inset.bar(x_inset, r_select, color=bar_colors_inset,
              edgecolor="k", lw=0.4)
    inset.axhline(0, c="k", lw=0.4)
    inset.set_xticks(x_inset)
    inset.set_xticklabels(select_short, fontsize=INSET_TICK_FS - 2,
                          rotation=0)
    inset.set_ylabel("r", fontsize=INSET_LABEL_FS - 2, labelpad=1)
    inset.tick_params(axis="y", labelsize=INSET_TICK_FS - 2)
    inset.set_ylim(-0.85, 0.10)
    inset.grid(alpha=0.3, lw=0.3, axis="y")
    inset.set_facecolor((1, 1, 1, 0.92))
    # Caption to the right of the inset (in right-panel axes coordinates)
    axD.text(0.68, 0.06, r"$r$ at epoch 700",
             transform=axD.transAxes, fontsize=INSET_LABEL_FS,
             va="center", ha="left")

    out = figures_dir() / out_name
    # The published files were rendered once before saving; constrained layout is
    # recomputed on each draw, so the extra draw keeps the saved layout identical.
    fig.canvas.draw()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    return out


FIGS = {"1": fig_information_plane, "2": fig_neuron_mei}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fig", default="all", choices=["all"] + list(FIGS))
    args = parser.parse_args()
    for key, func in FIGS.items():
        if args.fig in ("all", key):
            func()


if __name__ == "__main__":
    main()
