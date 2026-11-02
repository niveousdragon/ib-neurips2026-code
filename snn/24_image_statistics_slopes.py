"""Slopes of image-statistics and class decodability over training; Supplementary Figure (from cached results).

Per sublayer: OLS slope of held-out R^2 against log10(epoch) over epochs >= 30, for
the six image statistics (mean R^2) and for the class (ridge decoder of
10_image_statistics_decoding.py); interval = 2.5-97.5 percentiles of the slope
refitted on each of the 30 splits of 11_image_statistics_bootstrap.py. The four
stage-4 sublayers, where H_gauss(T) reverses, are compared with the other 13.

Inputs:  <IB_RESULTS>/snn/image_statistics_decoding.csv, image_statistics_bootstrap.csv,
         latent_mi.npz (H_gauss peak epochs)
Outputs: <IB_RESULTS>/snn/image_statistics_slopes.csv, <IB_FIGURES>/fig_supp_snn_aux.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from snn_common import AUX_NAMES, short_name

from ib.paths import figures_dir, results_dir

REVERSING = ["layer4.0.sn1", "layer4.0.sn2", "layer4.1.sn1", "layer4.1.sn2"]
N_SPLITS = 30


def slope(x, y):
    m = np.isfinite(y)
    return np.polyfit(np.log10(x[m]), y[m], 1)[0]


def compute(res):
    lmi = np.load(res / "latent_mi.npz")
    layers = [str(x) for x in lmi["layers"]]
    epochs = np.array(lmi["epochs"])
    h = lmi["H_gauss"]
    a = pd.read_csv(res / "image_statistics_decoding.csv")
    b = pd.read_csv(res / "image_statistics_bootstrap.csv")
    print(f"decoding rows: {len(a)}; bootstrap rows: {len(b)}")
    rows = []
    for d, layer in enumerate(layers):
        s = a[a.layer == layer].sort_values("epoch")
        s30 = s[s.epoch >= 30]
        ep = s30.epoch.values.astype(float)
        row = dict(layer=layer, depth=d, reversing=layer in REVERSING,
                   aux_slope=slope(ep, s30.r2_aux.values),
                   class_slope=slope(ep, s30.r2_class.values),
                   aux_e30=float(s30.r2_aux.values[0]), aux_e900=float(s30.r2_aux.values[-1]))
        for st in AUX_NAMES:
            row[f"slope_{st}"] = slope(ep, s30[f"r2_{st}"].values)
        bb = b[b.layer == layer].sort_values("epoch")
        bep = bb.epoch.values.astype(float)
        bs = [slope(bep, bb[f"b{k}"].values.astype(float)) for k in range(N_SPLITS)]
        row["boot_mean"] = float(np.mean(bs))
        row["boot_lo"], row["boot_hi"] = np.percentile(bs, [2.5, 97.5])
        # Slope measured from this sublayer's own H_gauss maximum onward.
        pk = int(epochs[np.nanargmax(h[d])])
        row["H_peak_epoch"] = pk
        post = s[s.epoch >= max(pk, 1)]
        row["aux_slope_after_Hpeak"] = (slope(post.epoch.values.astype(float), post.r2_aux.values)
                                        if len(post) >= 3 else np.nan)
        row["aux_argmax_epoch"] = int(s.epoch.values[np.argmax(s.r2_aux.values)])
        rows.append(row)
    return pd.DataFrame(rows)


def report(df):
    pd.set_option("display.width", 250)
    print(df[["layer", "aux_slope", "boot_mean", "boot_lo", "boot_hi", "class_slope", "aux_e30",
              "aux_e900", "H_peak_epoch", "aux_slope_after_Hpeak", "aux_argmax_epoch"]]
          .round(3).to_string(index=False))
    ctrl, rev = df[~df.reversing], df[df.reversing]
    print(f"\nnon-reversing aux slopes: {ctrl.aux_slope.min():+.3f} to {ctrl.aux_slope.max():+.3f}; "
          f"negative {int((ctrl.aux_slope < 0).sum())}/{len(ctrl)}; "
          f"lowest interval bound {ctrl.boot_lo.min():+.3f}")
    print("reversing aux slopes: " + ", ".join(
        f"{short_name(r.layer)} {r.aux_slope:+.3f} [{r.boot_lo:+.3f},{r.boot_hi:+.3f}]"
        for r in rev.itertuples()))
    print(f"Mann-Whitney (reversing < others) p = "
          f"{mannwhitneyu(rev.aux_slope, ctrl.aux_slope, alternative='less').pvalue:.4f}")
    for q in ["aux_slope", "class_slope"]:
        rs, ps = spearmanr(df.depth, df[q])
        rp, pp = pearsonr(df.depth, df[q])
        print(f"depth vs {q}: Spearman {rs:+.2f} (p={ps:.2g}); Pearson {rp:+.2f} (p={pp:.2g})")
    print(f"class slope: reversing {rev.class_slope.min():+.3f}..{rev.class_slope.max():+.3f}; "
          f"others {ctrl.class_slope.min():+.3f}..{ctrl.class_slope.max():+.3f}")
    print("aux slope from own H_gauss peak onward: " + ", ".join(
        f"{short_name(r.layer)} (peak {r.H_peak_epoch}) {r.aux_slope_after_Hpeak:+.3f}"
        for r in rev.itertuples()))


def figure(df, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), layout="constrained")
    x = np.arange(len(df))
    cols = ["#c0392b" if r else "0.45" for r in df.reversing]
    ax[0].bar(x, df.aux_slope, color=cols, edgecolor="k", lw=0.3)
    ax[0].errorbar(x, df.boot_mean, yerr=[df.boot_mean - df.boot_lo, df.boot_hi - df.boot_mean],
                   fmt="none", ecolor="k", capsize=1.2, lw=0.6)
    ax[0].set_title("six auxiliary image statistics")
    ax[1].bar(x, df.class_slope, color=cols, edgecolor="k", lw=0.3)
    ax[1].set_title("class label (same ridge decoder)")
    for a_ in ax:
        a_.axhline(0, c="k", lw=0.5)
        a_.set_xticks(x)
        a_.set_xticklabels([short_name(lay) for lay in df.layer], rotation=60, ha="right",
                           rotation_mode="anchor", fontsize=5.5)
        a_.set_xlabel("sublayer (network depth)")
        a_.set_ylabel("slope of held-out $R^2$\n(per decade of epochs)")
        a_.grid(alpha=0.3, axis="y", lw=0.4)
    ax[0].legend(handles=[Patch(color="0.45", label="$H_\\mathrm{gauss}(T)$ does not reverse (13)"),
                          Patch(color="#c0392b", label="$H_\\mathrm{gauss}(T)$ reverses (L4, 4)")],
                 frameon=False, loc="lower left", handlelength=1.2)
    # The published file was rendered once on screen before saving; constrained layout
    # is recomputed on each draw, so the extra draw keeps the saved layout identical.
    fig.canvas.draw()
    fig.savefig(out, dpi=300)
    print(f"[OK] saved {out}")


def main():
    res = results_dir("snn")
    df = compute(res)
    df.to_csv(res / "image_statistics_slopes.csv", index=False, float_format="%.4f")
    report(df)
    figure(df, figures_dir() / "fig_supp_snn_aux.png")


if __name__ == "__main__":
    main()
