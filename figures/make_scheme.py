"""Schematic X -> T -> Z -> Y at the top of Table 1 (scheme_XTZ_compact.pdf).

One row: input X, representation T, probe Z = f(T), target Y, with the
retention and relevance estimates beside the arrows. Text is embedded as
TrueType (Type 42) so the PDF carries no Type 3 fonts. Size 5.5 x 0.66 in.

Writes figures_dir()/scheme_XTZ_compact.{pdf,png}. Needs no data.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ib.paths import figures_dir


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8})
    W, H = 5.5, 0.66
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=9)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=7,
                color="#333333")

    def arrow(x0, x1, y, style="-|>", color="#333333"):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle=style, mutation_scale=8,
                                     lw=0.8, color=color))

    Y0, HB = 3.6, 7.6
    box(1, Y0, 13, HB, r"input $X$", "image / grating /\negocentric state", "#f2f2f2")
    box(24, Y0, 17, HB, r"representation $T$", "spikes / calcium\n(system under study)",
        "#dbe8f5")
    box(55, Y0, 16, HB, r"probe $Z=f(T)$", "autoencoder latent\n(estimator probe)", "#fbe7cf")
    box(86, Y0, 13, HB, r"target $Y$", "class / category /\nposition", "#f2f2f2")
    yc = Y0 + HB / 2
    arrow(14.8, 23.2, yc)
    arrow(41.8, 54.2, yc)
    ax.text(48, yc + 0.6, "probe $f$", ha="center", va="bottom", fontsize=7)
    arrow(71.8, 85.2, yc, style="<|-|>", color="#a0522d")
    ax.text(78.5, 0.9, r"relevance $I(Z;Y)\leq I(T;Y)$", ha="center", va="center",
            fontsize=7, color="#a0522d")
    ax.text(32, 0.9, r"retention $I(X;Z)\leq I(X;T)$  (SNN: $H_{\mathrm{gauss}}(T)$ on $T$)",
            ha="center", va="center", fontsize=7, color="#1f4e79")
    out = figures_dir()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"scheme_XTZ_compact.{ext}", dpi=300)
    print(f"[OK] scheme written to {out}")


if __name__ == "__main__":
    main()
