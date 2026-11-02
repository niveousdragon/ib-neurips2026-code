"""Relevance axis with the MLP decoder vs the logistic decoder (main text and
Supplementary "Decoder family for the relevance axis": 11/16 rising, p = 0.037,
per-mouse Pearson r = 0.83 between the two decoders' slopes).

Inputs: results/hippocampus/ib_plane_mlp_lambda1.csv, ib_plane_lambda1.csv
Output: printed; results/hippocampus/stats/decoder_mlp.txt
Usage:  python hippocampus/stats_decoder_mlp.py
"""
import pandas as pd
from hippo_common import per_mouse_slopes, stats_dir, table_path
from scipy import stats
from sessions import cohort


def main():
    mlp = pd.read_csv(table_path("ib_plane_mlp_lambda1.csv"))
    lr = pd.read_csv(table_path("ib_plane_lambda1.csv"))
    L = []
    for name in ("drop3", "all64"):
        sl = {}
        for dec, df in (("MLP", mlp), ("logreg", lr)):
            sl[dec] = per_mouse_slopes(cohort(df, name), "I_zy_bin_corr")
            p = stats.wilcoxon(sl[dec].values, alternative="greater").pvalue
            L.append(f"{name} {dec:6s}: rising {int((sl[dec] > 0).sum())}/{len(sl[dec])}, "
                     f"one-sided Wilcoxon p={p:.4f}")
        ab = pd.concat([sl["MLP"], sl["logreg"]], axis=1).dropna()
        r = stats.pearsonr(ab[0], ab[1])
        L.append(f"{name} per-mouse Pearson r(MLP slope, logreg slope) = {r[0]:.3f} "
                 f"(p={r[1]:.4f}, n={len(ab)})")
        m, g = cohort(mlp, name), cohort(lr, name)
        L.append(f"{name} mean raw MI: MLP {m.I_zy_bin_real.mean():.3f}, logreg {g.I_zy_bin_real.mean():.3f}; "
                 f"shuffle: MLP {m.I_zy_bin_shuf_mean.mean():.3f}, logreg {g.I_zy_bin_shuf_mean.mean():.3f}")
    txt = "\n".join(L)
    (stats_dir() / "decoder_mlp.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
