"""Summary of the SNN label-permutation null, clipped and unclipped (from cached results).

Backs the statements of Methods "Bias correction" and Supplementary "Shuffle nulls":
the permuted KSG estimate sat at the estimator's floor of zero in every cell, and
without the clip it was negative in every cell, so subtracting it could only raise
I(Z;Y).

Inputs:  <IB_RESULTS>/snn/latent_mi_unclipped_null.npz (04_unclipped_shuffle_null.py),
         latent_mi_with_shuffle.npz (03_shuffle_null.py), latent_mi.npz (main result)
Outputs: <IB_RESULTS>/snn/shuffle_null_corrected.csv, shuffle_null_mean.csv,
         shuffle_null_raw_unclipped.csv (sublayer x epoch tables), printed summary.
The latents behind the null come from the autoencoders of 03_shuffle_null.py, not
from those behind latent_mi.npz (which were not saved); the two runs are compared below.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ib.paths import results_dir


def main():
    res = results_dir("snn")
    u = np.load(res / "latent_mi_unclipped_null.npz")
    p = np.load(res / "latent_mi.npz")
    s = np.load(res / "latent_mi_with_shuffle.npz")
    layers = [str(x) for x in u["layers"]]
    epochs = np.array(u["epochs"])
    raw, null = u["I_raw_unclipped"], u["I_null"]
    nm, ns = null.mean(2), null.std(2, ddof=1)
    corr = raw - nm
    paper = p["I_latent"]

    n_cells = int(np.isfinite(s["I_shuf"]).sum())
    print(f"clipped null (5 permutations): I_shuf == 0 in {(s['I_shuf'] == 0).sum()}/{n_cells} cells")
    print(f"unclipped null (7 permutations): mean < 0 in {(nm < 0).sum()}/{np.isfinite(nm).sum()} cells; "
          f"max over all cells and permutations {np.nanmax(null):+.3f} bits")
    print(f"main-result I_latent == 0 (clipped) in {(paper == 0).sum()}/{np.isfinite(paper).sum()} cells; "
          f"NaN cells (layer, epoch index): {np.argwhere(~np.isfinite(paper)).tolist()}")
    pos = s["I_real"] > 0
    print(f"check: unclipped raw == clipped I_real where positive: max |diff| = "
          f"{np.nanmax(np.abs(raw[pos] - s['I_real'][pos])):.2e}")
    both = (paper > 0) & (s["I_real"] > 0)
    print(f"main result vs shuffle run (different autoencoder training): max |diff| "
          f"{np.nanmax(np.abs(paper - s['I_real'])):.3f}, median |diff| where both > 0 "
          f"{np.nanmedian(np.abs(paper - s['I_real'])[both]):.3f} bits")
    print(f"\nnull mean over cells: {np.nanmean(nm):+.3f}, range [{np.nanmin(nm):+.3f}, "
          f"{np.nanmax(nm):+.3f}] bits; median within-cell SD {np.nanmedian(ns):.3f}")
    print(f"unclipped raw: {(raw < 0).sum()} negative cells, min {np.nanmin(raw):+.3f}")
    print(f"corrected = raw - null: {(corr <= 0).sum()} cells <= 0, min {np.nanmin(corr):+.3f}, "
          f"max {np.nanmax(corr):.3f}")

    rows = []
    for i, layer in enumerate(layers):
        m = np.isfinite(nm[i])
        rows.append(dict(
            layer=layer, null_mean=np.nanmean(nm[i]), null_sd_across_epochs=np.nanstd(nm[i]),
            raw_end=raw[i, -1], corr_end=corr[i, -1], paper_end=paper[i, -1],
            r_paper_corr=pearsonr(paper[i, m], corr[i, m])[0] if np.nanstd(paper[i, m]) > 0 else np.nan,
            peak_ep_paper=int(epochs[np.nanargmax(paper[i])]) if np.nanmax(paper[i]) > 0 else -1,
            peak_ep_corr=int(epochs[np.nanargmax(corr[i])]),
            n_clipped_paper=int((paper[i] == 0).sum())))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nargmax over the 17 sublayers per epoch (main result vs corrected):")
    for j, e in enumerate(epochs):
        if e in (5, 20, 30, 50, 100, 200, 400, 700, 900):
            a = int(np.nanargmax(paper[:, j])) if np.nanmax(paper[:, j]) > 0 else -1
            b = int(np.nanargmax(corr[:, j]))
            print(f"  epoch {e}: main {layers[a] if a >= 0 else 'all zero'} | "
                  f"corrected {layers[b]} ({corr[b, j]:.2f})")
    for name, arr in (("corrected", corr), ("mean", nm), ("raw_unclipped", raw)):
        pd.DataFrame(arr, index=layers, columns=epochs).to_csv(
            res / f"shuffle_null_{name}.csv", float_format="%.4f")
    print(f"\nsaved shuffle_null_{{corrected,mean,raw_unclipped}}.csv in {res}")


if __name__ == "__main__":
    main()
