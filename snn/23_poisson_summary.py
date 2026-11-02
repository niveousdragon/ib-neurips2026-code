"""Numbers of Supplementary "Spike-count reconstruction loss" (from cached results).

Inputs: <IB_RESULTS>/snn/poisson_recon_e900.csv and poisson_recon_trajectory.csv
(05_poisson_reconstruction.py). Prints the epoch-900 depth profile under both
reconstruction losses, the trajectory over epochs 30/100/300/900, and the summary
numbers quoted in the paper.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from snn_common import PANEL_LAYERS

from ib.paths import results_dir

COLS = ["layer", "epoch", "I_latent_paper", "I_mse", "I_poisson", "acc_mse", "acc_poisson"]


def main():
    res = results_dir("snn")
    d = pd.read_csv(res / "poisson_recon_e900.csv").drop_duplicates("layer", keep="first")
    d = d.set_index("layer").reindex([lay for lay in PANEL_LAYERS if lay in set(d.layer)])
    print("epoch 900 (I_latent_paper = main pipeline, latent_mi.npz):")
    print(d[COLS[2:] + ["t_cell_s"]].round(3).to_string())
    rho = spearmanr(d.I_poisson, d.I_mse).correlation
    diff = d.I_poisson - d.I_mse
    print(f"peak: Poisson {d.I_poisson.idxmax()}, MSE {d.I_mse.idxmax()}, "
          f"main pipeline {d.I_latent_paper.idxmax()}; depth-profile Spearman {rho:.2f}; "
          f"Poisson - MSE mean {diff.mean():+.3f}, range [{diff.min():+.3f}, {diff.max():+.3f}]")

    t = pd.concat([pd.read_csv(res / "poisson_recon_trajectory.csv"), d.reset_index()[COLS]])
    t = t.drop_duplicates(["layer", "epoch"]).sort_values(["layer", "epoch"])
    print("\ntrajectory:")
    print(t[COLS].round(3).to_string(index=False))
    full = [lay for lay in PANEL_LAYERS if set(t[t.layer == lay].epoch) >= {30, 100, 300, 900}]

    def rises(lay, col):
        return bool(np.all(np.diff(t[t.layer == lay].sort_values("epoch")[col]) > 0))

    both = [lay for lay in full if rises(lay, "I_poisson") and rises(lay, "I_mse")]
    rho_all = spearmanr(t.I_poisson, t.I_mse).correlation
    late = t[t.epoch >= 100]
    print(f"\nsublayers with all four epochs: {len(full)}; I(Z;Y) rises monotonically under both "
          f"losses in {len(both)} of {len(full)}")
    print(f"rank correlation Poisson vs MSE over all {len(t)} cells: {rho_all:.2f}")
    print(f"epoch 900: max |Poisson - MSE| = {diff.abs().max():.3f} bits")
    print(f"epochs >= 100: max |main pipeline - MSE replicate| = "
          f"{(late.I_latent_paper - late.I_mse).abs().max():.3f} bits")
    print(f"max |acc Poisson - acc MSE| = {(t.acc_poisson - t.acc_mse).abs().max():.3f}")


if __name__ == "__main__":
    main()
