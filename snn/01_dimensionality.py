"""Per-sublayer dimensionality of SNN spike counts across training (needs raw activity).

For every (sublayer, epoch) on the 10 000 CIFAR-10 test images:
  - effective dimension (participation ratio, q=2, finite-sample spectrum correction);
  - intrinsic dimension, two-nearest-neighbour (TWO-NN) estimator, on a fixed
    5 000-image subsample.

Output: <IB_RESULTS>/snn/dimensionality.npz with ed, id_nn (17 x 26), layers, epochs.
The intrinsic dimension (id_nn) is the "ID" of the paper's Figure 1 (bottom left).
"""
import argparse
import gc

import numpy as np
from snn_common import EPOCHS, LAYERS, load_activity, require_activity

from ib.paths import results_dir

ID_SUBSAMPLE = 5000
ID_SEED = 42


def compute(layers, epochs):
    from driada.dimensionality import eff_dim, nn_dimension

    ed = np.full((len(layers), len(epochs)), np.nan)
    id_nn = np.full((len(layers), len(epochs)), np.nan)
    for j, epoch in enumerate(epochs):
        print(f"  epoch {epoch}...", end=" ", flush=True)
        for i, layer in enumerate(layers):
            act = load_activity(epoch, layer)
            if act is None:
                continue
            try:
                ed[i, j] = eff_dim(act, enable_correction=True, q=2)
            except Exception as e:
                print(f"ED error {layer} e{epoch}: {e}")
            try:
                n_sub = min(ID_SUBSAMPLE, act.shape[0])
                idx = np.random.RandomState(ID_SEED).choice(act.shape[0], n_sub, replace=False)
                id_nn[i, j] = nn_dimension(act[idx], k=2)
            except Exception as e:
                print(f"ID error {layer} e{epoch}: {e}")
            del act
            gc.collect()
        print("done")
    return ed, id_nn


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", nargs="+", default=LAYERS)
    p.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--out", default="dimensionality.npz")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = results_dir("snn") / args.out
    if out.exists() and not args.overwrite:
        d = np.load(out)
        ed, id_nn, layers = d["ed"], d["id_nn"], list(d["layers"])
        print(f"Loaded {out} (pass --overwrite to recompute)")
    else:
        require_activity()
        layers = args.layers
        ed, id_nn = compute(layers, args.epochs)
        np.savez(out, ed=ed, id_nn=id_nn, layers=layers, epochs=args.epochs)
        print(f"Saved {out}")
    print(f"\n{'sublayer':14s} {'ED final':>9s} {'ID final':>9s}")
    for i, layer in enumerate(layers):
        print(f"{layer:14s} {ed[i, -1]:9.2f} {id_nn[i, -1]:9.2f}")


if __name__ == "__main__":
    main()
