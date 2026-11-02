"""Spike-count (Poisson) reconstruction control for the SNN relevance axis (needs raw activity).

Same autoencoder as 02_latent_mi.py (latent 16, hidden min(256, 2N), 50 epochs,
Adam 1e-3, batch 128, classification weight 1, stratified 10 000-image training
subsample, KSG k=5 on the 10 000 test images). The Poisson arm replaces the
reconstruction MSE(recon, T) with PoissonNLL(log_rate=recon, counts=T), i.e. the
decoder output is read as a log expected spike count. Both arms are trained from the
same torch seed, so the MSE arm is a matched replicate of the main pipeline.
I(Z;Y) is reported unclipped (negative values are kept).

The paper's grid (Supplementary "Spike-count reconstruction loss") is the five panel
sublayers at epochs 30, 100, 300 and 900; it was run in pieces:
  --epochs 900             --out poisson_recon_e900.csv
  --epochs 30 100 300      --out poisson_recon_trajectory.csv
(the cached files merge the pieces). 23_poisson_summary.py turns them into the
numbers quoted in the paper. One cell takes about 1.5-3 min on 3-4 CPU threads
(up to 17 min when the training-split file loads slowly).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from snn_common import (
    AE_BATCH,
    AE_EPOCHS,
    AE_LR,
    CLASS_WEIGHT,
    HIDDEN_DIM,
    LATENT_DIM,
    N_CLASSES,
    PANEL_LAYERS,
    load_activity,
    load_labels,
    require_activity,
    stratified_subsample,
)

from ib.paths import results_dir

TORCH_SEED = 0
KSG_SEED = 7


def ksg_unclipped(z, y, k=5):
    from driada.information.ksg import add_noise, nonparam_entropy_c

    z = add_noise(z.astype(np.float64))
    hx = nonparam_entropy_c(z, k=k, base=2.0)
    hc = sum((y == c).mean() * nonparam_entropy_c(z[y == c], k=k, base=2.0) for c in np.unique(y))
    return hx - hc


def train(act_train, lab_train, act_test, lab_test, loss_kind, ae_epochs=AE_EPOCHS):
    from driada.dim_reduction.flexible_ae import ModularAutoencoder

    torch.manual_seed(TORCH_SEED)
    n_in = act_train.shape[1]
    model = ModularAutoencoder(
        input_dim=n_in, latent_dim=LATENT_DIM, hidden_dim=min(HIDDEN_DIM, n_in * 2),
        loss_components=[{"name": "reconstruction", "weight": 1.0},
                         {"name": "classification", "weight": CLASS_WEIGHT,
                          "num_classes": N_CLASSES}])
    if loss_kind == "poisson":
        model.losses[0].criterion = nn.PoissonNLLLoss(log_input=True, full=False)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(act_train, dtype=torch.float32),
                                       torch.tensor(lab_train, dtype=torch.long)),
        batch_size=AE_BATCH, shuffle=True)
    model.train()
    for _ in range(ae_epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss, _ = model.compute_loss(bx, labels=by)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.tensor(act_test, dtype=torch.float32))
        acc = float((model.losses[1].classifier(z).argmax(1).numpy() == lab_test).mean())
    return z.numpy(), acc


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", nargs="+", default=PANEL_LAYERS)
    p.add_argument("--epochs", nargs="+", type=int, default=[900])
    p.add_argument("--out", default="poisson_recon_e900.csv")
    p.add_argument("--budget-s", type=float, default=3600,
                   help="stop cleanly before starting a cell once this wall time is spent")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--ae-epochs", type=int, default=AE_EPOCHS, help="for quick tests only")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    require_activity()

    t0 = time.time()
    y_tr_all, y_te = load_labels("train"), load_labels("test")
    ref = np.load(results_dir("snn") / "latent_mi.npz")
    ref_layers, ref_ep = list(ref["layers"]), list(ref["epochs"])
    out = results_dir("snn") / args.out
    rows = []
    for ep in args.epochs:
        for lay in args.layers:
            if time.time() - t0 > args.budget_s:
                print(f"[STOP] wall budget reached after {len(rows)} cells", flush=True)
                return
            c0 = time.time()
            a_te = load_activity(ep, lay, "test")
            a_tr, l_tr = stratified_subsample(load_activity(ep, lay, "train"), y_tr_all)
            t_load = time.time() - c0
            row = dict(layer=lay, epoch=ep, I_latent_paper=float(
                ref["I_latent"][ref_layers.index(lay), ref_ep.index(ep)]))
            for kind in ["mse", "poisson"]:
                z, acc = train(a_tr, l_tr, a_te, y_te, kind, args.ae_epochs)
                np.random.seed(KSG_SEED)
                row[f"I_{kind}"] = ksg_unclipped(z, y_te)
                row[f"acc_{kind}"] = acc
            row["t_load_s"] = t_load
            row["t_cell_s"] = time.time() - c0
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
            print(f"{lay} e{ep}: I_mse={row['I_mse']:.3f} I_pois={row['I_poisson']:.3f} "
                  f"(main pipeline {row['I_latent_paper']:.3f}) acc {row['acc_mse']:.3f}/"
                  f"{row['acc_poisson']:.3f} cell {row['t_cell_s']:.0f}s", flush=True)
    print(f"[OK] {len(rows)} cells in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
