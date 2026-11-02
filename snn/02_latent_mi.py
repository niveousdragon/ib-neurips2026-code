"""Relevance axis I(Z;Y) of the SNN via a classification autoencoder plus KSG (needs raw activity).

For each (sublayer, epoch):
  1. Train the autoencoder T(N) -> Z(16) -> T'(N), Z -> Y'(10) on a class-stratified
     subsample of 10 000 of the 50 000 CIFAR-10 training images (1 000 per class).
  2. Encode the 10 000 test images and estimate I(Z;Y) with KSG (k=5, bits; the
     estimator clips negative values at 0).
  3. Controls on the test images: accuracy of the head acc(Z->Y), logistic-regression
     accuracy from T itself acc(T->Y) (fitted on the same training subsample),
     reconstruction MSE.
  4. Also on the test images, independent of the autoencoder: H_gauss(T) (the input
     axis) and I_ind = sum_n I(t_n; Y) (plug-in MI of each channel's spike count).

The same script produces the robustness variants of the paper:
  default                                     -> latent_mi.npz          (main result)
  --latent-dim 8 --relevance-only --test-split-fallback --out latent_mi_dim8.npz
  --class-weight 0 --layers <5 panel sublayers> --relevance-only --out latent_mi_lambda0.npz
--relevance-only skips I_ind, H_gauss and acc(T->Y), which do not depend on the
autoencoder. The autoencoder training is not seeded, so a rerun reproduces the
cached values only up to training noise (see the notes for the size of it).
"""
import argparse
import gc
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from snn_common import (
    EPOCHS,
    LATENT_DIM,
    LAYERS,
    LN2,
    PANEL_LAYERS,
    gaussian_entropy,
    load_activity,
    load_labels,
    require_activity,
    short_name,
    stratified_subsample,
    train_ae_classifier,
)

from ib.paths import results_dir

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*lbfgs.*")


def per_neuron_mi_sum(act, labels):
    return np.array([mutual_info_score(labels, act[:, k]) / LN2
                     for k in range(act.shape[1])]).sum()


def baseline_accuracy(act_train, labels_train, act_test, labels_test):
    """Logistic-regression accuracy from T: fit on the training subsample, score on test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
        lr.fit(act_train, labels_train)
        return lr.score(act_test, labels_test)


def compute(args, labels_train, labels_test):
    from driada.information.ksg import nonparam_mi_cd

    shape = (len(args.layers), len(args.epochs))
    res = {k: np.full(shape, np.nan) for k in
           ("I_latent", "acc_z", "mse", "I_ind", "H_gauss", "acc_full")}
    z_dict = {}
    t0 = time.time()
    for j, epoch in enumerate(args.epochs):
        print(f"  epoch {epoch}...", flush=True)
        for i, layer in enumerate(args.layers):
            c0 = time.time()
            act_test = load_activity(epoch, layer, "test")
            if act_test is None:
                continue
            lab_test = labels_test
            if args.test_subsample:
                act_test, lab_test = act_test[:args.test_subsample], labels_test[:args.test_subsample]
            if not args.relevance_only:
                res["I_ind"][i, j] = per_neuron_mi_sum(act_test, lab_test)
                res["H_gauss"][i, j] = gaussian_entropy(act_test)

            try:
                act_train_full = load_activity(epoch, layer, "train")
            except Exception:
                # The d_Z = 8 run guarded the large training-split load (memory).
                if not args.test_split_fallback:
                    raise
                act_train_full = None
            if act_train_full is None:
                if not args.test_split_fallback:
                    continue
                # Fallback of the d_Z = 8 run: train on one half of the test images and
                # estimate on the other half.
                n = act_test.shape[0]
                perm = np.random.RandomState(42).permutation(n)
                act_train, lab_train = act_test[perm[:n // 2]], lab_test[perm[:n // 2]]
                act_eval, lab_eval = act_test[perm[n // 2:]], lab_test[perm[n // 2:]]
            else:
                act_train, lab_train = stratified_subsample(act_train_full, labels_train)
                del act_train_full
                act_eval, lab_eval = act_test, lab_test

            if not args.relevance_only:
                res["acc_full"][i, j] = baseline_accuracy(act_train, lab_train, act_eval, lab_eval)
            try:
                z_test, az, rm = train_ae_classifier(
                    act_train, lab_train, act_eval, lab_eval,
                    latent_dim=args.latent_dim, class_weight=args.class_weight,
                    ae_epochs=args.ae_epochs)
                res["acc_z"][i, j] = az
                res["mse"][i, j] = rm
                res["I_latent"][i, j] = nonparam_mi_cd(z_test, lab_eval, k=5, base=2.0)
                if args.save_latents:
                    z_dict[f"L{i}_e{j}"] = z_test.astype(np.float32)
            except Exception as e:
                print(f"    AE error {layer}: {e}")
            print(f"    {short_name(layer):10s} I(Z;Y)={res['I_latent'][i, j]:.3f} "
                  f"acc_z={res['acc_z'][i, j]:.3f} [{time.time() - c0:.0f}s, "
                  f"total {(time.time() - t0) / 60:.1f} min]", flush=True)
            del act_test, act_train
            gc.collect()
    return res, z_dict


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    p.add_argument("--class-weight", type=float, default=1.0,
                   help="weight lambda of the classification loss (0 = reconstruction only)")
    p.add_argument("--layers", nargs="+", default=LAYERS,
                   help=f"sublayers; 'panel' = the five figure sublayers {PANEL_LAYERS}")
    p.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--relevance-only", action="store_true",
                   help="compute only I(Z;Y), acc(Z->Y) and MSE")
    p.add_argument("--test-split-fallback", action="store_true",
                   help="if the training split cannot be loaded, train on half of the test set")
    p.add_argument("--save-latents", action="store_true",
                   help="also save Z_test of every cell to <out>_zs.npz")
    p.add_argument("--out", default="latent_mi.npz")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ae-epochs", type=int, default=50, help="for quick tests only")
    p.add_argument("--test-subsample", type=int, default=0,
                   help="use only the first N test images (quick tests only; 0 = all)")
    args = p.parse_args()
    if args.layers == ["panel"]:
        args.layers = PANEL_LAYERS

    out = results_dir("snn") / args.out
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to recompute. Summary of the cached file:")
        d = np.load(out)
        res = {k: d[k] for k in d.files if k not in ("layers", "epochs")}
        layers = list(d["layers"])
    else:
        require_activity()
        labels_train, labels_test = load_labels("train"), load_labels("test")
        print(f"layers {len(args.layers)}, epochs {args.epochs}, d_Z={args.latent_dim}, "
              f"lambda={args.class_weight}")
        res, z_dict = compute(args, labels_train, labels_test)
        if args.relevance_only:
            res = {k: res[k] for k in ("I_latent", "acc_z", "mse")}
        np.savez(out, **res, layers=args.layers, epochs=args.epochs,
                 latent_dim=args.latent_dim, class_weight=args.class_weight)
        print(f"Saved {out}")
        if args.save_latents:
            zs = out.with_name(out.stem + "_zs.npz")
            np.savez(zs, **z_dict, layers=args.layers, epochs=args.epochs)
            print(f"Saved {zs} ({len(z_dict)} latents)")
        layers = args.layers

    acc = res.get("acc_z", np.full_like(res["I_latent"], np.nan))
    print(f"\n{'sublayer':14s} {'I(Z;Y) last epoch':>18s} {'acc_z':>7s}")
    for i, layer in enumerate(layers):
        print(f"{layer:14s} {res['I_latent'][i, -1]:18.3f} {acc[i, -1]:7.3f}")


if __name__ == "__main__":
    main()
