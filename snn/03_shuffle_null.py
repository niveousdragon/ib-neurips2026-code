"""Label-permutation null for the SNN relevance axis (needs raw activity, or cached latents).

For each (sublayer, epoch):
  - Z_test is taken from latent_zs.npz if present; otherwise the autoencoder of
    02_latent_mi.py is trained (same settings) and Z_test is added to that cache;
  - I_real = KSG I(Z_test; Y), and I_shuf = mean over 5 label permutations of
    KSG I(Z_test; perm(Y)) (k=5, bits, clipped at 0 by the estimator).

This run trains its own autoencoders, so I_real differs from I_latent of
latent_mi.npz by autoencoder training noise. The paper uses this file for the
statement that the permuted estimate sat at the estimator's floor of zero in every
cell; 04_unclipped_shuffle_null.py re-estimates the null without the clip.

Outputs: <IB_RESULTS>/snn/latent_zs.npz (Z_test cache, about 280 MB, not shipped),
         <IB_RESULTS>/snn/latent_mi_with_shuffle.npz.
Re-running resumes: cells with a finite I_shuf are skipped.
"""
import argparse
import gc
import time

import numpy as np
from sklearn.metrics import mutual_info_score
from snn_common import (
    EPOCHS,
    LAYERS,
    LN2,
    gaussian_entropy,
    load_activity,
    load_labels,
    require_activity,
    stratified_subsample,
    train_ae_classifier,
)

from ib.paths import results_dir

N_PERMS = 5
KSG_K = 5
RNG_SEED = 42
SAVE_EVERY = 5
KEYS = ("I_real", "I_shuf", "I_shuf_std", "I_ind", "H_gauss", "acc_z", "acc_full", "mse")


def log(msg):
    print(msg, flush=True)


def get_or_train_z(i, j, layer, epoch, labels_train, labels_test, z_dict):
    """Return (Z_test, acc_z, mse, trained). Cached latents are reused as is."""
    key = f"L{i}_e{j}"
    if key in z_dict:
        return z_dict[key], np.nan, np.nan, False
    act_test = load_activity(epoch, layer, "test")
    act_train_full = load_activity(epoch, layer, "train")
    if act_test is None or act_train_full is None:
        return None, np.nan, np.nan, False
    act_train, lab_train = stratified_subsample(act_train_full, labels_train)
    del act_train_full
    z_test, az, rm = train_ae_classifier(act_train, lab_train, act_test, labels_test)
    z_test = z_test.astype(np.float32)
    z_dict[key] = z_test
    del act_test, act_train
    gc.collect()
    return z_test, az, rm, True


def shuffle_null(z_test, labels_test, rng):
    from driada.information.ksg import nonparam_mi_cd

    vals = np.array([nonparam_mi_cd(z_test, rng.permutation(labels_test), k=KSG_K, base=2.0)
                     for _ in range(N_PERMS)])
    return float(vals.mean()), float(vals.std(ddof=1))


def main():
    from driada.information.ksg import nonparam_mi_cd

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    require_activity()
    out_dir = results_dir("snn")
    zs_path, out_path = out_dir / "latent_zs.npz", out_dir / "latent_mi_with_shuffle.npz"
    t0 = time.time()
    n_layers, n_epochs = len(LAYERS), len(EPOCHS)
    labels_train, labels_test = load_labels("train"), load_labels("test")

    z_dict = {}
    if zs_path.exists():
        d = np.load(zs_path)
        z_dict = {k: d[k] for k in d.files if k not in ("layers", "epochs")}
    log(f"Z cache: {len(z_dict)} latents")
    if out_path.exists():
        d = np.load(out_path)
        res = {k: d[k] for k in KEYS}
    else:
        res = {k: np.full((n_layers, n_epochs), np.nan) for k in KEYS}

    def save():
        np.savez(zs_path, **z_dict, layers=LAYERS, epochs=EPOCHS)
        np.savez(out_path, **res, layers=LAYERS, epochs=EPOCHS)

    rng = np.random.default_rng(RNG_SEED)
    n_done = int(np.isfinite(res["I_shuf"]).sum())
    new_since_save = 0
    for j, epoch in enumerate(EPOCHS):
        for i, layer in enumerate(LAYERS):
            if np.isfinite(res["I_shuf"][i, j]):
                continue
            c0 = time.time()
            act_test = load_activity(epoch, layer, "test")
            if act_test is None:
                continue
            res["I_ind"][i, j] = np.array([mutual_info_score(labels_test, act_test[:, k]) / LN2
                                           for k in range(act_test.shape[1])]).sum()
            res["H_gauss"][i, j] = gaussian_entropy(act_test)
            del act_test
            try:
                z_test, az, rm, trained = get_or_train_z(i, j, layer, epoch, labels_train,
                                                         labels_test, z_dict)
            except Exception as e:
                log(f"  [e={epoch} {layer}] AE training failed: {e}")
                continue
            if z_test is None:
                continue
            if trained:
                res["acc_z"][i, j], res["mse"][i, j] = az, rm
            try:
                res["I_real"][i, j] = nonparam_mi_cd(z_test, labels_test, k=KSG_K, base=2.0)
                m, s = shuffle_null(z_test, labels_test, rng)
                res["I_shuf"][i, j], res["I_shuf_std"][i, j] = m, s
            except Exception as e:
                log(f"  [e={epoch} {layer}] KSG failed: {e}")
                continue
            n_done += 1
            new_since_save += 1
            log(f"  [e={epoch:>3} {layer:<14}] I_real={res['I_real'][i, j]:.3f} "
                f"I_shuf={m:.3f}+/-{s:.3f} trained={trained} {time.time() - c0:.0f}s "
                f"({n_done}/{n_layers * n_epochs})")
            if new_since_save >= SAVE_EVERY:
                save()
                new_since_save = 0
    save()
    log(f"Done in {(time.time() - t0) / 60:.1f} min; saved {out_path} and {zs_path}")
    log(f"I_shuf == 0 in {(res['I_shuf'] == 0).sum()} of {np.isfinite(res['I_shuf']).sum()} cells")


if __name__ == "__main__":
    main()
