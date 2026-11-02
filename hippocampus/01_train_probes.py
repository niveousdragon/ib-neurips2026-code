"""Raw sessions -> per-session probe latents (needs raw data).

For every session (mouse, day) of the matched cohort:
  - keep the neurons matched on all four days of the mouse;
  - frames with finite X_ego and position, every 5th of them (4 Hz);
  - contiguous split: first 70 % train, last 30 % test (random folds would
    leak through calcium autocorrelation);
  - per-neuron z-score with train statistics;
  - train the probe autoencoder T -> Z (d_Z = 16) with a position head,
    loss = reconstruction MSE + lambda * position MSE;
  - repeat on five roll-shuffled copies of the calcium (seed 42 + 1000 (k + 1));
  - PCA-30 of the z-scored activity (real and shuffled), fit on the train fold.

Run twice for the paper: --regime lambda1 (primary) and --regime lambda0
(reconstruction only). Output, per session:
  results/hippocampus/probes/<regime>/probe_<mouse>_<day>D.npz
with Z_tr, Z_te, Z_tr_shuf, Z_te_shuf, T_tr_pca, T_te_pca, T_tr_pca_shuf,
T_te_pca_shuf, X_tr, X_te, Y_tr, Y_te, valid_idx, matched_idx, n_matched,
n_valid, n_train, n_pca, recon_mse, recon_mse_shuf, loss histories.
Existing files are skipped (resume).

Usage:
  python hippocampus/01_train_probes.py --regime lambda1
  python hippocampus/01_train_probes.py --regime lambda0 --sessions H01:1 --epochs 2
"""
import argparse
import time

import numpy as np
from hippo_common import (
    REGIMES,
    build_features,
    list_mice,
    load_config,
    load_matching,
    parse_sessions,
    probe_file,
    probes_dir,
    roll_shuffle_calcium,
    session_path,
    valid_frames,
    zscore_split,
)
from sklearn.decomposition import PCA

from ib.ae_predict import train_ae_predict


def train_block(T_tr_z, T_te_z, Y_tr, Y_te, cfg, lam, n_pca, seed, epochs):
    """Probe AE + PCA on one (real or shuffled) activity matrix."""
    Z_tr, Z_te, _, recon_mse, meta = train_ae_predict(
        T_tr_z, Y_tr.astype(np.float32), T_te_z, Y_te.astype(np.float32),
        latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"],
        epochs=epochs, batch_size=cfg["batch_size"], lr=cfg["lr"],
        lambda_predict=lam, seed=seed, predict_hidden=cfg["predict_hidden"],
    )
    pca = PCA(n_components=n_pca, random_state=seed).fit(T_tr_z)
    return (Z_tr.astype(np.float32), Z_te.astype(np.float32),
            pca.transform(T_tr_z).astype(np.float32), pca.transform(T_te_z).astype(np.float32),
            float(recon_mse), meta)


def run_session(cfg, mouse, day, regime, out_path, epochs):
    lam = REGIMES[regime]
    sr = float(cfg["sampling_rate_hz"])
    matched_idx = load_matching(cfg, mouse)[day]
    sess_file = session_path(cfg, mouse, day)
    calcium = np.load(sess_file)["calcium"][matched_idx, :]
    X_full, Y_full, _ = build_features(sess_file, sr)
    valid_idx = valid_frames(X_full, Y_full, int(cfg["downsample"]))
    n = len(valid_idx)
    n_tr = int(cfg["train_frac"] * n)
    n_matched = len(matched_idx)
    print(f"  {mouse}/{day}D {regime}: {n_matched} matched cells, "
          f"{n} frames ({n_tr} train / {n - n_tr} test)", flush=True)

    X = X_full[valid_idx].astype(np.float32)
    Y = Y_full[valid_idx].astype(np.float32)
    Y_tr, Y_te = Y[:n_tr], Y[n_tr:]
    n_pca = min(int(cfg["pca_dim"]), n_matched)
    seed = int(cfg["seed"])

    T_tr_z, T_te_z = zscore_split(calcium[:, valid_idx].T, n_tr)
    Z_tr, Z_te, P_tr, P_te, mse, meta = train_block(
        T_tr_z, T_te_z, Y_tr, Y_te, cfg, lam, n_pca, seed, epochs)

    K = int(cfg["n_shuffles"])
    dz = int(cfg["latent_dim"])
    Zs_tr = np.zeros((K, n_tr, dz), np.float32)
    Zs_te = np.zeros((K, n - n_tr, dz), np.float32)
    Ps_tr = np.zeros((K, n_tr, n_pca), np.float32)
    Ps_te = np.zeros((K, n - n_tr, n_pca), np.float32)
    mse_s = np.zeros(K, np.float32)
    for k in range(K):
        seed_k = seed + 1000 * (k + 1)
        cal_s = roll_shuffle_calcium(calcium, seed_k)
        Ts_tr, Ts_te = zscore_split(cal_s[:, valid_idx].T, n_tr)
        Zs_tr[k], Zs_te[k], Ps_tr[k], Ps_te[k], mse_s[k], _ = train_block(
            Ts_tr, Ts_te, Y_tr, Y_te, cfg, lam, n_pca, seed_k, epochs)

    np.savez(
        out_path, mouse=mouse, day=f"{day}D", n_matched=n_matched,
        n_valid=n, n_train=n_tr, n_pca=n_pca,
        valid_idx=valid_idx.astype(np.int32), matched_idx=matched_idx.astype(np.int32),
        Z_tr=Z_tr, Z_te=Z_te, Z_tr_shuf=Zs_tr, Z_te_shuf=Zs_te,
        T_tr_pca=P_tr, T_te_pca=P_te, T_tr_pca_shuf=Ps_tr, T_te_pca_shuf=Ps_te,
        X_tr=X[:n_tr], X_te=X[n_tr:], Y_tr=Y_tr, Y_te=Y_te,
        recon_mse=mse, recon_mse_shuf=mse_s,
        loss_history_total=np.array(meta["loss_history"]["total"]),
        loss_history_recon=np.array(meta["loss_history"]["recon"]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--sessions", default=None,
                    help="restrict to sessions, e.g. H01:1,H01:2 (default: all mice x days 1-4)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override AE epochs (smoke tests only; paper: config ae_epochs)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    epochs = a.epochs or int(cfg["ae_epochs"])
    todo = parse_sessions(a.sessions) or [(m, d) for m in list_mice(cfg) for d in cfg["days"]]
    probes_dir(a.regime)
    t0 = time.time()
    n_done = 0
    for mouse, day in todo:
        out = probe_file(a.regime, mouse, day)
        if out.exists():
            print(f"  skip {mouse}/{day}D (exists)")
            continue
        ts = time.time()
        run_session(cfg, mouse, day, a.regime, out, epochs)
        n_done += 1
        print(f"  saved {out.name}  {time.time() - ts:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[OK] {n_done} sessions trained, {len(todo) - n_done} skipped")


if __name__ == "__main__":
    main()
