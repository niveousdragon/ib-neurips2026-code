"""Earlier 20 Hz all-neuron pipeline (needs raw data). It produced the latents behind
the GLM heatmap figure (Supplementary Fig. "Per-session GLM cohort summary", 64 sessions,
lambda = 0) and the train/test split of the raw-calcium Ridge table; the paper's
information-plane analysis uses 01_train_probes.py instead.

Per session: all recorded neurons at 20 Hz; X = (x, y, v_x, v_y) with
v = speed * (cos, sin)(head direction); Y = position Delta_t later. Valid frames are
those with finite X and finite Y for every Delta_t in --deltas (so the AE and the 70/30
contiguous split are shared across Delta_t; the paper's runs used 1,2,3,4,5 s). One
probe AE (reconstruction only, lambda = 0) and PCA-30 on the z-scored activity;
Gaussian-copula MI and Ridge scores per Delta_t.

--shuffled replaces the calcium by driada's roll-based shuffle of every neuron
(Experiment.get_multicell_shuffled_calcium(method="roll_based")): each trace is rolled
by its own random shift in [3 t_off, n_frames - 3 t_off). driada draws these shifts
from the global numpy generator; --shuffle-seed seeds it (the original runs did not,
so their shuffles cannot be regenerated bit for bit).

Output: results/hippocampus/legacy_20hz/pilot_<mouse>_<day>D_dt<k>[_shuffled].npz
Usage:
  python hippocampus/legacy_01_pilot_20hz.py --mouse H01 --day 1
  python hippocampus/legacy_01_pilot_20hz.py --mouse H01 --day 1 --shuffled --shuffle-seed 0
"""
import argparse

import numpy as np
from hippo_common import load_config, session_path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from ib.ae_predict import train_ae_predict
from ib.mi_estimators import cmi_gcmi_ccc, mi_gcmi_cc
from ib.paths import results_dir

LAMBDA = 0.0


def build_X(x, y, speed, head_direction):
    """(x, y, v_x, v_y): Cartesian velocity keeps the Gaussian copula well conditioned
    (unlike (cos, sin) of the heading, which lies on a circle)."""
    return np.column_stack([x, y, speed * np.cos(head_direction),
                            speed * np.sin(head_direction)]).astype(np.float32)


def build_Y(x, y, delta_frames):
    """Position delta_frames later; the last delta_frames rows are NaN."""
    n = len(x)
    Y = np.full((n, 2), np.nan, dtype=np.float32)
    if delta_frames < n:
        Y[:n - delta_frames, 0] = x[delta_frames:]
        Y[:n - delta_frames, 1] = y[delta_frames:]
    return Y


def common_valid_frames(sess, sampling_rate, deltas):
    """Frames with finite X and finite future position for every Delta_t (seconds)."""
    X = build_X(sess["x"], sess["y"], sess["speed"], sess["headdirection"])
    idx = None
    for dt in deltas:
        Y = build_Y(sess["x"], sess["y"], int(round(dt * sampling_rate)))
        ok = np.where(np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y), 1))[0]
        idx = ok if idx is None else np.intersect1d(idx, ok)
    return X, idx


def ridge_r2(x_tr, y_tr, x_te, y_te):
    return float(Ridge(alpha=1.0).fit(x_tr, y_tr).score(x_te, y_te))


def shuffled_calcium(path, mouse, day, seed):
    """Roll-based shuffle through driada's public Experiment API."""
    from driada.experiment import load_exp_from_aligned_data

    # Metadata entries of the session file are not behavioural variables.
    data = {k: v for k, v in np.load(path, allow_pickle=True).items()
            if k not in ("_metadata", "_sync_info")}
    exp = load_exp_from_aligned_data(
        data_source="IABS", exp_params={"track": "NOF", "animal_id": mouse, "session": f"{day}D"},
        data=data, verbose=False, enable_parallelization=False)
    if seed is not None:
        np.random.seed(seed)
    return exp.get_multicell_shuffled_calcium(method="roll_based", return_array=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mouse", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--deltas", default="1,2,3,4,5", help="Delta_t values in seconds")
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override AE epochs (smoke tests)")
    a = ap.parse_args()

    cfg = load_config()
    sr = float(cfg["sampling_rate_hz"])
    deltas = [float(v) for v in a.deltas.split(",")]
    path = session_path(cfg, a.mouse, a.day)
    sess = np.load(path)
    calcium = sess["calcium"]
    if a.shuffled:
        calcium = shuffled_calcium(path, a.mouse, a.day, a.shuffle_seed)
    X_full, idx = common_valid_frames(sess, sr, deltas)
    n = len(idx)
    n_tr = int(cfg["train_frac"] * n)
    X = X_full[idx]
    T = calcium[:, idx].T
    X_tr, X_te = X[:n_tr], X[n_tr:]
    T_mean = T[:n_tr].mean(0, keepdims=True)
    T_std = T[:n_tr].std(0, keepdims=True)
    T_std[T_std < 1e-8] = 1.0
    T_tr_z = ((T[:n_tr] - T_mean) / T_std).astype(np.float32)
    T_te_z = ((T[n_tr:] - T_mean) / T_std).astype(np.float32)
    print(f"{a.mouse}/{a.day}D: {calcium.shape[0]} neurons, {n} frames ({n_tr} train)")

    Y_ae = build_Y(sess["x"], sess["y"], int(round(deltas[0] * sr)))[idx]
    Z_tr, Z_te, _, recon_mse, meta = train_ae_predict(
        T_tr_z, Y_ae[:n_tr], T_te_z, Y_ae[n_tr:],
        latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"],
        epochs=a.epochs or cfg["ae_epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        lambda_predict=LAMBDA, seed=cfg["seed"], predict_hidden=cfg["predict_hidden"])
    n_pca = min(int(cfg["pca_dim"]), T_tr_z.shape[1])
    pca = PCA(n_components=n_pca, random_state=cfg["seed"]).fit(T_tr_z)
    P_tr = pca.transform(T_tr_z).astype(np.float32)
    P_te = pca.transform(T_te_z).astype(np.float32)
    I_xz = mi_gcmi_cc(X_te, Z_te)
    r2_zx = ridge_r2(Z_tr, X_tr, Z_te, X_te)

    out_dir = results_dir("hippocampus", "legacy_20hz")
    tag = "_shuffled" if a.shuffled else ""
    for dt in deltas:
        df = int(round(dt * sr))
        Y = build_Y(sess["x"], sess["y"], df)[idx]
        Y_tr, Y_te = Y[:n_tr], Y[n_tr:]
        dt_tag = f"dt{int(dt)}" if dt == int(dt) else f"dt{str(dt).replace('.', 'p')}"
        np.savez(out_dir / f"pilot_{a.mouse}_{a.day}D_{dt_tag}{tag}.npz",
                 mouse=a.mouse, day=f"{a.day}D", delta_t_sec=dt, delta_frames=df,
                 n_neurons=calcium.shape[0], n_valid=n, n_train=n_tr,
                 valid_idx=idx.astype(np.int32), n_pca=n_pca, recon_mse=recon_mse,
                 I_xz_gcmi=I_xz, I_zy_gcmi=mi_gcmi_cc(Z_te, Y_te),
                 I_xy_gcmi=mi_gcmi_cc(X_te, Y_te), I_ty_gcmi=mi_gcmi_cc(P_te, Y_te),
                 I_ty_given_x=cmi_gcmi_ccc(P_te, Y_te, X_te),
                 I_zy_given_x=cmi_gcmi_ccc(Z_te, Y_te, X_te),
                 r2_zx=r2_zx, r2_zy=ridge_r2(Z_tr, Y_tr, Z_te, Y_te),
                 r2_physics=ridge_r2(X_tr, Y_tr, X_te, Y_te),
                 Z_tr=Z_tr.astype(np.float32), Z_te=Z_te.astype(np.float32),
                 T_tr_pca=P_tr, T_te_pca=P_te, X_tr=X_tr, X_te=X_te, Y_tr=Y_tr, Y_te=Y_te,
                 loss_history_total=np.array(meta["loss_history"]["total"]),
                 loss_history_recon=np.array(meta["loss_history"]["recon"]),
                 loss_history_predict=np.array(meta["loss_history"]["predict"]))
    print(f"[OK] {len(deltas)} files in {out_dir}")


if __name__ == "__main__":
    main()
