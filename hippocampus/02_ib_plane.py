"""Per-session information-plane coordinates from the saved probe latents.

Relevance  I(Z; Y_alloc): confusion-matrix MI of a decoder Z -> 10 x 10 position bin
           (train fold -> test fold); --decoder logreg is the paper's estimator,
           --decoder mlp the robustness check (one hidden layer of 32, alpha 0.01,
           early stopping).
Retention  I(X_ego; Z): Gaussian-copula MI on the test fold (after 1e-4 relative jitter).
Both are bias-corrected: real minus the mean over the five roll-shuffle latents.

Outputs (results/hippocampus/):
  ib_plane_<regime>.csv          --decoder logreg
  ib_plane_mlp_<regime>.csv      --decoder mlp
Usage:
  python hippocampus/02_ib_plane.py --regime lambda1
  python hippocampus/02_ib_plane.py --regime lambda0
  python hippocampus/02_ib_plane.py --regime lambda1 --decoder mlp
"""
import argparse
import time

import numpy as np
import pandas as pd
from hippo_common import (
    N_CLASSES,
    REGIMES,
    _decodable,
    add_session_args,
    bin_2d,
    confusion_mi,
    confusion_mi_from_predictions,
    jitter,
    list_probe_files,
    npz_day,
    parse_sessions,
    probes_dir,
    safe_gcmi,
    table_path,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def mlp_confusion_mi(Z_tr, bid_tr, Z_te, bid_te, hidden=(32,), alpha=0.01, seed=42):
    keep = _decodable(bid_tr, bid_te)
    if keep is None:
        return float("nan")
    sc = StandardScaler().fit(Z_tr)
    try:
        clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=200,
                            early_stopping=True, n_iter_no_change=10,
                            random_state=seed).fit(sc.transform(Z_tr), bid_tr)
    except Exception:
        return float("nan")
    return confusion_mi_from_predictions(clf.predict(sc.transform(Z_te[keep])), bid_te[keep])


def session_row(d, decoder, hidden, alpha):
    Z_tr = d["Z_tr"].astype(np.float32)
    Z_te = d["Z_te"].astype(np.float32)
    Zs_tr = d["Z_tr_shuf"].astype(np.float32)
    Zs_te = d["Z_te_shuf"].astype(np.float32)
    X_te = d["X_te"].astype(np.float32)
    b_tr = bin_2d(d["Y_tr"].astype(np.float32))
    b_te = bin_2d(d["Y_te"].astype(np.float32))
    K = Zs_tr.shape[0]

    if decoder == "logreg":
        rel, h_y = confusion_mi(Z_tr, b_tr, Z_te, b_te, N_CLASSES)
        rel_s = [confusion_mi(Zs_tr[k], b_tr, Zs_te[k], b_te, N_CLASSES)[0] for k in range(K)]
    else:
        rel = mlp_confusion_mi(Z_tr, b_tr, Z_te, b_te, hidden, alpha)
        rel_s = [mlp_confusion_mi(Zs_tr[k], b_tr, Zs_te[k], b_te, hidden, alpha) for k in range(K)]

    # One generator per session, drawn in this order: X, real Z, then each shuffle.
    rng = np.random.default_rng(0)
    Xj = jitter(X_te, rng)
    ret = safe_gcmi(Xj, jitter(Z_te, rng))
    ret_s = [safe_gcmi(Xj, jitter(Zs_te[k], rng)) for k in range(K)]

    row = {"mouse": str(d["mouse"]), "day": npz_day(d)}
    if decoder == "logreg":
        row["H_Y_bin"] = h_y
    row.update({
        "I_zy_bin_real": rel,
        "I_zy_bin_shuf_mean": float(np.nanmean(rel_s)),
        "I_zy_bin_shuf_std": float(np.nanstd(rel_s)),
        "I_zy_bin_corr": rel - float(np.nanmean(rel_s)),
        "I_xz_real": ret,
        "I_xz_shuf_mean": float(np.nanmean(ret_s)),
    })
    if decoder == "logreg":
        row["I_xz_shuf_std"] = float(np.nanstd(ret_s))
    row["I_xz_corr"] = ret - float(np.nanmean(ret_s))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--decoder", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--hidden", nargs="+", type=int, default=[32], help="MLP hidden layers")
    ap.add_argument("--alpha", type=float, default=0.01, help="MLP L2 penalty")
    add_session_args(ap)
    a = ap.parse_args()

    files = list_probe_files(probes_dir(a.regime, a.probes_dir), parse_sessions(a.sessions))
    t0 = time.time()
    rows = []
    for f in files:
        rows.append(session_row(np.load(f), a.decoder, tuple(a.hidden), a.alpha))
        r = rows[-1]
        print(f"  {r['mouse']} d{r['day']}: I(Z;Y) corr {r['I_zy_bin_corr']:+.3f}  "
              f"I(X;Z) corr {r['I_xz_corr']:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    if a.decoder == "logreg":
        # Coverage-normalised relevance; kept for completeness, not used in the paper.
        df["I_zy_bin_norm"] = df["I_zy_bin_corr"] / df["H_Y_bin"]
    name = f"ib_plane_{a.regime}.csv" if a.decoder == "logreg" else f"ib_plane_mlp_{a.regime}.csv"
    out = a.out or table_path(name)
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} sessions -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
