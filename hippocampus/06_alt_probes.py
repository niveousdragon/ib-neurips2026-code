"""Alternative probes of the same activity (Supplementary "Alternative probes of the same activity").

Probes that involve no trained network, built from the saved per-session files
of the lambda = 1 run, real and each of the five shuffles alike:
  pca<k>  first k = 4, 8, 16, 30 of the saved PCA-30 of the z-scored activity
          (fit on the train fold);
  iso16   Isomap (10 neighbours, 16 components) fit on 800 random train frames
          of the PCA-30 activity.
Both axes use the paper's estimators (relevance: confusion-matrix MI of the
logistic decoder; retention: Gaussian-copula I(X_ego; Z)), bias-corrected by the
mean over the shuffles. Also stored: Gaussian-copula I(Z; Y | X_ego) and I(Z; Y)
on continuous position for the AE latent, and I(X_ego; Y).

Output: results/hippocampus/alt_probes.csv
Usage:  python hippocampus/06_alt_probes.py [--jobs 8]
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from hippo_common import (
    add_session_args,
    bin_2d,
    confusion_mi,
    jitter,
    list_probe_files,
    npz_day,
    parse_sessions,
    probes_dir,
    safe_gcmi,
    table_path,
)
from joblib import Parallel, delayed
from sklearn.manifold import Isomap

from ib.mi_estimators import cmi_gcmi_ccc

KS = [4, 8, 16, 30]


def isomap(T_tr, T_te, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T_tr), min(800, len(T_tr)), replace=False)
    iso = Isomap(n_neighbors=10, n_components=16).fit(T_tr[idx])
    return iso.transform(T_tr), iso.transform(T_te)


def one_session(path):
    warnings.filterwarnings("ignore")
    d = np.load(path)
    Y_tr, Y_te = d["Y_tr"].astype(np.float32), d["Y_te"].astype(np.float32)
    b_tr, b_te = bin_2d(Y_tr), bin_2d(Y_te)
    K = d["T_tr_pca_shuf"].shape[0]
    rng = np.random.default_rng(0)
    Xj = jitter(d["X_te"].astype(np.float32), rng)
    rec = dict(mouse=str(d["mouse"]), day=npz_day(d), n_matched=int(d["n_matched"]),
               n_test=len(Y_te), n_pca=int(d["n_pca"]))

    def rel(Z_tr, Z_te):
        return confusion_mi(Z_tr, b_tr, Z_te, b_te)[0]

    def axes(Z_tr, Z_te, Zs_tr, Zs_te):
        rel_c = rel(Z_tr, Z_te) - np.nanmean([rel(Zs_tr[k], Zs_te[k]) for k in range(K)])
        r2 = np.random.default_rng(1)
        ret_r = safe_gcmi(Xj, jitter(Z_te, r2))
        ret_s = np.nanmean([safe_gcmi(Xj, jitter(Zs_te[k], r2)) for k in range(K)])
        return rel_c, ret_r - ret_s

    P_tr, P_te = d["T_tr_pca"], d["T_te_pca"]
    Ps_tr, Ps_te = d["T_tr_pca_shuf"], d["T_te_pca_shuf"]
    for k in KS:
        kk = min(k, P_tr.shape[1])
        rec[f"rel_pca{k}"], rec[f"ret_pca{k}"] = axes(
            P_tr[:, :kk], P_te[:, :kk], Ps_tr[:, :, :kk], Ps_te[:, :, :kk])
    real = isomap(P_tr, P_te)
    shuf = [isomap(Ps_tr[k], Ps_te[k], seed=k) for k in range(K)]
    rec["rel_iso16"], rec["ret_iso16"] = axes(
        real[0], real[1], np.stack([s[0] for s in shuf]), np.stack([s[1] for s in shuf]))

    r3 = np.random.default_rng(2)
    Yj = jitter(Y_te, r3)
    Z = jitter(d["Z_te"], r3)
    try:
        c_r = float(cmi_gcmi_ccc(Z, Yj, Xj))
        c_s = np.nanmean([float(cmi_gcmi_ccc(jitter(d["Z_te_shuf"][k], r3), Yj, Xj))
                          for k in range(K)])
        u_r = safe_gcmi(Z, Yj)
        u_s = np.nanmean([safe_gcmi(jitter(d["Z_te_shuf"][k], r3), Yj) for k in range(K)])
    except Exception:
        c_r = c_s = u_r = u_s = np.nan
    rec.update(I_zy_given_x_corr=c_r - c_s, I_zy_cont_corr=u_r - u_s, I_xy=safe_gcmi(Xj, Yj))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1, help="parallel sessions")
    add_session_args(ap)
    a = ap.parse_args()
    files = list_probe_files(probes_dir("lambda1", a.probes_dir), parse_sessions(a.sessions))
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_session)(f) for f in files)
    df = pd.DataFrame(rows).sort_values(["mouse", "day"])
    out = a.out or table_path("alt_probes.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} sessions -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
