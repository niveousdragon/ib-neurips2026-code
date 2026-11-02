"""Target-conditioned relevance I(Z; Y_alloc | X_ego) with the relevance-axis decoder
(Supplementary "Target-conditioned relevance").

Chain rule: I(Z; Y | X) = I(Z, X; Y) - I(X; Y). Both terms are confusion-matrix MI
of the same logistic decoder onto the 10 x 10 bin, on the same split. Bias
correction replaces Z by each of its five shuffles while keeping X_ego real, so
the I(X; Y) term cancels in expectation: cmi_corr = I([Z, X]; Y) - mean_k I([Z_k, X]; Y).
Uses the lambda = 1 latents; no network is trained.

Output: results/hippocampus/conditional_relevance.csv
Usage:  python hippocampus/07_conditional_relevance.py [--jobs 8]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from hippo_common import (
    add_session_args,
    bin_2d,
    confusion_mi,
    list_probe_files,
    npz_day,
    parse_sessions,
    probes_dir,
    table_path,
)
from joblib import Parallel, delayed


def one_session(path):
    warnings.filterwarnings("ignore")
    d = np.load(path)
    b_tr, b_te = bin_2d(d["Y_tr"]), bin_2d(d["Y_te"])
    X_tr, X_te = d["X_tr"].astype(float), d["X_te"].astype(float)

    def mi(A_tr, A_te):
        return confusion_mi(A_tr, b_tr, A_te, b_te)[0]

    def cat(Z, X):
        return np.hstack([Z.astype(float), X])

    K = d["Z_tr_shuf"].shape[0]
    zx = mi(cat(d["Z_tr"], X_tr), cat(d["Z_te"], X_te))
    x = mi(X_tr, X_te)
    z = mi(d["Z_tr"], d["Z_te"])
    zx_s = np.nanmean([mi(cat(d["Z_tr_shuf"][k], X_tr), cat(d["Z_te_shuf"][k], X_te))
                       for k in range(K)])
    z_s = np.nanmean([mi(d["Z_tr_shuf"][k], d["Z_te_shuf"][k]) for k in range(K)])
    return dict(mouse=str(d["mouse"]), day=npz_day(d), I_x_y=x, I_zx_y=zx, I_zx_y_shuf=zx_s,
                I_z_y=z, I_z_y_shuf=z_s, cmi_corr=zx - zx_s, rel_corr=z - z_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=1, help="parallel sessions")
    add_session_args(ap)
    a = ap.parse_args()
    files = list_probe_files(probes_dir("lambda1", a.probes_dir), parse_sessions(a.sessions))
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_session)(f) for f in files)
    df = pd.DataFrame(rows).sort_values(["mouse", "day"])
    out = a.out or table_path("conditional_relevance.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} sessions -> {out}")


if __name__ == "__main__":
    main()
