"""Small per-session facts taken from the probe files, so that the statistics and
figures run without them.

Per session (lambda = 1 files; frames, split and X/Y are the same in both regimes):
  n_matched            neurons matched on all four days (constant per mouse)
  n_valid, n_train, n_test
  n_bins_visited_test  occupied 10 x 10 bins in the test fold (coverage table)
  X_dim_kept           non-constant X_ego components over all frames: the
                       "effective dimensionality" of the session-exclusion table
                       (an object distance is constant when its zone was never entered)
  H_X, H_Y_cont        Gaussian entropy (bits) of X_ego (kept components) and of (x, y)
  H_Y_bin10            entropy of 10 x 10 occupancy over all frames

Output: results/hippocampus/session_summary.csv
Usage:  python hippocampus/10_session_summary.py
"""
import argparse

import numpy as np
import pandas as pd
from hippo_common import (
    add_session_args,
    bin_2d,
    h_discrete,
    list_probe_files,
    npz_day,
    parse_sessions,
    probes_dir,
    table_path,
)


def gaussian_entropy(A):
    """0.5 log2 det(2 pi e Cov) over the non-constant columns; returns (H, n_kept)."""
    kept = A.std(0) > 1e-8
    cov = np.atleast_2d(np.cov(A[:, kept].T))
    _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
    return 0.5 * logdet / np.log(2), int(kept.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_session_args(ap)
    a = ap.parse_args()
    rows = []
    for f in list_probe_files(probes_dir("lambda1", a.probes_dir), parse_sessions(a.sessions)):
        d = np.load(f)
        X = np.vstack([d["X_tr"], d["X_te"]]).astype(float)
        Y = np.vstack([d["Y_tr"], d["Y_te"]]).astype(float)
        h_x, x_dim = gaussian_entropy(X)
        h_y, _ = gaussian_entropy(Y)
        rows.append(dict(mouse=str(d["mouse"]), day=npz_day(d), n_matched=int(d["n_matched"]),
                         n_valid=int(d["n_valid"]), n_train=int(d["n_train"]),
                         n_test=int(d["n_valid"]) - int(d["n_train"]),
                         n_bins_visited_test=int(len(np.unique(bin_2d(d["Y_te"])))),
                         X_dim_kept=x_dim, H_X=h_x, H_Y_cont=h_y, H_Y_bin10=h_discrete(bin_2d(Y))))
    df = pd.DataFrame(rows)
    out = a.out or table_path("session_summary.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} sessions -> {out}")


if __name__ == "__main__":
    main()
