"""What the probe latent encodes: per-session r^2 of each latent dimension with each
behavioural variable (Supplementary "Per-session GLM: what does Z encode?").

Per session and regime, on the test fold: squared Pearson correlation of each of
the 16 latent dimensions with x, y, d_obj1..4, speed and omega. Latent dimensions
are greedily ordered per variable (variable j takes its best remaining dimension,
in this variable order; leftovers by total r^2), separately for the real latent
and for each of the five shuffle latents; the shuffle matrix is the mean of the
five ordered matrices.

Output: results/hippocampus/glm_r2_<regime>.csv (long: mouse, day, rank, predictor,
        r2_real, r2_shuf); the table is computed by stats_glm.py.
Usage:  python hippocampus/09_glm_latents.py --regime lambda1
"""
import argparse

import numpy as np
import pandas as pd
from hippo_common import (
    REGIMES,
    add_session_args,
    list_probe_files,
    npz_day,
    parse_sessions,
    probes_dir,
    table_path,
)

PREDICTORS = ["x", "y", "d_obj1", "d_obj2", "d_obj3", "d_obj4", "speed", "omega"]


def r2_matrix(Z, P):
    """(n_dims, n_predictors) squared Pearson correlations."""
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Pc = (P - P.mean(0)) / (P.std(0) + 1e-12)
    return (Zc.T @ Pc / len(Z)) ** 2


def greedy_order(m):
    order, used = [], set()
    for j in range(m.shape[1]):
        k = max((k for k in range(m.shape[0]) if k not in used), key=lambda k: m[k, j])
        order.append(k)
        used.add(k)
    rest = sorted((k for k in range(m.shape[0]) if k not in used), key=lambda k: -m[k].sum())
    return order + rest


def ordered(m):
    return m[greedy_order(m)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    add_session_args(ap)
    a = ap.parse_args()
    rows = []
    for f in list_probe_files(probes_dir(a.regime, a.probes_dir), parse_sessions(a.sessions)):
        d = np.load(f)
        P = np.column_stack([d["Y_te"], d["X_te"][:, 2:6], d["X_te"][:, 0:2]]).astype(float)
        real = ordered(r2_matrix(d["Z_te"].astype(float), P))
        shuf = np.mean([ordered(r2_matrix(d["Z_te_shuf"][k].astype(float), P))
                        for k in range(d["Z_te_shuf"].shape[0])], axis=0)
        for rank in range(real.shape[0]):
            for j, p in enumerate(PREDICTORS):
                rows.append(dict(mouse=str(d["mouse"]), day=npz_day(d), rank=rank + 1,
                                 predictor=p, r2_real=real[rank, j], r2_shuf=shuf[rank, j]))
    df = pd.DataFrame(rows)
    out = a.out or table_path(f"glm_r2_{a.regime}.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"[OK] {df[['mouse', 'day']].drop_duplicates().shape[0]} sessions -> {out}")


if __name__ == "__main__":
    main()
