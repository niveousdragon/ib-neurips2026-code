"""Why the retention axis uses the Gaussian copula: KSG vs GCMI for I(X_ego; Z)
(Supplementary "Decoder family for the relevance axis", last part).

On the 16 day-1 sessions of the lambda = 1 run (test fold, X_ego 6-D, Z 16-D):
Gaussian-copula MI and KSG (k = 5 and k = 23, driada nonparam_mi_cc with its
local non-uniformity correction, alpha="auto"), each also on Z with its rows
permuted. driada switches the correction on only when k exceeds the joint
dimension (6 + 16 = 22), hence k = 23. One random generator over all sessions,
in file order.

Output: results/hippocampus/ksg_vs_gcmi.csv
Usage:  python hippocampus/08_ksg_vs_gcmi.py
"""
import argparse

import numpy as np
import pandas as pd
from driada.information.ksg import nonparam_mi_cc
from hippo_common import add_session_args, list_probe_files, probes_dir, table_path

from ib.mi_estimators import mi_gcmi_cc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_session_args(ap)
    a = ap.parse_args()
    files = [f for f in list_probe_files(probes_dir("lambda1", a.probes_dir))
             if f.stem.endswith("_1D")]
    if a.sessions:
        keep = {m for m, _ in (s.split(":") for s in a.sessions.split(","))}
        files = [f for f in files if f.stem.split("_")[-2] in keep]
    rng = np.random.default_rng(0)
    rows = []
    for f in files:
        d = np.load(f)
        X = d["X_te"].astype(float)
        Z = d["Z_te"].astype(float)
        X = X + 1e-4 * X.std(0) * rng.standard_normal(X.shape)
        Z = Z + 1e-4 * Z.std(0) * rng.standard_normal(Z.shape)
        Zp = Z[rng.permutation(len(Z))]
        r = dict(mouse=str(d["mouse"]), n=len(X), dX=X.shape[1], dZ=Z.shape[1],
                 gcmi=float(mi_gcmi_cc(X, Z)), gcmi_perm=float(mi_gcmi_cc(X, Zp)))
        for k in (5, 23):
            r[f"ksg_k{k}"] = float(nonparam_mi_cc(X, Z, k=k, base=2.0, alpha="auto"))
            r[f"ksg_k{k}_perm"] = float(nonparam_mi_cc(X, Zp, k=k, base=2.0, alpha="auto"))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    out = a.out or table_path("ksg_vs_gcmi.csv")
    df.to_csv(out, index=False)
    print(df.drop(columns="mouse").mean().round(3).to_string())
    print(f"[OK] {len(df)} sessions -> {out}")


if __name__ == "__main__":
    main()
