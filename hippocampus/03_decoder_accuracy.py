"""Per-session raw decoder accuracy on the 10 x 10 position bin (Supplementary
"Why raw decoder accuracy gives the wrong sign").

Accuracy of the relevance-axis logistic decoder and of the MLP decoder on the
saved latents (test fold, bins seen in training), and the majority-class chance
level, joined with the information-plane table of the same regime.

Output: results/hippocampus/decoder_accuracy_<regime>.csv
Usage:  python hippocampus/03_decoder_accuracy.py --regime lambda1
"""
import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
from hippo_common import (
    REGIMES,
    _decodable,
    add_session_args,
    bin_2d,
    list_probe_files,
    npz_day,
    parse_sessions,
    probes_dir,
    table_path,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def accuracy(clf, Z_tr, b_tr, Z_te, b_te):
    keep = _decodable(b_tr, b_te)
    if keep is None:
        return float("nan")
    sc = StandardScaler().fit(Z_tr)
    try:
        clf.fit(sc.transform(Z_tr), b_tr)
    except Exception:
        return float("nan")
    return float((clf.predict(sc.transform(Z_te[keep])) == b_te[keep]).mean())


def chance_accuracy(b_tr, b_te):
    """Accuracy of always predicting the most frequent test bin among those seen in training."""
    seen = set(int(c) for c in np.unique(b_tr))
    counts = {k: v for k, v in Counter(b_te.tolist()).items() if k in seen}
    n = sum(counts.values())
    if not counts or n < 20:
        return float("nan")
    return max(counts.values()) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--ib-plane", default=None,
                    help="information-plane table to join (default: ib_plane_<regime>.csv)")
    add_session_args(ap)
    a = ap.parse_args()

    rows = []
    t0 = time.time()
    for f in list_probe_files(probes_dir(a.regime, a.probes_dir), parse_sessions(a.sessions)):
        d = np.load(f)
        Z_tr = d["Z_tr"].astype(np.float32)
        Z_te = d["Z_te"].astype(np.float32)
        b_tr = bin_2d(d["Y_tr"].astype(np.float32))
        b_te = bin_2d(d["Y_te"].astype(np.float32))
        lr = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")
        mlp = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=200,
                            early_stopping=True, n_iter_no_change=10, random_state=42)
        rows.append({"mouse": str(d["mouse"]), "day": npz_day(d),
                     "acc_logreg": accuracy(lr, Z_tr, b_tr, Z_te, b_te),
                     "acc_mlp": accuracy(mlp, Z_tr, b_tr, Z_te, b_te),
                     "acc_chance": chance_accuracy(b_tr, b_te)})
        r = rows[-1]
        print(f"  {r['mouse']} d{r['day']}: logreg {r['acc_logreg']:.3f}  "
              f"mlp {r['acc_mlp']:.3f}  chance {r['acc_chance']:.3f}", flush=True)

    ib = pd.read_csv(a.ib_plane or table_path(f"ib_plane_{a.regime}.csv"))
    df = pd.DataFrame(rows).merge(
        ib[["mouse", "day", "I_xz_corr", "I_zy_bin_corr", "I_xz_real", "I_zy_bin_real"]],
        on=["mouse", "day"], how="inner")
    out = a.out or table_path(f"decoder_accuracy_{a.regime}.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} sessions -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
