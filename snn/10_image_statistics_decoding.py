"""Decodability of six image statistics and of the class from SNN spike counts (needs raw activity).

Six statistics of each CIFAR-10 test image that the network was never trained on and
that are not the class (brightness, contrast, mean R/G/B, high-frequency energy;
snn_common.image_statistics). Per (sublayer, epoch): ridge regression (alpha=100,
standardized inputs) from the channel-pooled spike counts T (the T of H_gauss) to the
six z-scored statistics and to the 10 one-hot class indicators, fitted on a fixed
random half (5 000) of the test images and scored (R^2) on the other half.

Output: <IB_RESULTS>/snn/image_statistics_decoding.csv with r2_aux (mean over the six
statistics), r2_class (mean over the ten indicators), r2_<statistic>, H_gauss, C.
Re-running resumes from the rows already in the file.
"""
import argparse
import csv
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from snn_common import (
    AUX_NAMES,
    EPOCHS,
    LAYERS,
    cifar_test_images,
    gaussian_entropy,
    image_statistics,
    load_activity,
    load_labels,
    require_activity,
)

from ib.paths import results_dir

SPLIT_SEED = 0
RIDGE_ALPHA = 100.0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", nargs="+", default=LAYERS)
    p.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--out", default="image_statistics_decoding.csv")
    args = p.parse_args()
    require_activity()
    out = results_dir("snn") / args.out

    aux = image_statistics(cifar_test_images())
    labels = load_labels("test")
    y_all = np.column_stack([aux, np.eye(10)[labels]])
    idx = np.random.default_rng(SPLIT_SEED).permutation(len(labels))
    tr, te = idx[:5000], idx[5000:]

    rows, done = [], set()
    if out.exists():
        with open(out, newline="") as fh:
            for r0 in csv.DictReader(fh):
                rows.append({k: (v if k == "layer" else float(v)) for k, v in r0.items()})
                done.add((r0["layer"], int(float(r0["epoch"]))))
        print(f"resume: {len(done)} cells done", flush=True)
    for lay in args.layers:
        for ep in args.epochs:
            if (lay, ep) in done:
                continue
            t0 = time.time()
            t = load_activity(ep, lay)
            if t is None:
                print("missing", lay, ep, flush=True)
                continue
            t = t.astype(np.float64)
            sc = StandardScaler().fit(t[tr])
            pred = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(t[tr]), y_all[tr]).predict(sc.transform(t[te]))
            r2 = 1 - np.sum((y_all[te] - pred) ** 2, axis=0) / np.sum(
                (y_all[te] - y_all[te].mean(0)) ** 2, axis=0)
            r = {"layer": lay, "epoch": ep, "C": t.shape[1], "H_gauss": float(gaussian_entropy(t)),
                 "r2_aux": float(np.mean(r2[:6])), "r2_class": float(np.mean(r2[6:])),
                 **{f"r2_{n}": float(r2[i]) for i, n in enumerate(AUX_NAMES)}}
            rows.append(r)
            print(f"{lay} e={ep} H={r['H_gauss']:.0f} aux={r['r2_aux']:.3f} "
                  f"class={r['r2_class']:.3f} [{time.time() - t0:.0f}s]", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"[OK] {len(rows)} rows in {out}")


if __name__ == "__main__":
    main()
