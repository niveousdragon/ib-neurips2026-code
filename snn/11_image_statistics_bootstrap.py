"""Split-resampling intervals for the image-statistics decoding (needs raw activity).

Repeats the ridge decoding of the six image statistics of 10_image_statistics_decoding.py
for 30 random 5 000 / 5 000 splits of the CIFAR-10 test images (seeds 1000..1029),
at the epochs >= 30 used for the slopes. 24_image_statistics_slopes.py refits the
slope per split and takes the 2.5-97.5 percentiles as the interval.

Output: <IB_RESULTS>/snn/image_statistics_bootstrap.csv with columns layer, epoch,
b0..b29 (held-out R^2 averaged over the six statistics). Resumes from existing rows.
"""
import argparse
import csv
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from snn_common import (
    EPOCHS,
    LAYERS,
    cifar_test_images,
    image_statistics,
    load_activity,
    require_activity,
)

from ib.paths import results_dir

N_SPLITS = 30
MIN_EPOCH = 30
RIDGE_ALPHA = 100.0


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    require_activity()
    out = results_dir("snn") / "image_statistics_bootstrap.csv"
    aux = image_statistics(cifar_test_images())
    n = len(aux)
    epochs = [e for e in EPOCHS if e >= MIN_EPOCH]
    rows, done = [], set()
    if out.exists():
        with open(out) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add((r["layer"], int(float(r["epoch"]))))
        print(f"resume: {len(done)} cells done", flush=True)
    for lay in LAYERS:
        for ep in epochs:
            if (lay, ep) in done:
                continue
            t0 = time.time()
            t = load_activity(ep, lay)
            if t is None:
                continue
            t = t.astype(np.float64)
            r2s = []
            for b in range(N_SPLITS):
                idx = np.random.default_rng(1000 + b).permutation(n)
                tr, te = idx[:5000], idx[5000:]
                sc = StandardScaler().fit(t[tr])
                pred = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(t[tr]), aux[tr]).predict(sc.transform(t[te]))
                r2s.append(np.mean(1 - np.sum((aux[te] - pred) ** 2, axis=0)
                                   / np.sum((aux[te] - aux[te].mean(0)) ** 2, axis=0)))
            rows.append({"layer": lay, "epoch": ep, **{f"b{i}": f"{v:.6f}" for i, v in enumerate(r2s)}})
            print(f"{lay} e={ep} mean={np.mean(r2s):.4f} sd={np.std(r2s):.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"[OK] {len(rows)} rows in {out}")


if __name__ == "__main__":
    main()
