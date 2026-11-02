"""Per-neuron mutual information I(t_n; Y) joined with per-neuron MEI sharpness (needs raw activity).

For each (epoch, layer, neuron) covered by the MEI pipeline, I(t_n; Y) is the plug-in
mutual information (bits) between the neuron's spike count and the class label on the
10 000 CIFAR-10 test images. The stored counts are spatial averages and can be
fractional; they are truncated to integers before the plug-in estimate. The MEI
pipeline ran three optimisation runs per neuron; rel_perpl, max_activation and
max_prob are averaged over them. Both rules were checked to reproduce the shipped
neuron_mi_vs_mei.csv exactly (see the release notes).

Input:  <IB_RESULTS>/snn/mei_summary.csv (07_mei_convert.py) and the test activity.
Output: <IB_RESULTS>/snn/neuron_mi_vs_mei.csv with columns
        layer, epoch, neuron, mi_neuron, rel_perpl, max_activation, max_prob
(input of Figure 2, Supplementary Figure "Per-layer (rho_n, I_n) scatter" and
paper_numbers.py). --layer/--epoch restrict the run to some cells (quick checks).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from snn_common import LN2, load_activity, load_labels, require_activity

from ib.paths import results_dir


def per_neuron_mi(act, labels):
    """I(t_n; Y) of every channel on integer-truncated counts, bits."""
    counts = act.astype(np.int64)
    return np.array([mutual_info_score(labels, counts[:, i]) / LN2
                     for i in range(counts.shape[1])], dtype=np.float64)


def mean_over_runs(mei):
    """One row per (epoch, layer, neuron): MEI scores averaged over the optimisation runs."""
    return (mei.groupby(["epoch", "layer", "neuron"], as_index=False)
               [["max_activation", "rel_perpl", "max_prob"]].mean())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layer", nargs="*", help="only these sublayers")
    p.add_argument("--epoch", nargs="*", type=int, help="only these epochs")
    p.add_argument("--out", default="neuron_mi_vs_mei.csv")
    args = p.parse_args()
    require_activity()

    mei_path = results_dir("snn") / "mei_summary.csv"
    if not mei_path.exists():
        raise SystemExit(f"{mei_path} not found; run 07_mei_convert.py first.")
    mei_agg = mean_over_runs(pd.read_csv(mei_path))
    labels = load_labels("test")

    pairs = mei_agg[["epoch", "layer"]].drop_duplicates()
    if args.layer:
        pairs = pairs[pairs.layer.isin(args.layer)]
    if args.epoch:
        pairs = pairs[pairs.epoch.isin(args.epoch)]
    print(f"(epoch, layer) cells with MEI coverage: {len(pairs)}")
    rows = []
    for _, row in pairs.iterrows():
        epoch, layer = int(row["epoch"]), str(row["layer"])
        act = load_activity(epoch, layer)
        if act is None:
            print(f"  missing: e={epoch} {layer}")
            continue
        mi = per_neuron_mi(act, labels)
        sub = mei_agg[(mei_agg.epoch == epoch) & (mei_agg.layer == layer)]
        for _, m in sub.iterrows():
            n_idx = int(m["neuron"])
            if n_idx >= mi.size:
                continue
            rows.append({"layer": layer, "epoch": epoch, "neuron": n_idx,
                         "mi_neuron": float(mi[n_idx]), "rel_perpl": float(m["rel_perpl"]),
                         "max_activation": float(m["max_activation"]),
                         "max_prob": float(m["max_prob"])})
        print(f"  ok: e={epoch} {layer} (N={act.shape[1]}, covered={len(sub)})")
    out = results_dir("snn") / args.out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
