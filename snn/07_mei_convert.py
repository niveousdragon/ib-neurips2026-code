"""Flatten the MEI results of the external MEI pipeline (YAML) into a table (needs raw data).

The maximally exciting inputs (MEIs) and their sharpness scores come from the MEI
pipeline of Pospelov et al. (cited in the paper), run separately; this script only
converts its YAML output:
  - <IB_RESULTS>/snn/mei_summary.csv: one row per (epoch, layer, neuron, method) with
    max_activation, complexity, rel_perpl (relative perplexity, rho_n in the paper),
    argmax_class, class_name, max_prob;
  - <IB_RESULTS>/snn/mei_probs.npz: the 10-class probability vectors, rows in the
    order of mei_summary.csv (not used by the released scripts; not shipped).

Input: $IB_DATA_SNN/SJ-SNN-T50/SJ-SNN evolution data full 3.yaml (about 770 MB),
or --yaml <path>.
"""
import argparse
import csv

import numpy as np
import yaml
from snn_common import MEI_YAML

from ib.paths import data_dir, results_dir

LABEL_NAMES = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
               5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
FIELDS = ["epoch", "layer", "neuron", "method", "max_activation", "complexity",
          "rel_perpl", "argmax_class", "class_name", "max_prob"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--yaml", default=None, help=f"default: $IB_DATA_SNN/{MEI_YAML}")
    args = p.parse_args()
    yaml_path = args.yaml or data_dir("snn") / MEI_YAML
    out_csv = results_dir("snn") / "mei_summary.csv"
    out_npz = results_dir("snn") / "mei_probs.npz"

    print(f"Loading {yaml_path}...")
    with open(yaml_path) as f:
        # The file stores numpy scalars as Python objects, which safe_load rejects.
        data = yaml.unsafe_load(f)
    epochs = sorted(data.keys())
    print(f"Epochs: {epochs}")

    rows, all_probs = [], []
    for epoch in epochs:
        layers = data[epoch]
        for layer in sorted(layers.keys()):
            for neuron_id, methods in layers[layer].items():
                for method_key, info in methods.items():
                    probs = info.get("probs", None)
                    if probs is not None:
                        probs = np.array(probs, dtype=np.float32)
                        argmax_class, max_prob = int(np.argmax(probs)), float(np.max(probs))
                    else:
                        probs = np.full(10, np.nan, dtype=np.float32)
                        argmax_class, max_prob = -1, np.nan
                    all_probs.append(probs)
                    rows.append({
                        "epoch": epoch, "layer": layer, "neuron": int(np.array(neuron_id)),
                        "method": method_key,
                        "max_activation": float(np.array(info.get("max_a", np.nan))),
                        "complexity": info.get("complexity", None),
                        "rel_perpl": float(info.get("rel_perpl", np.nan)),
                        "argmax_class": argmax_class,
                        "class_name": LABEL_NAMES.get(argmax_class, ""),
                        "max_prob": max_prob,
                    })
    print(f"Total rows: {len(rows)}")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")
    probs_array = np.stack(all_probs)
    np.savez_compressed(out_npz, probs=probs_array, epochs=np.array(epochs),
                        rows_info="Same order as mei_summary.csv")
    print(f"Saved {out_npz}, shape={probs_array.shape}")


if __name__ == "__main__":
    main()
