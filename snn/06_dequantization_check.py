"""Dequantization check of H_gauss(T) on the SNN (needs raw activity).

Spike counts are integers, so a Gaussian differential entropy is not a strict upper
bound on their discrete entropy. The standard fix adds U[0,1] noise, i.e. shifts the
covariance by I/12 (the variance of that noise) before the eigen-decomposition:

  H_raw     = 0.5 * sum log2(2 pi e lambda_k),             lambda_k of cov(T)
  H_dequant = 0.5 * sum log2(2 pi e lambda_k),             lambda_k of cov(T) + I/12

for both spike layers (sn1, sn2) of the five main-figure blocks at all 26 epochs.

Outputs: <IB_RESULTS>/snn/dequant_check_per_cell.csv, dequant_check_summary.txt
(tables of Supplementary "Validity of the Gaussian-form estimator").
"""
import argparse
import time

import numpy as np
import pandas as pd
from snn_common import EPOCHS, load_activity, require_activity

from ib.paths import results_dir

SUBLAYERS = [
    "layer2.1.sn1", "layer2.1.sn2",
    "layer3.0.sn1", "layer3.0.sn2",
    "layer3.1.sn1", "layer3.1.sn2",
    "layer4.0.sn1", "layer4.0.sn2",
    "layer4.1.sn1", "layer4.1.sn2",
]


def h_gauss(cov):
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]
    return 0.5 * np.sum(np.log2(2 * np.pi * np.e * eigs)), len(eigs)


def reversal(df, sublayer):
    """Peak, peak epoch, end-of-training value and drop of both estimators."""
    sub = df[df.sublayer == sublayer].sort_values("epoch")
    out = {}
    for col in ("H_raw", "H_dequant"):
        vals = sub[col].values
        if np.isnan(vals).all():
            out[col] = None
            continue
        k = int(np.nanargmax(vals))
        out[col] = dict(peak_epoch=int(sub.iloc[k].epoch), peak=float(vals[k]),
                        end=float(vals[-1]), drop=float(vals[k] - vals[-1]))
    return out


def summarize(df):
    lines = ["Dequantization correction (cov += I/12) on H_gauss", "",
             "Per-sublayer summary across epochs:", "",
             f"{'sublayer':<16s}{'N':>5s}  {'mean H_raw':>11s}{'mean H_deq':>11s}"
             f"{'mean delta':>11s}  {'min delta':>10s}{'max delta':>10s}"]
    for layer in SUBLAYERS:
        sub = df[df.sublayer == layer].dropna(subset=["H_raw"])
        if len(sub) == 0:
            continue
        lines.append(f"{layer:<16s}{int(sub.N_neurons.iloc[0]):>5d}  "
                     f"{sub.H_raw.mean():>11.2f}{sub.H_dequant.mean():>11.2f}"
                     f"{sub.delta.mean():>11.4f}  {sub.delta.min():>10.4f}{sub.delta.max():>10.4f}")
    lines += ["", "Two-phase reversal check (peak vs end of training):", ""]
    for layer in [s for s in SUBLAYERS if s in set(df.sublayer)]:
        chk = reversal(df, layer)
        lines.append(f"  {layer}:")
        for col, label in (("H_raw", "raw    "), ("H_dequant", "dequant")):
            c = chk[col]
            lines.append(f"    {label}: no data" if c is None else
                         f"    {label}: peak={c['peak']:.2f} at e={c['peak_epoch']:>3d}, "
                         f"end={c['end']:.2f}, drop={c['drop']:.2f} bits")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", nargs="+", default=SUBLAYERS)
    p.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--out", default="dequant_check_per_cell.csv")
    args = p.parse_args()

    out = results_dir("snn") / args.out
    if out.exists() and args.layers == SUBLAYERS and args.epochs == EPOCHS:
        print(f"Loaded {out} (delete it to recompute)")
        df = pd.read_csv(out)
    else:
        require_activity()
        rows, t0 = [], time.time()
        for li, layer in enumerate(args.layers):
            for epoch in args.epochs:
                act = load_activity(epoch, layer)
                if act is None:
                    rows.append(dict(sublayer=layer, epoch=epoch, note="missing"))
                    continue
                cov = np.cov(act, rowvar=False)
                hr, nr = h_gauss(cov)
                hd, nd = h_gauss(cov + np.eye(cov.shape[0]) / 12.0)
                rows.append(dict(sublayer=layer, epoch=epoch, N_neurons=act.shape[1],
                                 n_samples=act.shape[0], H_raw=hr, H_dequant=hd, delta=hd - hr,
                                 n_eigs_kept_raw=nr, n_eigs_kept_dequant=nd, note=""))
            print(f"  [{li + 1}/{len(args.layers)}] {layer} ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        print(f"Saved {out}")
    summary = summarize(df)
    print(summary)
    summary_path = out.with_name(out.stem.removesuffix("_per_cell") + "_summary.txt")
    summary_path.write_text(summary + "\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
