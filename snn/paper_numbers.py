"""SNN numbers quoted in the paper that no other script prints (from cached results).

Each line shows the value printed in the paper next to the value recomputed here:
test accuracy, the across-cell and per-layer MEI correlations, the epoch-700 tables of
Supplementary "SNN per-neuron tuning diagnostics", the lambda = 0 agreement, and the
checks of the dequantization paragraphs.

Inputs (all in <IB_RESULTS>/snn): test_accuracy.npz, neuron_mi_vs_mei.csv,
latent_mi.npz, latent_mi_lambda0.npz, dequant_check_per_cell.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from snn_common import PANEL_LAYERS, short_name

from ib.paths import results_dir

SN1_DEPTH = ["sn1", "layer1.0.sn1", "layer1.1.sn1", "layer2.0.sn1", "layer2.1.sn1",
             "layer3.0.sn1", "layer3.1.sn1", "layer4.0.sn1", "layer4.1.sn1"]
BROAD = 0.75   # max-activation split between broadly tuned and sharp-tuned neurons
SHARP = 0.85   # threshold of the "most sharply tuned" subset
EPOCH = 700


def claim(label, paper, computed):
    print(f"  {label:58s} paper {paper:>14s} | computed {computed}")


def pear(df):
    if len(df) < 3 or df.mi_neuron.std() == 0 or df.rel_perpl.std() == 0:
        return np.nan
    return pearsonr(df.mi_neuron, df.rel_perpl)[0]


def accuracy(res):
    acc = np.load(res / "test_accuracy.npz")["arr_0"]
    print("Methods, test accuracy of the trained SNN (accuracy curve, % per saved checkpoint):")
    claim("final test accuracy", "83%", f"{acc[-1]:.2f}% (last of {len(acc)} entries; max {acc.max():.2f}%)")


def mei_numbers(res):
    d = pd.read_csv(res / "neuron_mi_vs_mei.csv")
    z = np.load(res / "latent_mi.npz")
    layers, epochs = list(z["layers"]), list(z["epochs"])
    cells = d.groupby(["layer", "epoch"]).agg(rho=("rel_perpl", "mean")).reset_index()
    cells["I"] = [z["I_latent"][layers.index(lay), epochs.index(e)]
                  for lay, e in zip(cells.layer, cells.epoch)]
    valid = cells.dropna()
    r, p = pearsonr(valid.rho, valid.I)
    dropped = cells[cells.I.isna()][["layer", "epoch"]].values.tolist()
    print("\nResults, MEI sharpness vs population I(Z;Y) across (sublayer, epoch) cells:")
    claim("MEI cells (sublayers x epochs, 64 neurons each)", "9 x 20 = 180",
          f"{d.layer.nunique()} x {d.epoch.nunique()} = {len(cells)}; "
          f"neurons per cell {sorted(int(x) for x in d.groupby(['layer', 'epoch']).size().unique())}")
    claim("cells in the correlation", "n = 179", f"n = {len(valid)} (dropped: {dropped}, "
          "I(Z;Y) is NaN there)")
    claim("r(mean rho_n, I(Z;Y))", "-0.61, p<<1e-3", f"{r:+.3f}, p={p:.1e}")

    e7 = d[d.epoch == EPOCH]
    print(f"\nResults / Supplementary table, epoch {EPOCH} (64 neurons per layer):")
    paper_r = {"sn1": "-0.15", "layer1.0.sn1": "+0.02", "layer1.1.sn1": "-0.19",
               "layer2.0.sn1": "-0.25", "layer2.1.sn1": "-0.34", "layer3.0.sn1": "-0.42",
               "layer3.1.sn1": "+0.01 (~0)", "layer4.0.sn1": "-0.67", "layer4.1.sn1": "-0.75"}
    paper_mi = {"sn1": "0.12", "layer1.0.sn1": "0.16", "layer1.1.sn1": "0.16",
                "layer2.0.sn1": "0.15", "layer2.1.sn1": "0.23", "layer3.0.sn1": "0.23",
                "layer3.1.sn1": "0.58", "layer4.0.sn1": "0.53", "layer4.1.sn1": "0.52"}
    paper_broad = {"sn1": "70.3%", "layer1.0.sn1": "32.8%", "layer1.1.sn1": "93.8%",
                   "layer2.0.sn1": "78.1%", "layer2.1.sn1": "60.9%", "layer3.0.sn1": "18.8%",
                   "layer3.1.sn1": "62.5%", "layer4.0.sn1": "12.5%", "layer4.1.sn1": "29.7%"}
    paper_sharp = {"layer3.0.sn1": "81%", "layer3.1.sn1": "38%", "layer4.0.sn1": "88%",
                   "layer4.1.sn1": "70%"}
    for lay in SN1_DEPTH:
        s = e7[e7.layer == lay]
        nm = short_name(lay).replace(".sn1", "") if lay != "sn1" else "sn1"
        claim(f"{nm}: r(I_n, rho_n)", paper_r[lay], f"{pear(s):+.3f} (n={len(s)})")
        claim(f"{nm}: mean I(t_n;Y), bits", paper_mi[lay], f"{s.mi_neuron.mean():.3f}")
        claim(f"{nm}: broadly tuned (max-activation < {BROAD})", paper_broad[lay],
              f"{100 * (s.max_activation < BROAD).mean():.1f}%")
        if lay in paper_sharp:
            claim(f"{nm}: above {BROAD}", paper_sharp[lay],
                  f"{100 * (s.max_activation > BROAD).mean():.1f}%")
    m30 = e7[e7.layer == "layer3.0.sn1"].mi_neuron.mean()
    m31 = e7[e7.layer == "layer3.1.sn1"].mi_neuron.mean()
    claim("mean I(t_n;Y) jump L3.0 -> L3.1", "2.5x", f"{m31 / m30:.2f}x")

    print(f"\nSupplementary threshold-filtered correlation (max-activation >= {SHARP}):")
    paper_thr = {"layer2.1.sn1": ("-0.30", 8), "layer3.0.sn1": ("-0.55", 33),
                 "layer3.1.sn1": ("-0.21", 14), "layer4.0.sn1": ("-0.25", 43),
                 "layer4.1.sn1": ("-0.52", 32)}
    for lay, (pr, pn) in paper_thr.items():
        s = e7[(e7.layer == lay) & (e7.max_activation >= SHARP)]
        claim(f"{short_name(lay)}: r and n at threshold", f"{pr}, n={pn}",
              f"{pear(s):+.3f}, n={len(s)}")
    print("\nSupplementary scatter figure panels (same correlation, epoch 700):")
    for lay, pr in (("layer3.0.sn1", "-0.42"), ("layer3.1.sn1", "+0.01"), ("layer4.0.sn1", "-0.66")):
        claim(f"{short_name(lay)}: r", pr, f"{pear(e7[e7.layer == lay]):+.3f}")


def lambda0_numbers(res):
    z = np.load(res / "latent_mi.npz")
    a = np.load(res / "latent_mi_lambda0.npz")
    layers = list(z["layers"])
    assert list(a["epochs"]) == list(z["epochs"])
    print("\nSupplementary lambda = 0, panel sublayers over the 26 checkpoints:")
    paper_r = {"layer2.1.sn1": "0.998", "layer3.0.sn1": "0.999", "layer3.1.sn1": "1.000",
               "layer4.0.sn1": "0.999", "layer4.1.sn1": "0.999"}
    paper_e900 = {"layer2.1.sn1": "1.77 vs 1.93", "layer3.0.sn1": "2.12 vs 2.18",
                  "layer3.1.sn1": "2.30 vs 2.27", "layer4.0.sn1": "2.01 vs 2.02",
                  "layer4.1.sn1": "1.72 vs 1.78"}
    rel = []
    for i, lay in enumerate(a["layers"]):
        x, y = a["I_latent"][i], z["I_latent"][layers.index(lay)]
        m = np.isfinite(x) & np.isfinite(y)
        rel.append(abs(x[-1] - y[-1]) / y[-1])
        claim(f"{short_name(lay)}: r(lambda=0, lambda=1)", paper_r[lay],
              f"{pearsonr(x[m], y[m])[0]:.4f} (n={m.sum()})")
        claim(f"{short_name(lay)}: epoch 900, lambda=0 vs lambda=1", paper_e900[lay],
              f"{x[-1]:.2f} vs {y[-1]:.2f} ({100 * rel[-1]:.1f}%)")
    claim("largest relative difference at epoch 900", "within 8%", f"{100 * max(rel):.1f}%")
    e900 = a["I_latent"][:, -1]
    claim("population-information peak at epoch 900, lambda=0", "L3.1",
          short_name(str(a["layers"][int(np.argmax(e900))])))
    i_all = z["I_latent"]
    for e in (700, 800, 900):
        j = list(z["epochs"]).index(e)
        claim(f"argmax of I(Z;Y) over the 17 sublayers, epoch {e}", "L3.1 (700-900)",
              short_name(layers[int(np.nanargmax(i_all[:, j]))]))
    assert set(a["layers"]) == set(PANEL_LAYERS)


def dequant_numbers(res):
    d = pd.read_csv(res / "dequant_check_per_cell.csv").dropna(subset=["H_raw"])
    print("\nSupplementary dequantization paragraphs (ten sublayers x 26 epochs):")
    claim("cells with H_gauss > 0", "260 of 260", f"{(d.H_raw > 0).sum()} of {len(d)}")
    claim("range of H_gauss, bits", "104 to 1634", f"{d.H_raw.min():.1f} to {d.H_raw.max():.1f}")
    e9 = d[d.epoch == 900]
    claim("epoch 900: eigenvalues kept = N in every sublayer", "yes",
          str(bool((e9.n_eigs_kept_raw == e9.N_neurons).all())))
    claim("all cells: eigenvalues kept = N", "(not claimed)",
          str(bool((d.n_eigs_kept_raw == d.N_neurons).all())))
    claim("largest feature-to-sample ratio N/n", "at most 0.05", f"{(d.N_neurons / d.n_samples).max():.4f}")
    rel_cell = 100 * d.delta / d.H_raw
    g = d.groupby("sublayer")
    per_layer = 100 * g.delta.mean() / g.H_raw.mean()
    claim("relative shift, per-sublayer means", "+0.6% to +6.8%",
          f"{per_layer.min():+.1f}% to {per_layer.max():+.1f}%")
    worst = d.loc[rel_cell.idxmax()]
    claim("relative shift, largest single cell", "under 7% in every cell",
          f"{rel_cell.max():.1f}% ({worst.sublayer}, epoch {int(worst.epoch)})")
    claim("cells with relative shift >= 7%", "0", f"{int((rel_cell >= 7).sum())} of {len(d)}")

    def drops(col):
        out = {}
        for lay, s in d.sort_values("epoch").groupby("sublayer"):
            v = s[col].values
            out[lay] = v.max() - v[-1]
        return pd.Series(out)

    raw, dq = drops("H_raw"), drops("H_dequant")
    deep = [x for x in raw.index if x.startswith("layer4")]
    shrink = 100 * (raw[deep] - dq[deep]) / raw[deep]
    claim("L4 drop shrinks under dequantization by", "4-19%",
          f"{shrink.min():.1f}-{shrink.max():.1f}%")
    claim("L4.0.sn2 raw drop (two SI tables)", "159 and 158", f"{raw['layer4.0.sn2']:.2f}")
    shallow = [x for x in raw.index if not x.startswith("layer4")]
    claim("L2.1-L3.1 raw drops, bits", "0-5", f"{raw[shallow].min():.2f}-{raw[shallow].max():.2f}")
    claim("L2.1-L3.1 |raw - dequant drop|, bits", "within 0.1",
          f"max {(raw[shallow] - dq[shallow]).abs().max():.2f} "
          f"({(raw[shallow] - dq[shallow]).abs().idxmax()})")


def main():
    res = results_dir("snn")
    accuracy(res)
    mei_numbers(res)
    lambda0_numbers(res)
    dequant_numbers(res)


if __name__ == "__main__":
    main()
