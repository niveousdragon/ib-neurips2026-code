"""Drift of the across-depth peak of ID and of I(Z;Y) over training (from cached results).

Per epoch, a parabola is fitted to the 17-sublayer profile around its argmax and its
vertex is the peak position (snn_common.fit_vertex). The drift is the slope of the
vertex against log10 epoch, in sublayers per decade. The paper quotes two
regressions of the same vertices:
  Supplementary "Hunchback parabolic-fit method", I(Z;Y): np.polyfit on log10(epoch),
      epochs >= 5 with a vertex                                   (+2.8, r = +0.98)
  main text and the same Supplementary section: scipy linregress on log10(epoch + 1),
      I(Z;Y) at epochs >= 5 (+2.9, r = +0.98), ID at epochs >= 30 (+1.8, r = +0.97)

Inputs:  <IB_RESULTS>/snn/dimensionality.npz, latent_mi.npz
Output:  <IB_RESULTS>/snn/peak_drift.csv (vertex and argmax per epoch)
"""
import numpy as np
import pandas as pd
from scipy import stats
from snn_common import fit_vertex

from ib.paths import results_dir


def drift_linregress(ep, v, label, emin):
    m = np.isfinite(v) & (ep >= emin)
    r = stats.linregress(np.log10(ep[m] + 1), v[m])
    print(f"  {label}: {r.slope:+.2f} sublayers/decade, r={r.rvalue:+.2f}, n={m.sum()}")
    return r.slope, r.rvalue


def main():
    res = results_dir("snn")
    d = np.load(res / "dimensionality.npz")
    mi = np.load(res / "latent_mi.npz")
    id_nn, i_lat = d["id_nn"], mi["I_latent"]
    epochs = d["epochs"]
    assert np.array_equal(epochs, mi["epochs"])

    rows = []
    for ie, e in enumerate(epochs):
        x_id, a_id = fit_vertex(id_nn[:, ie])
        x_i, a_i = fit_vertex(i_lat[:, ie])
        rows.append({"epoch": int(e),
                     "argmax_id": int(np.argmax(id_nn[:, ie])),
                     "argmax_I": int(np.argmax(i_lat[:, ie])),
                     "vertex_id": x_id, "vertex_I": x_i,
                     "concavity_id": a_id, "concavity_I": a_i})
    df = pd.DataFrame(rows)
    out = res / "peak_drift.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")
    print(df.to_string(index=False))

    print("\nSupplementary regression (polyfit on log10(epoch), epochs >= 5):")
    for col, name in (("vertex_id", "ID"), ("vertex_I", "I(Z;Y)")):
        v = df.dropna(subset=[col])
        v = v[v.epoch >= 5]
        le = np.log10(v.epoch.values)
        slope = np.polyfit(le, v[col].values, 1)[0]
        r = stats.pearsonr(le, v[col].values)[0]
        print(f"  {name} vertex drift: {slope:+.3f} sublayers/decade, Pearson r={r:+.3f}, n={len(v)}")

    print("\nMain-text regression (linregress on log10(epoch + 1)):")
    ep = df.epoch.values
    drift_linregress(ep, df.vertex_I.values, "I(Z;Y) vertex, epochs >= 5", 5)
    drift_linregress(ep, df.vertex_id.values, "ID vertex, epochs >= 30", 30)
    drift_linregress(ep, df.vertex_id.values, "ID vertex, epochs >= 5 (not quoted)", 5)
    first_i = ep[np.isfinite(df.vertex_I.values)].min()
    print(f"  first epoch with an I(Z;Y) vertex: {first_i}")
    print("  ID vertex by epoch: " + ", ".join(
        f"{e}:{v:.1f}" for e, v in zip(ep, df.vertex_id) if np.isfinite(v)))
    print("  I(Z;Y) vertex by epoch: " + ", ".join(
        f"{e}:{v:.1f}" for e, v in zip(ep, df.vertex_I) if np.isfinite(v)))


if __name__ == "__main__":
    main()
