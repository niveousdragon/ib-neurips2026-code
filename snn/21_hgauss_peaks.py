"""Peak, peak epoch, end value and drop of H_gauss(T) for all 17 sublayers (from cached results).

Gives the deep-layer table of Supplementary "Per-sublayer peak/end values", the
143-321-bit drops of the main text, and the at-most-22-bit drop of the 13 sublayers
outside stage 4 (Supplementary "Auxiliary image statistics").

Input:  <IB_RESULTS>/snn/latent_mi.npz
Output: <IB_RESULTS>/snn/hgauss_peaks.csv
"""
import numpy as np
import pandas as pd

from ib.paths import results_dir


def main():
    z = np.load(results_dir("snn") / "latent_mi.npz")
    ep, h, i_lat = z["epochs"], z["H_gauss"], z["I_latent"]
    rows = []
    for i, layer in enumerate(z["layers"]):
        k = int(np.nanargmax(h[i]))
        ki = int(np.nanargmax(i_lat[i]))
        rows.append(dict(layer=str(layer), H0=h[i, 0], H_peak=h[i, k], peak_epoch=int(ep[k]),
                         H_end=h[i, -1], drop_bits=h[i, k] - h[i, -1],
                         drop_pct=100 * (h[i, k] - h[i, -1]) / h[i, k], rise_bits=h[i, k] - h[i, 0],
                         I_peak_epoch=int(ep[ki]), I_end=i_lat[i, -1], I_max=i_lat[i, ki]))
    df = pd.DataFrame(rows)
    out = results_dir("snn") / "hgauss_peaks.csv"
    df.to_csv(out, index=False, float_format="%.2f")
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))

    deep = df[df.layer.str.startswith("layer4")]
    rest = df[~df.layer.str.startswith("layer4")]
    print(f"\nstage-4 sublayers: drop {deep.drop_bits.min():.0f}-{deep.drop_bits.max():.0f} bits, "
          f"peak epochs {sorted(deep.peak_epoch.tolist())}")
    print(f"other {len(rest)} sublayers: largest drop {rest.drop_bits.max():.2f} bits "
          f"({rest.loc[rest.drop_bits.idxmax(), 'layer']})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
