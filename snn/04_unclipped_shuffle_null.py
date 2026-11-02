"""Unclipped KSG I(Z;Y) and its label-permutation null from cached SNN latents.

Uses the same estimator as driada's nonparam_mi_cd, I = H_KL(Z) - sum_y p(y) H_KL(Z|y)
(k=5, bits, tiny uniform noise to break ties), but without its final max(0, .)
clip, so the null keeps its natural, negative values. 7 label permutations per cell.

Input:  <IB_RESULTS>/snn/latent_zs.npz (written by 03_shuffle_null.py; not shipped,
        about 280 MB) and the CIFAR-10 test labels under IB_DATA_SNN.
Output: <IB_RESULTS>/snn/latent_mi_unclipped_null.npz
        (I_raw_unclipped, H_Z, I_null[layer, epoch, perm]).
Each cell is seeded on its own (1000 * layer + epoch index), so the result does not
depend on the number of worker processes.
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
from snn_common import load_labels

from ib.paths import results_dir

N_PERMS = 7
K = 5


def cond_entropy(z, y):
    from driada.information.ksg import nonparam_entropy_c

    h = 0.0
    n = len(y)
    for c in np.unique(y):
        m = y == c
        if m.sum() > K:
            h += m.sum() / n * nonparam_entropy_c(z[m], k=K, base=2.0)
    return h


def work(job):
    from driada.information.ksg import add_noise, nonparam_entropy_c

    zs_path, key, i, j = job
    rng = np.random.default_rng(1000 * i + j)
    np.random.seed(1000 * i + j)  # add_noise draws from numpy's global generator
    z = np.load(zs_path)[key].astype(np.float64)
    y = load_labels("test")
    z = add_noise(z)  # as inside nonparam_mi_cd
    hx = nonparam_entropy_c(z, k=K, base=2.0)
    raw = hx - cond_entropy(z, y)
    null = np.array([hx - cond_entropy(z, rng.permutation(y)) for _ in range(N_PERMS)])
    return i, j, hx, raw, null


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--keys", nargs="*", help="only these cells, e.g. L11_e23 (quick tests)")
    p.add_argument("--out", default="latent_mi_unclipped_null.npz")
    args = p.parse_args()

    zs_path = results_dir("snn") / "latent_zs.npz"
    if not zs_path.exists():
        raise SystemExit(f"{zs_path} not found; run 03_shuffle_null.py first.")
    d = np.load(zs_path)
    layers, epochs = d["layers"], d["epochs"]
    keys = args.keys or [k for k in d.files if k not in ("layers", "epochs")]
    jobs = []
    for k in keys:
        a, b = k.split("_")
        jobs.append((str(zs_path), k, int(a[1:]), int(b[1:])))
    n_l, n_e = len(layers), len(epochs)
    raw = np.full((n_l, n_e), np.nan)
    hx = np.full((n_l, n_e), np.nan)
    null = np.full((n_l, n_e, N_PERMS), np.nan)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for n, (i, j, h, r, nu) in enumerate(pool.imap_unordered(work, jobs)):
            raw[i, j], hx[i, j], null[i, j] = r, h, nu
            if n % 20 == 0:
                print(f"[{n + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    out = results_dir("snn") / args.out
    np.savez(out, I_raw_unclipped=raw, H_Z=hx, I_null=null,
             layers=layers, epochs=epochs, n_perms=N_PERMS, k=K)
    print(f"[OK] saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
