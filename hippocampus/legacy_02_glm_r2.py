"""Per-session r^2 of the earlier 20 Hz pipeline's latents with behavioural predictors
(needs raw data and the legacy_01_pilot_20hz.py files, Delta_t = 1 s, real and shuffled).

Squared Pearson correlation of each of the 16 latent dimensions with x, y, d_obj1..4,
speed and omega on the test frames; latent dimensions greedily ordered per predictor,
separately for the real and the shuffled latent (one shuffle per session).

Output: results/hippocampus/glm_r2_legacy_20hz.csv (long: mouse, day, rank, predictor,
        r2_real, r2_shuf); the figure is drawn by fig_si_glm_heatmap.py.
Usage:  python hippocampus/legacy_02_glm_r2.py [--pilot-dir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hippo_common import load_config, session_path, table_path

from ib.paths import results_dir

PREDICTORS = ["x", "y", "d_obj1", "d_obj2", "d_obj3", "d_obj4", "speed", "ang_vel"]


def session_predictors(sess, sr):
    x = sess["x"].astype(np.float64)
    y = sess["y"].astype(np.float64)
    omega = np.gradient(np.unwrap(sess["headdirection"].astype(np.float64))) * sr
    dists = []
    for i in (1, 2, 3, 4):
        key = f"object{i}"
        if key in sess.files and sess[key].astype(bool).sum() > 0:
            zone = sess[key].astype(bool)
            dists.append(np.sqrt((x - x[zone].mean()) ** 2 + (y - y[zone].mean()) ** 2))
        else:
            dists.append(np.zeros_like(x))
    return np.column_stack([x, y, *dists, sess["speed"].astype(np.float64), omega])


def session_r2(Z, X):
    r2 = np.zeros((Z.shape[1], X.shape[1]))
    for k in range(Z.shape[1]):
        if Z[:, k].std() < 1e-9:
            continue
        for j in range(X.shape[1]):
            if X[:, j].std() < 1e-9:
                continue
            r2[k, j] = np.corrcoef(Z[:, k], X[:, j])[0, 1] ** 2
    return r2


def greedy_reorder(r2):
    order, used = [], set()
    for j in range(r2.shape[1]):
        best_k, best_v = -1, -np.inf
        for k in range(r2.shape[0]):
            if k not in used and r2[k, j] > best_v:
                best_k, best_v = k, r2[k, j]
        if best_k >= 0:
            order.append(best_k)
            used.add(best_k)
    rest = [k for k in range(r2.shape[0]) if k not in used]
    order += [k for _, k in sorted(zip([r2[k].sum() for k in rest], rest), reverse=True)]
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-dir", default=None,
                    help="folder of pilot_<mouse>_<day>D_dt1[_shuffled].npz "
                         "(default: results/hippocampus/legacy_20hz)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    sr = float(cfg["sampling_rate_hz"])
    folder = Path(a.pilot_dir) if a.pilot_dir else results_dir("hippocampus", "legacy_20hz")
    rows = []
    for pilot in sorted(p for p in folder.glob("pilot_*_*D_dt1.npz")):
        _, mouse, day_str, _ = pilot.stem.split("_")
        day = int(day_str.rstrip("D"))
        shuf = folder / f"pilot_{mouse}_{day_str}_dt1_shuffled.npz"
        path = session_path(cfg, mouse, day)
        if not (path.exists() and shuf.exists()):
            continue
        saved = np.load(pilot)
        X_te = session_predictors(np.load(path), sr)[saved["valid_idx"].astype(int)][int(saved["n_train"]):]
        real = session_r2(saved["Z_te"].astype(np.float32), X_te)
        sh = session_r2(np.load(shuf)["Z_te"].astype(np.float32), X_te)
        real, sh = real[greedy_reorder(real)], sh[greedy_reorder(sh)]
        for rank in range(real.shape[0]):
            for j, p in enumerate(PREDICTORS):
                rows.append(dict(mouse=mouse, day=day, rank=rank + 1, predictor=p,
                                 r2_real=real[rank, j], r2_shuf=sh[rank, j]))
        print(f"  {mouse} {day_str}", flush=True)
    df = pd.DataFrame(rows)
    out = a.out or table_path("glm_r2_legacy_20hz.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"[OK] {df[['mouse', 'day']].drop_duplicates().shape[0]} sessions -> {out}")


if __name__ == "__main__":
    main()
