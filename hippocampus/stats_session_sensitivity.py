# -*- coding: utf-8 -*-
"""How much do the sessions with a constant-zero object distance carry the retention results?
H39 day 4 and H06 day 3 keep Z; each object never visited in the session takes its position from the
mouse's other sessions, so the distance is real. Compared with dropping those sessions."""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from hippo_common import jitter, safe_gcmi, per_mouse_slopes, _object_centres, load_config, session_path, probes_dir
from ib.paths import results_dir
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--probes-dir", default=None, help="saved probe latents (default: results/hippocampus/probes/lambda1)")
args = ap.parse_args()
CFG = load_config()
RES = results_dir("hippocampus")
PROBE = probes_dir("lambda1", args.probes_dir)
ex3 = {("H32", 4), ("H06", 4), ("H33", 3)}
def centres(m, d):
    s = np.load(session_path(CFG, m, d))
    return s, _object_centres(s, s["x"].astype(float), s["y"].astype(float))
pc = pd.read_csv(RES / "per_component_mi_lambda1.csv"); ip = pd.read_csv(RES / "ib_plane_lambda1.csv")
new = {}
for m, d in (("H39", 4), ("H06", 3)):
    f = [x for x in os.listdir(PROBE) if x.endswith(".npz") and m in x and ("%dD" % d) in x]
    p = np.load(PROBE / f[0]); idx = p["valid_idx"][int(p["n_train"]):]
    s, c = centres(m, d); x, y = s["x"].astype(float), s["y"].astype(float)
    others = [centres(m, o)[1] for o in (1, 2, 3, 4) if o != d]
    X = p["X_te"].astype(float).copy()
    filled = []
    for i in range(4):
        if c[i] is None:
            xy = [o[i] for o in others if o[i] is not None]
            cx, cy = np.mean([a[0] for a in xy]), np.mean([a[1] for a in xy])
            X[:, 2 + i] = np.sqrt((x[idx] - cx) ** 2 + (y[idx] - cy) ** 2); filled.append(i + 1)
    print(m, d, "objects filled from other days:", filled)
    Zs = p["Z_te_shuf"].astype(np.float32); Z = p["Z_te"].astype(np.float32)
    r = {}
    rng = np.random.default_rng(0)
    Zj = jitter(Z, rng); Zsj = [jitter(Zs[k], rng) for k in range(Zs.shape[0])]
    for i in range(4):
        v = jitter(X[:, 2 + i], rng)
        r["I_d_obj%d_corr" % (i + 1)] = safe_gcmi(Zj, v) - float(np.nanmean([safe_gcmi(k, v) for k in Zsj]))
    rng = np.random.default_rng(0)
    Xj = jitter(X, rng); ret = safe_gcmi(Xj, jitter(Z, rng))
    rs = [safe_gcmi(Xj, jitter(Zs[k], rng)) for k in range(Zs.shape[0])]
    r["I_xz_corr"] = ret - float(np.nanmean(rs))
    new[(m, d)] = r
    old = ip[(ip.mouse == m) & (ip.day == d)].I_xz_corr.values[0]
    print("  retention corr cached %.3f -> filled %.3f;" % (old, r["I_xz_corr"]), {k: round(v, 3) for k, v in r.items()})
def table(mode):
    a = pc.copy(); b = ip.copy()
    keep = lambda df, e: df[[(m, int(x)) not in e for m, x in zip(df.mouse, df.day)]].copy()
    if mode == "drop3":
        return keep(a, ex3), keep(b, ex3)
    if mode == "drop H39d4+H06d3":
        e = ex3 | set(new); return keep(a, e), keep(b, e)
    a, b = keep(a, ex3), keep(b, ex3)
    for (m, d), r in new.items():
        for k, v in r.items():
            if k == "I_xz_corr": b.loc[(b.mouse == m) & (b.day == d), k] = v
            else: a.loc[(a.mouse == m) & (a.day == d), k] = v
    return a, b
for mode in ("drop3", "filled distances", "drop H39d4+H06d3"):
    a, b = table(mode)
    print("==", mode)
    for i in (1, 2, 3, 4):
        s = per_mouse_slopes(a, "I_d_obj%d_corr" % i, min_days=3).values
        print("  d_obj%d mean slope %+.3f declining %d/%d one-sided p %.3f" % (i, s.mean(), (s < 0).sum(), len(s), stats.wilcoxon(s, alternative="less").pvalue))
    s = per_mouse_slopes(b, "I_xz_corr").values
    print("  retention declining %d/%d p %.3f mean slope %+.3f" % ((s < 0).sum(), len(s), stats.wilcoxon(s, alternative="less").pvalue, s.mean()))
    d4 = b[b.day == 4]
    print("  day-4 retention~relevance Spearman %.2f p %.3f n=%d" % (*stats.spearmanr(d4.I_xz_corr, d4.I_zy_bin_corr), len(d4)))
    m = d4.sort_values("I_zy_bin_corr"); k = len(m) // 2
    print("  day-4 MWU worse vs better p %.3f" % stats.mannwhitneyu(m.I_xz_corr.values[:k], m.I_xz_corr.values[-k:], alternative="two-sided").pvalue)
