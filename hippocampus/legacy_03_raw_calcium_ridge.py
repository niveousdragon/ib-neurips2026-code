"""Are speed, angular velocity and head direction in raw CA1 activity at all?
(needs raw data; Supplementary "Raw-calcium decoder confirms speed/omega/head direction
are not in T", 64 sessions).

Per session: Ridge regression (alpha = 1) of each target on the raw calcium of all
recorded neurons at 20 Hz, no autoencoder; targets speed, angular velocity (per frame),
head direction, and the distance to the first visited object as a positive control.
Frames and the contiguous 70/30 split are those of the earlier 20 Hz pipeline
(legacy_01_pilot_20hz.py: finite X and finite position 1..5 s later), recomputed here
from the raw session. Also stored: a KDE-based MI proxy H(target) - H(residual).

Output: results/hippocampus/raw_calcium_ridge.csv (mouse, day, var, R2, MI, H_marg, note)
Usage:  python hippocampus/legacy_03_raw_calcium_ridge.py [--sessions H01:1]
"""
import argparse

import numpy as np
import pandas as pd
from hippo_common import list_mice, load_config, parse_sessions, session_path, table_path
from legacy_01_pilot_20hz import common_valid_frames
from scipy.stats import gaussian_kde
from sklearn.linear_model import Ridge

DELTAS = [1.0, 2.0, 3.0, 4.0, 5.0]


def kde_entropy_1d(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 5 or x.std() < 1e-9:
        return 0.0
    try:
        log_p = np.log(np.clip(gaussian_kde(x)(x), 1e-300, None))
        return float(-log_p.mean() / np.log(2))
    except np.linalg.LinAlgError:
        return float(0.5 * np.log2(2 * np.pi * np.e * x.var() + 1e-30))


def targets(sess):
    x = sess["x"].astype(np.float64)
    y = sess["y"].astype(np.float64)
    hd = sess["headdirection"].astype(np.float64)
    d_obj1 = None
    for i in (1, 2, 3, 4):
        key = f"object{i}"
        if key in sess.files and sess[key].astype(bool).sum() > 0 and d_obj1 is None:
            zone = sess[key].astype(bool)
            d_obj1 = np.sqrt((x - x[zone].mean()) ** 2 + (y - y[zone].mean()) ** 2)
    return dict(speed=sess["speed"].astype(np.float64), ang_vel=np.gradient(np.unwrap(hd)),
                head_dir=hd, d_obj1=d_obj1)


def decode(T, target, idx, n_train):
    y = target[idx].astype(np.float64)
    T = T[idx]
    y_tr, y_te = y[:n_train], y[n_train:]
    if y_te.std() < 1e-9:
        return float("nan"), float("nan"), float("nan")
    h_marg = kde_entropy_1d(y_te)
    m = Ridge(alpha=1.0).fit(T[:n_train], y_tr)
    pred = m.predict(T[n_train:])
    return float(m.score(T[n_train:], y_te)), max(h_marg - kde_entropy_1d(y_te - pred), 0.0), h_marg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", default=None, help="e.g. H01:1 (default: all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    sr = float(cfg["sampling_rate_hz"])
    todo = parse_sessions(a.sessions) or [(m, d) for m in list_mice(cfg) for d in cfg["days"]]
    rows = []
    for mouse, day in todo:
        path = session_path(cfg, mouse, day)
        if not path.exists():
            continue
        sess = np.load(path)
        _, idx = common_valid_frames(sess, sr, DELTAS)
        n_train = int(cfg["train_frac"] * len(idx))
        T = sess["calcium"].T.astype(np.float64)
        for var, target in targets(sess).items():
            if target is None:
                rows.append(dict(mouse=mouse, day=day, var=var, R2=np.nan, MI=np.nan,
                                 H_marg=np.nan, note="missing"))
                continue
            r2, mi, h = decode(T, target, idx, n_train)
            rows.append(dict(mouse=mouse, day=day, var=var, R2=r2, MI=mi, H_marg=h, note=""))
        print(f"  {mouse} {day}D: " + "  ".join(f"{r['var']} {r['R2']:+.2f}" for r in rows[-4:]),
              flush=True)
    df = pd.DataFrame(rows)
    out = a.out or table_path("raw_calcium_ridge.csv")
    df.to_csv(out, index=False)
    print(df.groupby("var").R2.agg(["count", "median"]).round(3).to_string())
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
