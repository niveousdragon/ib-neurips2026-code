"""Shift-shuffle null for the input axis R^2(Z -> X_phys).

Same shuffle as the relevance-axis null in 02_run_validation.py: per session
and fold, the activity is circularly shifted along the trial axis by a random
offset in [10, n_trials - 10), the autoencoder is retrained on the shifted
activity, and RidgeCV R^2 is scored against the real X_phys. Repeated
--shuffles times (paper: 5). K = 2 sessions are skipped (X_phys degenerate),
leaving 69 of the 90 sessions.

Needs the raw data (IB_DATA_CORTEX). Writes
  results/cortex/xaxis_shuffle_per_session.csv
with columns mouse, area, date, K, n_trials, R2_real, R2_shuf_mean,
R2_shuf_std, R2raw_real, R2raw_shuf_mean, R2raw_shuf_std (R2 = floored at 0
per fold, R2raw = unfloored).
"""
import argparse
import time

import numpy as np
import pandas as pd
from goltstein import PAPER_MICE, ae_config_kwargs, apply_window, discover_sessions, load_config
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ib.estimator import AEConfig, train_ae_classifier
from ib.paths import data_dir, results_dir


def _x_probes(Z_tr, Z_te, X_tr, X_te):
    """(R^2 floored at 0, raw R^2) on the held-out fold."""
    if Z_tr is None or len(np.unique(Z_tr.std(axis=0))) == 0:
        return float("nan"), float("nan")
    sc = StandardScaler().fit(Z_tr)
    Ztr = sc.transform(Z_tr)
    Zte = sc.transform(Z_te)
    alphas = np.logspace(-2, 3, 20)
    try:
        ridge = RidgeCV(alphas=alphas, cv=min(5, max(2, len(Ztr) // 4))).fit(Ztr, X_tr)
        r2_raw = float(ridge.score(Zte, X_te))
        r2_clip = float(np.clip(r2_raw, 0.0, 1.0))
    except Exception:
        r2_raw = float("nan")
        r2_clip = float("nan")
    return r2_clip, r2_raw


def run_session(s, cfg, ae_kwargs, n_folds=5, n_shuffles=5):
    wstart, wend = cfg["windows"]["post_stim"]
    act = apply_window(s.activity, wstart, wend, cfg["pre_frames"])
    Y = s.labels.astype(int)
    X = s.stim_phys

    if len(np.unique(Y)) < 2 or len(Y) < 20:
        return None
    if np.any(np.isnan(act)):
        return None
    if s.K <= 2:
        return None

    n_trials = len(Y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(n_trials), Y))
    kw = {**ae_kwargs, "class_weight": 1.0, "seed": 42}

    real_r2c, real_r2r = [], []
    for tr, te in splits:
        Z_tr, Z_te, _, _, _ = train_ae_classifier(
            act[tr], Y[tr], act[te], Y[te], n_classes=2, cfg=AEConfig(**kw))
        r2c, r2r = _x_probes(Z_tr, Z_te, X[tr], X[te])
        real_r2c.append(r2c)
        real_r2r.append(r2r)

    sh_r2c, sh_r2r = [], []
    for s_idx in range(n_shuffles):
        rng = np.random.default_rng(seed=42 + s_idx)
        f_r2c, f_r2r = [], []
        for tr, te in splits:
            shift = int(rng.integers(low=10, high=n_trials - 10))
            act_shifted = np.roll(act, shift, axis=0)
            Z_tr, Z_te, _, _, _ = train_ae_classifier(
                act_shifted[tr], Y[tr], act_shifted[te], Y[te], n_classes=2,
                cfg=AEConfig(**kw))
            r2c, r2r = _x_probes(Z_tr, Z_te, X[tr], X[te])
            f_r2c.append(r2c)
            f_r2r.append(r2r)
        sh_r2c.append(float(np.nanmean(f_r2c)))
        sh_r2r.append(float(np.nanmean(f_r2r)))

    return {
        "mouse": s.mouse_id, "area": s.area, "date": s.date, "K": s.K,
        "n_trials": n_trials,
        "R2_real": float(np.nanmean(real_r2c)),
        "R2_shuf_mean": float(np.nanmean(sh_r2c)),
        "R2_shuf_std": float(np.nanstd(sh_r2c)),
        "R2raw_real": float(np.nanmean(real_r2r)),
        "R2raw_shuf_mean": float(np.nanmean(sh_r2r)),
        "R2raw_shuf_std": float(np.nanstd(sh_r2r)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mice", nargs="*", default=list(PAPER_MICE))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--shuffles", type=int, default=5)
    p.add_argument("--max-sessions", type=int, default=0,
                   help="process only the first N sessions (0 = all; for testing)")
    p.add_argument("--ae-epochs", type=int, default=None,
                   help="override autoencoder epochs (for testing only)")
    args = p.parse_args()

    cfg = load_config()
    ae_kwargs = ae_config_kwargs(cfg)
    if args.ae_epochs is not None:
        ae_kwargs["ae_epochs"] = args.ae_epochs
    out_path = results_dir("cortex") / "xaxis_shuffle_per_session.csv"

    sessions = discover_sessions(data_dir("cortex"), mice=tuple(args.mice),
                                 min_trials=cfg["min_trials"], min_neurons=cfg["min_neurons"])
    if args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]
    print(f"loaded {len(sessions)} sessions; folds={args.folds}, shuffles={args.shuffles}")

    rows = []
    t0 = time.time()
    for i, s in enumerate(sessions, 1):
        t_sess = time.time()
        try:
            r = run_session(s, cfg, ae_kwargs, n_folds=args.folds, n_shuffles=args.shuffles)
        except Exception as e:
            print(f"  [{i}/{len(sessions)}] {s.mouse_id} {s.area} {s.date}: FAILED ({e})")
            continue
        if r is None:
            print(f"  [{i}/{len(sessions)}] {s.mouse_id} {s.area} {s.date}: skipped")
            continue
        rows.append(r)
        print(f"  [{i}/{len(sessions)}] {r['mouse']} {r['area']} {r['date']} K={r['K']}: "
              f"R2 real={r['R2_real']:+.3f} shuf={r['R2_shuf_mean']:+.3f} "
              f"({time.time() - t_sess:.1f}s)")
        # written after every session so a crashed long run keeps its finished rows
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nsaved {len(rows)} sessions to {out_path}")
    print(f"total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
