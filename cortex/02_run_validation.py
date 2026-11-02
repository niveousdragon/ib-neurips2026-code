"""Relevance-axis controls per session (lambda = 0, blocked CV, shift-shuffle null).

For every session, on the post_stim window, with the probe of 01_run_sessions.py:

  baseline   lambda = 1, stratified 5-fold CV (the primary reading)
  lambda0    lambda = 0: reconstruction-only autoencoder, same folds; asks
             whether category information appears in Z without supervision
  blocked    lambda = 1, five chronological blocks instead of random folds;
             asks whether random folds inflate I(Z;Y) via trial autocorrelation
  shuffle    lambda = 1, same random folds, activity circularly shifted along
             the trial axis by a random offset in [10, n_trials - 10), drawn
             anew in each fold; repeated --shuffles times (paper: 3). This is
             the null subtracted for bias correction.

Relevance is I(Z;Y) from a temperature-scaled logistic regression on Z (bits).

Needs the raw data (IB_DATA_CORTEX). Writes one file per mouse:
  results/cortex/ib_goltstein_validation_<mouse>.npz
"""
import argparse
import time

import numpy as np
from goltstein import PAPER_MICE, ae_config_kwargs, apply_window, discover_sessions, load_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ib.estimator import AEConfig, _mi_from_ce, train_ae_classifier
from ib.paths import data_dir, results_dir


def _ce_mi(Z_tr, Y_tr, Z_te, Y_te, n_classes=2):
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    lr.fit(Z_tr, Y_tr)
    acc = float(lr.score(Z_te, Y_te))
    logits = lr.decision_function(Z_te)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    return _mi_from_ce(logits, Y_te, n_classes=n_classes), acc


def _build_blocked_splits(n_trials, n_folds, Y, min_per_class=2):
    """Consecutive chronological blocks; a fold is skipped if either side has
    fewer than min_per_class trials of a class."""
    fold_size = n_trials // n_folds
    splits = []
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = test_start + fold_size if k < n_folds - 1 else n_trials
        te = np.arange(test_start, test_end)
        tr = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_trials)])
        if (Y[te] == 0).sum() < min_per_class or (Y[te] == 1).sum() < min_per_class:
            continue
        if (Y[tr] == 0).sum() < min_per_class or (Y[tr] == 1).sum() < min_per_class:
            continue
        splits.append((tr, te))
    return splits


def _build_random_splits(n_trials, n_folds, Y):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return list(skf.split(np.zeros(n_trials), Y))


def run_session(s, cfg, ae_kwargs, n_folds=5, n_shuffles=3):
    wstart, wend = cfg["windows"]["post_stim"]
    act = apply_window(s.activity, wstart, wend, cfg["pre_frames"])
    Y = s.labels.astype(int)

    if len(np.unique(Y)) < 2 or len(Y) < 20:
        return None
    if np.any(np.isnan(act)):
        return None

    n_trials = len(Y)
    # baseline, lambda0 and the shuffle null share the same random folds
    random_splits = _build_random_splits(n_trials, n_folds, Y)
    blocked_splits = _build_blocked_splits(n_trials, n_folds, Y)
    if len(blocked_splits) < 2:
        blocked_splits = None  # session too small for blocked CV

    def run_with(splits, class_weight):
        ice_vals, acc_vals = [], []
        for tr, te in splits:
            ae_cfg = AEConfig(**{**ae_kwargs, "class_weight": class_weight, "seed": 42})
            Z_tr, Z_te, _, _, _ = train_ae_classifier(
                act[tr], Y[tr], act[te], Y[te], n_classes=2, cfg=ae_cfg)
            ice, acc = _ce_mi(Z_tr, Y[tr], Z_te, Y[te])
            ice_vals.append(ice)
            acc_vals.append(acc)
        return float(np.mean(ice_vals)), float(np.std(ice_vals)), float(np.mean(acc_vals))

    base_ice, base_std, base_acc = run_with(random_splits, class_weight=1.0)
    l0_ice, l0_std, l0_acc = run_with(random_splits, class_weight=0.0)
    if blocked_splits is not None:
        blk_ice, blk_std, blk_acc = run_with(blocked_splits, class_weight=1.0)
    else:
        blk_ice = blk_std = blk_acc = float("nan")

    shuf_ices = []
    for s_idx in range(n_shuffles):
        rng_shuf = np.random.default_rng(seed=42 + s_idx)
        shuf_ice_vals = []
        for tr, te in random_splits:
            ae_cfg = AEConfig(**{**ae_kwargs, "class_weight": 1.0, "seed": 42})
            shift = rng_shuf.integers(low=10, high=n_trials - 10)
            act_shifted = np.roll(act, shift, axis=0)
            Z_tr, Z_te, _, _, _ = train_ae_classifier(
                act_shifted[tr], Y[tr], act_shifted[te], Y[te], n_classes=2, cfg=ae_cfg)
            ice, _ = _ce_mi(Z_tr, Y[tr], Z_te, Y[te])
            shuf_ice_vals.append(ice)
        shuf_ices.append(float(np.mean(shuf_ice_vals)))

    return {
        "mouse_id": s.mouse_id,
        "area": s.area,
        "date": s.date,
        "K": s.K,
        "n_trials": n_trials,
        "n_neurons": s.n_neurons,
        "behavior_acc": s.accuracy,
        "ice_baseline": base_ice, "ice_baseline_std": base_std, "acc_baseline": base_acc,
        "ice_lambda0": l0_ice, "ice_lambda0_std": l0_std, "acc_lambda0": l0_acc,
        "ice_blocked": blk_ice, "ice_blocked_std": blk_std, "acc_blocked": blk_acc,
        "ice_shuf_mean": float(np.mean(shuf_ices)),
        "ice_shuf_std": float(np.std(shuf_ices)),
        "ice_shuf_max": float(np.max(shuf_ices)),
        "ice_shuf_per_run": shuf_ices,
        "n_blocked_folds_valid": len(blocked_splits) if blocked_splits is not None else 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mice", nargs="*", default=list(PAPER_MICE))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--shuffles", type=int, default=3)
    p.add_argument("--max-sessions", type=int, default=0,
                   help="process only the first N sessions per mouse (0 = all; for testing)")
    p.add_argument("--ae-epochs", type=int, default=None,
                   help="override autoencoder epochs (for testing only)")
    args = p.parse_args()

    cfg = load_config()
    ae_kwargs = ae_config_kwargs(cfg)
    if args.ae_epochs is not None:
        ae_kwargs["ae_epochs"] = args.ae_epochs
    out_dir = results_dir("cortex")

    for mouse in args.mice:
        t0 = time.time()
        sessions = discover_sessions(data_dir("cortex"), mice=(mouse,),
                                     min_trials=cfg["min_trials"],
                                     min_neurons=cfg["min_neurons"])
        if args.max_sessions > 0:
            sessions = sessions[:args.max_sessions]
        print(f"{mouse}: {len(sessions)} sessions, folds={args.folds}, shuffles={args.shuffles}")
        results = []
        for i, s in enumerate(sessions):
            print(f"  [{i + 1}/{len(sessions)}] {s.mouse_id} {s.area:5s} {s.date} "
                  f"K={s.K:>2} ({s.n_neurons}n, {s.n_trials}tr)", end="", flush=True)
            try:
                r = run_session(s, cfg, ae_kwargs, n_folds=args.folds, n_shuffles=args.shuffles)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            if r is None:
                print(" SKIP")
                continue
            results.append(r)
            blk = "skip" if np.isnan(r["ice_blocked"]) else f"{r['ice_blocked']:.3f}"
            print(f"  base={r['ice_baseline']:.3f} l0={r['ice_lambda0']:.3f} "
                  f"blk={blk} shuf={r['ice_shuf_mean']:.3f}+/-{r['ice_shuf_std']:.3f}")
        out = out_dir / f"ib_goltstein_validation_{mouse}.npz"
        np.savez(out, results=np.array(results, dtype=object), allow_pickle=True)
        print(f"Saved {len(results)} sessions to {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
