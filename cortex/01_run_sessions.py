"""Per-session information-plane coordinates for the visual-cortex data.

For every session a classification autoencoder (probe Z, d_Z = 8) is trained
on the trial-averaged post-stimulus activity in stratified 5-fold
cross-validation, and on each held-out fold:

  relevance  I_ce_zy  usable information I(Z;Y) on the category, from a
                      temperature-scaled logistic regression on Z (bits)
  input      R2_zx    RidgeCV decoding score R^2(Z -> X_phys),
                      X_phys = (sin 2theta, cos 2theta, log2 SF), floored at 0
                      per fold before averaging (R2_zx_raw keeps the raw value)
  decoder    acc_logreg_y  held-out category accuracy of that regression

Sessions with K = 2 stimuli (prototype stage) have only two X_phys values,
identical to the category, so R^2 is not computed there.

Needs the raw data (IB_DATA_CORTEX). Writes one file per mouse:
  results/cortex/ib_goltstein_<mouse>_kf5.npz

Usage:
  python cortex/01_run_sessions.py                     # five paper mice
  python cortex/01_run_sessions.py --mice K01 --windows pre_stim post_stim
The paper cache was produced with the defaults (5 folds, seed 42, post_stim
window); K01 was additionally run with all five windows, which does not change
its post_stim values because every autoencoder is reseeded.
"""
import argparse
import time

import numpy as np
from goltstein import PAPER_MICE, ae_config_kwargs, apply_window, discover_sessions, load_config
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from ib.estimator import AEConfig, _mi_from_ce, train_ae_classifier
from ib.paths import data_dir, results_dir


def compute_probes(Z_train, Z_test, Y_train, Y_test, X_train, X_test, degenerate_x=False):
    """Information-plane readings on one train/test split of a frozen Z."""
    out = {}

    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    lr.fit(Z_train, Y_train)
    out["acc_logreg_y"] = float(lr.score(Z_test, Y_test))
    logits_bin = lr.decision_function(Z_test)
    if logits_bin.ndim == 1:  # sklearn returns one column for two classes
        logits = np.column_stack([-logits_bin, logits_bin])
    else:
        logits = logits_bin
    out["I_ce_zy"] = _mi_from_ce(logits, Y_test, n_classes=2)

    if degenerate_x:
        out["R2_zx"] = float("nan")
        out["R2_zx_raw"] = float("nan")
    else:
        # Z is standardised so one ridge penalty grid fits all sessions;
        # X_phys keeps its natural scale.
        scaler = StandardScaler().fit(Z_train)
        Z_tr_s = scaler.transform(Z_train)
        Z_te_s = scaler.transform(Z_test)
        alphas = np.logspace(-2, 3, 20)
        ridge = RidgeCV(alphas=alphas, cv=min(5, len(Z_train) // 4)).fit(Z_tr_s, X_train)
        r2_raw = float(ridge.score(Z_te_s, X_test))
        # A negative held-out R^2 means no linearly decodable stimulus
        # information, so it counts as 0 on the input axis.
        out["R2_zx"] = float(np.clip(r2_raw, 0.0, 1.0))
        out["R2_zx_raw"] = r2_raw
    return out


def run_session_window(session, frame_start, frame_end, pre_frames, n_classes, test_frac,
                       ae_kwargs, seeds, n_folds=5):
    """All folds x seeds of one session in one window; None if unusable."""
    act = apply_window(session.activity, frame_start, frame_end, pre_frames)
    Y = session.labels
    X = session.stim_phys

    if len(np.unique(Y)) < 2 or len(Y) < 10:
        return None
    if np.any(np.isnan(act)):
        return None

    degenerate_x = session.K <= 2

    if n_folds > 1:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(act, Y))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
        splits = [next(sss.split(act, Y))]

    per_run = []
    for fold_idx, (tr, te) in enumerate(splits):
        for seed in seeds:
            cfg_run = AEConfig(**{**ae_kwargs, "seed": seed})
            Z_train, Z_test, acc_z_mlp, recon_mse, _ = train_ae_classifier(
                act[tr], Y[tr], act[te], Y[te], n_classes=n_classes, cfg=cfg_run)
            probes = compute_probes(Z_train, Z_test, Y[tr], Y[te], X[tr], X[te],
                                    degenerate_x=degenerate_x)
            probes["acc_z_mlp"] = float(acc_z_mlp)
            probes["recon_mse"] = float(recon_mse)
            probes["seed"] = seed
            probes["fold"] = fold_idx
            per_run.append(probes)

    agg = {}
    keys = [k for k in per_run[0].keys() if k not in ("seed", "fold")]
    for k in keys:
        vals = np.array([p[k] for p in per_run], dtype=float)
        agg[k] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))

    return {
        "mouse_id": session.mouse_id,
        "area": session.area,
        "date": session.date,
        "n_neurons": session.n_neurons,
        "n_trials": session.n_trials,
        "K": session.K,
        "degenerate_x": degenerate_x,
        "behavior_acc": session.accuracy,
        "n_seeds": len(seeds),
        "n_folds": n_folds,
        **agg,
        "per_run": per_run,
    }


def compute_all(cfg, sessions, window_names, seeds, n_folds, ae_kwargs):
    windows = {k: v for k, v in cfg["windows"].items() if k in window_names}
    results = {}
    for wname, (wstart, wend) in windows.items():
        print(f"\n=== Window: {wname} [frames {wstart} to {wend}], "
              f"seeds={list(seeds)}, folds={n_folds} ===")
        results[wname] = []
        for i, s in enumerate(sessions):
            print(f"  [{i + 1}/{len(sessions)}] {s.mouse_id} {s.area:4s} {s.date} "
                  f"K={s.K:2d} ({s.n_neurons}n, {s.n_trials}tr)", end="", flush=True)
            try:
                r = run_session_window(s, wstart, wend, cfg["pre_frames"], cfg["n_classes"],
                                       cfg["test_fraction"], ae_kwargs, seeds, n_folds=n_folds)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            if r is None:
                print(" SKIP")
                continue
            results[wname].append(r)
            r2 = "DEGEN" if r["degenerate_x"] else f"{r['R2_zx']:+.3f}"
            print(f"  I_ce={r['I_ce_zy']:.3f}  R2={r2}  acc={r['acc_logreg_y']:.1%}")
    return results


def save_results(results, path):
    out = {"window_names": list(results.keys())}
    for wname, rlist in results.items():
        out[f"results_{wname}"] = rlist
    np.savez(path, **out, allow_pickle=True)
    print(f"\nSaved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mice", nargs="*", default=list(PAPER_MICE))
    p.add_argument("--windows", nargs="*", default=["post_stim"],
                   help="named windows from config.yaml (default: post_stim)")
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--folds", type=int, default=5,
                   help="stratified k-fold CV; 1 = single test_fraction split")
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
        print(f"{mouse}: {len(sessions)} sessions")
        if not sessions:
            continue
        results = compute_all(cfg, sessions, args.windows, tuple(args.seeds), args.folds,
                              ae_kwargs)
        tag = f"_{mouse}"
        if len(args.seeds) > 1:
            tag += f"_ms{len(args.seeds)}"
        if args.folds > 1:
            tag += f"_kf{args.folds}"
        save_results(results, out_dir / f"ib_goltstein{tag}.npz")
        print(f"{mouse}: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
