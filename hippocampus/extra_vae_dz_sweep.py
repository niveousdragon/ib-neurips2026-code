"""Additional probes, not in the paper: trained-AE latent-size sweep (d_Z = 4, 8, 32) and
a variational autoencoder (d_Z = 16), on the paper's hippocampus pipeline (needs raw data
and, for the frame/split check, the lambda = 1 probe files).

Reused from the paper pipeline: data, matched neurons, 4 Hz frames, contiguous 70/30
split, per-neuron train z-score, five roll shuffles with seeds 42 + 1000 (k + 1)
(hippo_common); AE training (ib.ae_predict.train_ae_predict, lambda = 1, all other
hyperparameters from config.yaml, only latent_dim changes); both axes exactly as
02_ib_plane.py (confusion-matrix MI of the logistic decoder, Gaussian-copula retention,
same jitter and generator order, minus the shuffle mean). Every session's frames,
split, X_te and Y_te are checked against the saved lambda = 1 probe file first.

VAE: driada FlexibleVAE (encoder with 2 * d_Z outputs for mean and log-variance), the
same position head and lambda = 1, plus a KL term. The head reads the sampled z during
training (position information has to survive the posterior noise, as in a variational
information bottleneck); both axes use the posterior mean. driada's reconstruction loss
is the mean squared error over neurons while its KL is summed over latent dimensions;
the standard VAE objective (unit-variance Gaussian decoder) 0.5 N MSE + KL, divided by
0.5 N to keep the AE's reconstruction scale, is MSE + (2 / N) KL, so beta = 2 / N_matched
per session (a plain VAE, not a beta-VAE). Active latent dimensions (variance of the
mean over train frames > 0.01) are saved so posterior collapse is visible.

Criterion fixed before running: for each new probe and axis, the day effect has the
direction of the paper's AE (lambda = 1, d_Z = 16) on the 61 sessions: same sign of the
median per-mouse slope and one-sided Wilcoxon p < 0.5 in that direction (p < 0.05 is
reported, not required).

Outputs: results/hippocampus/vae_dz_sweep/<config>/sess_<mouse>_<day>D.npz,
row_<mouse>_<day>D.json; summary.csv, summary.log (--quick: .../vae_dz_sweep/quick/).
Usage:
  python hippocampus/extra_vae_dz_sweep.py [--resume] [--configs ae_dz4,ae_dz8,ae_dz32,vae_dz16]
                                           [--jobs 6] [--quick] [--summary-only] [--bench]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from hippo_common import (
    N_CLASSES,
    bin_2d,
    build_features,
    confusion_mi,
    jitter,
    list_mice,
    load_config,
    load_matching,
    probe_file,
    roll_shuffle_calcium,
    safe_gcmi,
    session_path,
    table_path,
    valid_frames,
    zscore_split,
)
from sessions import DROP3, cohort

from ib.paths import results_dir

warnings.filterwarnings("ignore")

LAMBDA = 1.0
CONFIGS = {
    "ae_dz4": dict(kind="ae", dz=4),
    "ae_dz8": dict(kind="ae", dz=8),
    "ae_dz32": dict(kind="ae", dz=32),
    "vae_dz16": dict(kind="vae", dz=16),
    # Not in the default run: the paper's AE retrained here gives the spread of
    # "same model, different hardware / run".
    "ae_dz16_rerun": dict(kind="ae", dz=16),
}
DEFAULT_CONFIGS = ["ae_dz4", "ae_dz8", "ae_dz32", "vae_dz16"]


def sessions_61(cfg):
    return [(m, d) for m in list_mice(cfg) for d in cfg["days"] if (m, d) not in DROP3]


# ---------------------------------------------------------------- data
def build_session(cfg, mouse, day):
    path = session_path(cfg, mouse, day)
    matched_idx = load_matching(cfg, mouse)[day]
    cal = np.load(path)["calcium"][matched_idx, :]
    X_full, Y_full, _ = build_features(path, float(cfg["sampling_rate_hz"]))
    valid = valid_frames(X_full, Y_full, int(cfg["downsample"]))
    n_tr = int(cfg["train_frac"] * len(valid))
    X = X_full[valid].astype(np.float32)
    Y = Y_full[valid].astype(np.float32)
    real = zscore_split(cal[:, valid].T, n_tr)
    shuf = [zscore_split(roll_shuffle_calcium(cal, cfg["seed"] + 1000 * (k + 1))[:, valid].T, n_tr)
            for k in range(cfg["n_shuffles"])]
    # Same frames, split and targets as the paper's primary run.
    p = np.load(probe_file("lambda1", mouse, day))
    assert np.array_equal(p["valid_idx"], valid), "valid_idx differs from the probe file"
    assert int(p["n_train"]) == n_tr and int(p["n_matched"]) == len(matched_idx)
    assert np.allclose(p["Y_te"], Y[n_tr:]) and np.allclose(p["X_te"], X[n_tr:])
    return dict(X_te=X[n_tr:], Y_tr=Y[:n_tr], Y_te=Y[n_tr:], real=real, shuf=shuf,
                n_matched=len(matched_idx), n_train=n_tr)


# ---------------------------------------------------------------- training
def train_ae(T_tr, Y_tr, T_te, Y_te, cfg, dz, seed, epochs):
    from ib.ae_predict import train_ae_predict
    Z_tr, Z_te, _, mse, _ = train_ae_predict(
        T_tr, Y_tr, T_te, Y_te, latent_dim=dz, hidden_dim=cfg["hidden_dim"], epochs=epochs,
        batch_size=cfg["batch_size"], lr=cfg["lr"], lambda_predict=LAMBDA, seed=seed,
        predict_hidden=cfg["predict_hidden"])
    return Z_tr.astype(np.float32), Z_te.astype(np.float32), float(mse), {}


def train_vae(T_tr, Y_tr, T_te, Y_te, cfg, dz, seed, epochs):
    """Training loop of train_ae_predict with driada FlexibleVAE; returns the posterior mean."""
    import torch
    import torch.nn as nn
    from driada.dim_reduction.flexible_ae import FlexibleVAE
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_neur = T_tr.shape[1]
    beta = 2.0 / n_neur
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = FlexibleVAE(input_dim=n_neur, latent_dim=dz, hidden_dim=cfg["hidden_dim"],
                      loss_components=[{"name": "reconstruction", "weight": 1.0},
                                       {"name": "beta_vae", "weight": 1.0, "beta": beta}],
                      device=dev).to(dev)
    ph = cfg["predict_hidden"]
    head = (nn.Sequential(nn.Linear(dz, ph), nn.ReLU(), nn.Linear(ph, Y_tr.shape[1])) if ph > 0
            else nn.Linear(dz, Y_tr.shape[1])).to(dev)
    opt = torch.optim.Adam(list(vae.parameters()) + list(head.parameters()), lr=cfg["lr"])
    Ttr = torch.tensor(T_tr, device=dev)
    Ytr = torch.tensor(Y_tr, device=dev)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Ttr, Ytr),
                                         batch_size=cfg["batch_size"], shuffle=True)
    vae.train()
    head.train()
    kl_last = 0.0
    for _ in range(epochs):
        kl_sum, nb = 0.0, 0
        for t_b, y_b in loader:
            z, mu, lv = vae.encode(t_b)
            recon = vae.decode(z)
            # The terms of driada's ReconstructionLoss (MSE, mean) and BetaVAELoss(beta),
            # written out because the head needs the same sampled z.
            rec = ((recon - t_b) ** 2).mean()
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / mu.shape[0]
            pred = ((head(z) - y_b) ** 2).mean()
            loss = rec + beta * kl + LAMBDA * pred
            opt.zero_grad()
            loss.backward()
            opt.step()
            kl_sum += float(kl.detach())
            nb += 1
        kl_last = kl_sum / max(nb, 1)
    vae.eval()
    with torch.no_grad():
        _, mu_tr, _ = vae.encode(Ttr)
        _, mu_te, _ = vae.encode(torch.tensor(T_te, device=dev))
        mse = float(((vae.decode(mu_te).cpu().numpy() - T_te) ** 2).mean())
        mu_tr, mu_te = mu_tr.cpu().numpy(), mu_te.cpu().numpy()
    return mu_tr.astype(np.float32), mu_te.astype(np.float32), mse, dict(
        beta=beta, kl_nats=kl_last, n_active=int((mu_tr.var(0) > 0.01).sum()))


# ---------------------------------------------------------------- axes
def paper_axes(Z_tr, Z_te, Zs_tr, Zs_te, Y_tr, Y_te, X_te):
    """The per-session computation of 02_ib_plane.py (logistic decoder)."""
    b_tr, b_te = bin_2d(Y_tr), bin_2d(Y_te)
    rel, h_y = confusion_mi(Z_tr, b_tr, Z_te, b_te, N_CLASSES)
    rel_s = [confusion_mi(Zs_tr[k], b_tr, Zs_te[k], b_te, N_CLASSES)[0] for k in range(Zs_tr.shape[0])]
    rng = np.random.default_rng(0)
    Xj = jitter(X_te, rng)
    ret = safe_gcmi(Xj, jitter(Z_te, rng))
    ret_s = [safe_gcmi(Xj, jitter(Zs_te[k], rng)) for k in range(Zs_te.shape[0])]
    return dict(H_Y_bin=h_y, I_zy_bin_real=rel, I_zy_bin_shuf_mean=float(np.nanmean(rel_s)),
                I_zy_bin_corr=rel - float(np.nanmean(rel_s)), I_xz_real=ret,
                I_xz_shuf_mean=float(np.nanmean(ret_s)), I_xz_corr=ret - float(np.nanmean(ret_s)))


# ---------------------------------------------------------------- one session
def run_one(cname, mouse, day, out_root, epochs, n_threads):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(n_threads)
    threadpool_limits(n_threads)  # BLAS threads of the logistic regressions too
    cdir = out_root / cname
    cdir.mkdir(parents=True, exist_ok=True)
    spec = CONFIGS[cname]
    cfg = load_config()
    t0 = time.time()
    S = build_session(cfg, mouse, day)
    fn = train_ae if spec["kind"] == "ae" else train_vae
    seed = int(cfg["seed"])
    Z_tr, Z_te, mse, meta = fn(S["real"][0], S["Y_tr"], S["real"][1], S["Y_te"], cfg,
                               spec["dz"], seed, epochs)
    Zs_tr, Zs_te, mses, metas = [], [], [], []
    for k in range(cfg["n_shuffles"]):
        a, b, m, mt = fn(S["shuf"][k][0], S["Y_tr"], S["shuf"][k][1], S["Y_te"], cfg,
                         spec["dz"], seed + 1000 * (k + 1), epochs)
        Zs_tr.append(a)
        Zs_te.append(b)
        mses.append(m)
        metas.append(mt)
    Zs_tr, Zs_te = np.stack(Zs_tr), np.stack(Zs_te)
    t_train = time.time() - t0
    axes = paper_axes(Z_tr, Z_te, Zs_tr, Zs_te, S["Y_tr"], S["Y_te"], S["X_te"])
    np.savez(cdir / f"sess_{mouse}_{day}D.npz", Z_tr=Z_tr, Z_te=Z_te, Z_tr_shuf=Zs_tr,
             Z_te_shuf=Zs_te, recon_mse=mse, recon_mse_shuf=np.array(mses))
    row = dict(config=cname, mouse=mouse, day=day, d_z=spec["dz"], kind=spec["kind"],
               epochs=epochs, n_matched=S["n_matched"], n_train=S["n_train"], recon_mse=mse,
               recon_mse_shuf_mean=float(np.mean(mses)), sec_train=round(t_train, 1),
               sec_total=round(time.time() - t0, 1), **axes)
    if spec["kind"] == "vae":
        row.update(beta=meta["beta"], kl_nats=meta["kl_nats"], n_active=meta["n_active"],
                   n_active_shuf_mean=float(np.mean([m["n_active"] for m in metas])))
    tmp = cdir / f"row_{mouse}_{day}D.json.tmp"
    tmp.write_text(json.dumps(row), encoding="utf-8")
    os.replace(tmp, cdir / f"row_{mouse}_{day}D.json")
    return (f"{cname} {mouse}/{day}D rel {axes['I_zy_bin_corr']:+.3f} "
            f"ret {axes['I_xz_corr']:+.3f} {row['sec_total']:.0f}s")


# ---------------------------------------------------------------- summary
def per_mouse_slopes(d, col):
    from scipy import stats
    out = {}
    for m, g in d.groupby("mouse"):
        if g.day.nunique() > 1:
            out[m] = stats.linregress(g.day, g[col]).slope
    return pd.Series(out, dtype=float)


def summarize(out_root, configs):
    from scipy import stats
    log = []

    def P(s=""):
        print(s, flush=True)
        log.append(s)

    paper = cohort(pd.read_csv(table_path("ib_plane_lambda1.csv")), "drop3")
    frames = {"ae_dz16_paper": paper.assign(config="ae_dz16_paper")}
    for c in configs:
        rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((out_root / c).glob("row_*.json"))]
        if rows:
            frames[c] = pd.DataFrame(rows)
    P("CRITERION: for each new probe and axis, same day-effect direction as the paper AE "
      "(lambda=1, d_Z=16) on 61 sessions: sign(median per-mouse slope) equal AND one-sided "
      "Wilcoxon over mice in that direction p < 0.5. p < 0.05 reported, not required.")
    ref = frames["ae_dz16_paper"]
    ref_sign, out, all_ok = {}, [], True
    for name, d in frames.items():
        extra = (f"  mean active dims (real) {d.n_active.mean():.1f}, beta median {d.beta.median():.4f}"
                 if "n_active" in d else "")
        P(f"\n[{name}] sessions: {len(d)}{extra}")
        for ax, col in (("relevance", "I_zy_bin_corr"), ("retention", "I_xz_corr")):
            sl = per_mouse_slopes(d, col).dropna()
            med = float(sl.median()) if len(sl) else np.nan
            if name == "ae_dz16_paper":
                ref_sign[ax] = np.sign(med)
            s0 = ref_sign[ax]
            alt = "greater" if s0 > 0 else "less"
            try:
                p1 = float(stats.wilcoxon(sl, alternative=alt).pvalue)
            except ValueError:
                p1 = np.nan
            n_same = int((np.sign(sl) == s0).sum())
            mm = d[["mouse", "day", col]].merge(ref[["mouse", "day", col]], on=["mouse", "day"],
                                                suffixes=("", "_ref"))
            rho = (stats.spearmanr(mm[col], mm[col + "_ref"]).correlation
                   if len(mm) > 2 and name != "ae_dz16_paper" else np.nan)
            ok = bool(np.sign(med) == s0 and np.isfinite(p1) and p1 < 0.5)
            if name != "ae_dz16_paper":
                all_ok &= ok
            out.append(dict(probe=name, axis=ax, n_sessions=len(d), n_mice=len(sl),
                            mean_value=float(d[col].mean()), median_slope=med,
                            n_mice_same_sign_as_AE=n_same, p_one_sided_AE_direction=p1,
                            spearman_vs_AE16=rho, criterion_ok=ok if name != "ae_dz16_paper" else None))
            tag = "" if name == "ae_dz16_paper" else ("[OK]" if ok else "[FAILED]")
            P(f"  {ax:9s} mean {d[col].mean():.3f}  median slope {med:+.4f}  same sign {n_same}/{len(sl)}"
              f"  p1({alt}) {p1:.3f}  Spearman vs AE16 {rho:.2f}  {tag}")
    pd.DataFrame(out).to_csv(out_root / "summary.csv", index=False)
    n_new = len(frames) - 1
    complete = all(len(frames.get(c, [])) == 61 for c in configs)
    P(f"\nCRITERION OVERALL ({n_new} new probes, {'complete 61/61' if complete else 'INCOMPLETE cohort'}): "
      + ("[OK]" if all_ok and n_new > 0 else "[FAILED]"))
    (out_root / "summary.log").write_text("\n".join(log), encoding="utf-8")


# ---------------------------------------------------------------- main
def bench():
    """Seconds per epoch for one real-calcium network (AE and VAE, d_Z 16), H01 day 1."""
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)
    cfg = load_config()
    S = build_session(cfg, "H01", 1)
    train_ae(S["real"][0], S["Y_tr"], S["real"][1], S["Y_te"], cfg, 16, 42, 1)  # warm-up
    for kind, fn in (("ae", train_ae), ("vae", train_vae)):
        dts = []
        for ep in (2, 12):
            t0 = time.time()
            fn(S["real"][0], S["Y_tr"], S["real"][1], S["Y_te"], cfg, 16, 42, ep)
            dts.append(time.time() - t0)
        print(f"BENCH {kind} d16: {(dts[1] - dts[0]) / 10:.3f} s/epoch, one net, 1 thread, "
              f"cuda {torch.cuda.is_available()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", action="store_true", help="skip sessions whose row json exists")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--jobs", type=int, default=6, help="parallel sessions (CPU processes)")
    ap.add_argument("--quick", action="store_true",
                    help="1 epoch, 2 sessions, configs ae_dz4,vae_dz16 unless --configs")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    configs = [c for c in (a.configs or ("ae_dz4,vae_dz16" if a.quick else ",".join(DEFAULT_CONFIGS))).split(",") if c]
    out_root = results_dir("hippocampus", "vae_dz_sweep", *(("quick",) if a.quick else ()))
    if a.bench:
        bench()
        return
    cfg = load_config()
    epochs = 1 if a.quick else int(cfg["ae_epochs"])
    sess = [("H01", 1), ("H01", 2)] if a.quick else sessions_61(cfg)
    print(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} configs={configs} sessions={len(sess)} "
          f"epochs={epochs} jobs={a.jobs} resume={a.resume} out={out_root}", flush=True)
    if not a.summary_only:
        todo = [(c, m, d) for c in configs for (m, d) in sess
                if not (a.resume and (out_root / c / f"row_{m}_{d}D.json").exists())]
        print(f"to run: {len(todo)} (skipped {len(configs) * len(sess) - len(todo)})", flush=True)
        thr = max(1, (os.cpu_count() or 8) // max(a.jobs, 1))  # tiny nets: 1 thread per process is fastest
        t0 = time.time()
        if a.jobs > 1 and len(todo) > 1:
            from joblib import Parallel, delayed
            gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
                delayed(run_one)(c, m, d, out_root, epochs, thr) for c, m, d in todo)
        else:
            gen = (run_one(c, m, d, out_root, epochs, thr) for c, m, d in todo)
        for i, msg in enumerate(gen, 1):
            print(f"  [{i}/{len(todo)}] {msg}  (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    summarize(out_root, configs)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
