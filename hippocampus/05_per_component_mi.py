"""Per-session, per-component retention I(v; Z) (needs raw data and probe latents).

For each component v of X_ego (speed, omega, d_obj1..4) and two variables outside
X_ego used as checks (head direction as (sin, cos), time since session start):
Gaussian-copula MI between the test-fold latent Z and v on the same test frames
(valid_idx[n_train:]), bias-corrected by the mean over the five roll-shuffle
latents. The test-frame X_ego is rebuilt from the raw session and compared with
the saved one (rebuild_ok).

Output: results/hippocampus/per_component_mi_<regime>.csv
        (Figure 4 right and the per-component table, Supplementary "Per-component slopes").
Usage:  python hippocampus/05_per_component_mi.py --regime lambda1
"""
import argparse
import time

import numpy as np
import pandas as pd
from hippo_common import (
    REGIMES,
    add_session_args,
    jitter,
    list_probe_files,
    load_config,
    npz_day,
    parse_sessions,
    probes_dir,
    safe_gcmi,
    session_path,
    session_variables,
    table_path,
)

# name -> session_variables() columns; head direction enters as one 2-D variable.
VARIABLES = [
    ("speed", ["speed"]),
    ("omega", ["omega"]),
    ("head_direction", ["hd_sin", "hd_cos"]),
    ("d_obj1", ["d_obj1"]),
    ("d_obj2", ["d_obj2"]),
    ("d_obj3", ["d_obj3"]),
    ("d_obj4", ["d_obj4"]),
    ("time_since_start", ["time_since_start"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    add_session_args(ap)
    a = ap.parse_args()
    cfg = load_config()
    sr = float(cfg["sampling_rate_hz"])

    rows = []
    t0 = time.time()
    for f in list_probe_files(probes_dir(a.regime, a.probes_dir), parse_sessions(a.sessions)):
        d = np.load(f)
        mouse, day = str(d["mouse"]), npz_day(d)
        path = session_path(cfg, mouse, day)
        if not path.exists():
            print(f"  WARNING: missing raw session {path.name}, skipped")
            continue
        test_idx = d["valid_idx"][int(d["n_train"]):]
        feats = session_variables(path, sr)
        rebuilt = np.column_stack([feats[c][test_idx] for c in
                                   ("speed", "omega", "d_obj1", "d_obj2", "d_obj3", "d_obj4")])
        rebuild_ok = bool(np.allclose(d["X_te"], rebuilt.astype(np.float32), atol=1e-3))
        if not rebuild_ok:
            print(f"  WARNING: {mouse}/{day}D test-frame X_ego differs from the probe file")

        Z_te_shuf = d["Z_te_shuf"].astype(np.float32)
        # One generator per session: real Z, each shuffle, then each variable in order.
        rng = np.random.default_rng(0)
        Zj = jitter(d["Z_te"].astype(np.float32), rng)
        Zj_shuf = [jitter(Z_te_shuf[k], rng) for k in range(Z_te_shuf.shape[0])]
        row = {"mouse": mouse, "day": day, "rebuild_ok": rebuild_ok}
        for name, cols in VARIABLES:
            vj = jitter(np.column_stack([feats[c][test_idx] for c in cols]), rng)
            real = safe_gcmi(Zj, vj)
            shuf = float(np.nanmean([safe_gcmi(Zk, vj) for Zk in Zj_shuf]))
            row.update({f"I_{name}_real": real, f"I_{name}_shuf": shuf,
                        f"I_{name}_corr": real - shuf})
        rows.append(row)
        print(f"  {mouse} d{day}: " + "  ".join(
            f"{n[:6]}={row[f'I_{n}_corr']:+.2f}" for n, _ in VARIABLES), flush=True)

    df = pd.DataFrame(rows)
    out = a.out or table_path(f"per_component_mi_{a.regime}.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} sessions -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
