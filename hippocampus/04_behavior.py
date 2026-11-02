"""Per-session behavioural readout of habituation (needs raw data).

From the full tracked session (all frames with finite x, y, speed):
  mean_speed, median_speed, locomotion_fraction, object_fraction (any object
  zone), rest_freeze_fraction (rest or freezing), rear_fraction,
  coverage_entropy_y_bin (entropy of 10 x 10 occupancy on every 5th valid frame,
  the probes' frame rate), distance_traveled (sum of steps, steps >= 0.1 arena
  units dropped as tracking jumps).
These are the covariates of the mixed-effects model and the habituation numbers
(Supplementary "Relevance-axis day-coefficient under behavioral-habituation controls").

Output: results/hippocampus/behavior.csv
Usage:  python hippocampus/04_behavior.py
"""
import argparse

import numpy as np
import pandas as pd
from hippo_common import (
    N_CLASSES,
    bin_2d,
    h_discrete,
    list_mice,
    load_config,
    session_path,
    table_path,
)


def session_metrics(path, downsample):
    sess = np.load(path)
    x = sess["x"].astype(np.float64)
    y = sess["y"].astype(np.float64)
    speed = sess["speed"].astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(speed)

    def flag(key):
        return sess[key].astype(bool) if key in sess.files else np.zeros_like(valid)

    objects, rest, freezing = flag("objects"), flag("rest"), flag("freezing")
    locomotion, rear = flag("locomotion"), flag("rear")
    Y = np.column_stack([x[valid], y[valid]])
    step = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    idx = np.where(valid)[0][::downsample]
    return {
        "n_valid_frames": int(valid.sum()),
        "mean_speed": float(speed[valid].mean()),
        "median_speed": float(np.median(speed[valid])),
        "locomotion_fraction": float(locomotion[valid].mean()),
        "object_fraction": float(objects[valid].mean()),
        "rest_freeze_fraction": float((rest | freezing)[valid].mean()),
        "rear_fraction": float(rear[valid].mean()),
        "coverage_entropy_y_bin": h_discrete(bin_2d(np.column_stack([x[idx], y[idx]])), N_CLASSES),
        "distance_traveled": float(step[step < 0.1].sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    rows = []
    for mouse in list_mice(cfg):
        for day in cfg["days"]:
            path = session_path(cfg, mouse, day)
            if not path.exists():
                continue
            m = session_metrics(path, int(cfg["downsample"]))
            m.update(mouse=mouse, day=day)
            rows.append(m)
            print(f"  {mouse} {day}D: speed {m['mean_speed']:.3f}  H {m['coverage_entropy_y_bin']:.2f}  "
                  f"distance {m['distance_traveled']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    out = a.out or table_path("behavior.csv")
    df.to_csv(out, index=False)
    print(df.groupby("day").mean(numeric_only=True).round(3).to_string())
    print(f"[OK] {len(df)} sessions -> {out}")


if __name__ == "__main__":
    main()
