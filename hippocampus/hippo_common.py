"""Data layout, features, shuffles and estimators shared by the hippocampus scripts.

Raw data (not redistributed) lives under ib.paths.data_dir("hippocampus"):

    sessions/NOF_<mouse>_<day>D syn data.npz   one synchronized session, 20 or 30 Hz:
        calcium (n_neurons, n_frames), x, y (arena coordinates scaled to [0, 1]),
        speed, headdirection, object1..object4 (zone masks), objects, rest,
        freezing, locomotion, rear, ...
    matching/NOF_<mouse>.csv                   cross-day neuron matching (CellReg):
        one row per tracked cell, one column per day (1..4), value = 1-based row
        of that cell in the day's calcium array, 0 if not found that day.

Folder and file names are set in config.yaml (section "data"); another config file
can be given with the environment variable IB_HIPPOCAMPUS_CONFIG.
Per-session probe outputs are written under results_dir("hippocampus", "probes", <regime>).
"""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ib.paths import data_dir, results_dir

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"

# Relevance target: current (x, y) in a 10 x 10 grid over the arena.
N_BINS = 10
N_CLASSES = N_BINS * N_BINS
# Relative Gaussian jitter added before Gaussian-copula MI so that tied values
# (e.g. a constant object distance) do not break the rank transform.
JITTER = 1e-4

# AE regimes: lambda is the weight of the position-prediction loss.
REGIMES = {"lambda0": 0.0, "lambda1": 1.0}

# X_ego column order in the probe files.
X_EGO = ["speed", "omega", "d_obj1", "d_obj2", "d_obj3", "d_obj4"]


# ----------------------------------------------------------------- config / files
def load_config(path=None) -> dict:
    """config.yaml next to the scripts, or the file named by IB_HIPPOCAMPUS_CONFIG."""
    with open(path or os.environ.get("IB_HIPPOCAMPUS_CONFIG") or CONFIG) as f:
        return yaml.safe_load(f)


def session_path(cfg: dict, mouse: str, day: int) -> Path:
    d = cfg["data"]
    return (data_dir("hippocampus") / d["sessions_subdir"]
            / d["session_file"].format(mouse=mouse, day=day))


def matching_path(cfg: dict, mouse: str) -> Path:
    d = cfg["data"]
    return (data_dir("hippocampus") / d["matching_subdir"]
            / d["matching_file"].format(mouse=mouse))


def list_mice(cfg: dict) -> list:
    """Mice that have a matching table, in numeric order (H01, H02, ...)."""
    d = cfg["data"]
    folder = data_dir("hippocampus") / d["matching_subdir"]
    prefix, suffix = d["matching_file"].split("{mouse}")
    mice = [p.name[len(prefix):len(p.name) - len(suffix)]
            for p in folder.glob(d["matching_file"].format(mouse="*"))]
    mice = [m for m in mice if m[:1] == "H" and m[1:].isdigit()]
    return sorted(mice, key=lambda m: int(m[1:]))


def probes_dir(regime: str, override=None) -> Path:
    if override:
        return Path(override)
    return results_dir("hippocampus", "probes", regime)


def probe_file(regime: str, mouse: str, day: int) -> Path:
    return probes_dir(regime) / f"probe_{mouse}_{day}D.npz"


def list_probe_files(folder: Path, sessions=None) -> list:
    """Per-session probe files in name order; optional filter of (mouse, day) pairs."""
    files = sorted(Path(folder).glob("*_H*_*D.npz"))
    if sessions:
        want = set(sessions)
        files = [f for f in files if session_key(f) in want]
    return files


def session_key(path: Path) -> tuple:
    mouse, day = Path(path).stem.split("_")[-2:]
    return mouse, int(day.rstrip("D"))


def npz_day(d) -> int:
    return int(str(d["day"]).rstrip("D"))


def parse_sessions(text):
    """'H01:1,H02:4' -> [('H01', 1), ('H02', 4)]; None -> None."""
    if not text:
        return None
    out = []
    for item in text.split(","):
        m, d = item.split(":")
        out.append((m.strip(), int(d)))
    return out


def add_session_args(ap):
    ap.add_argument("--probes-dir", default=None,
                    help="folder of per-session probe files (default: results/hippocampus/probes/<regime>)")
    ap.add_argument("--sessions", default=None,
                    help="restrict to sessions, e.g. H01:1,H01:2 (default: all found)")
    ap.add_argument("--out", default=None, help="output CSV (default: under results/hippocampus)")


# ----------------------------------------------------------------- raw session
def load_matching(cfg: dict, mouse: str) -> dict:
    """day -> 0-based calcium rows of the cells matched on all four days.

    The same position in every day's index array is the same cell, so
    calcium[idx[day]] gives the matched population in a consistent order.
    """
    df = pd.read_csv(matching_path(cfg, mouse), header=None)
    df.columns = [1, 2, 3, 4]
    df = df[(df > 0).all(axis=1)].reset_index(drop=True)
    return {day: (df[day].values - 1).astype(int) for day in df.columns}


def _object_centres(sess, x, y):
    """Centre of each object zone (mean tracked position inside it), None if never visited."""
    centres = []
    for i in (1, 2, 3, 4):
        key = f"object{i}"
        if key in sess.files and sess[key].sum() > 0:
            zone = sess[key].astype(bool)
            centres.append((float(np.mean(x[zone])), float(np.mean(y[zone]))))
        else:
            centres.append(None)
    return centres


def build_features(sess_path: Path, sampling_rate: float):
    """X_ego (speed, angular velocity, four object distances) and Y = current (x, y).

    An object whose zone was never entered gets a constant zero distance.
    Returns X (n_frames, 6), Y (n_frames, 2), number of visited objects.
    """
    sess = np.load(sess_path)
    x = sess["x"].astype(np.float64)
    y = sess["y"].astype(np.float64)
    speed = sess["speed"].astype(np.float64)
    hd = sess["headdirection"].astype(np.float64)
    centres = _object_centres(sess, x, y)
    omega = np.gradient(np.unwrap(hd)) * sampling_rate
    dists = [np.zeros_like(x) if c is None else np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)
             for c in centres]
    X = np.column_stack([speed, omega, np.column_stack(dists)])
    Y = np.column_stack([x, y])
    return X, Y, sum(c is not None for c in centres)


def session_variables(sess_path: Path, sampling_rate: float) -> dict:
    """Per-frame variables of the per-component analysis (full session, unmasked).

    Same construction as build_features, plus head direction (sin, cos) and
    time since session start, which are not part of X_ego.
    """
    sess = np.load(sess_path)
    x = sess["x"].astype(np.float64)
    y = sess["y"].astype(np.float64)
    hd = sess["headdirection"].astype(np.float64)
    n = len(x)
    centres = _object_centres(sess, x, y)
    dists = np.zeros((n, 4))
    for i, c in enumerate(centres):
        if c is not None:
            dists[:, i] = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)
    return {
        "speed": sess["speed"].astype(np.float64),
        "omega": np.gradient(np.unwrap(hd)) * sampling_rate,
        "hd_sin": np.sin(hd), "hd_cos": np.cos(hd),
        "d_obj1": dists[:, 0], "d_obj2": dists[:, 1],
        "d_obj3": dists[:, 2], "d_obj4": dists[:, 3],
        "time_since_start": np.arange(n, dtype=np.float64),
    }


def valid_frames(X: np.ndarray, Y: np.ndarray, downsample: int) -> np.ndarray:
    """Frames with finite X and Y, then every downsample-th of them."""
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    idx = np.where(mask)[0]
    return idx[::downsample] if downsample > 1 else idx


def roll_shuffle_calcium(calcium: np.ndarray, seed: int) -> np.ndarray:
    """Circularly shift each neuron's trace by its own random offset.

    Keeps every trace's marginal and autocorrelation, breaks the alignment
    with behaviour and between neurons. calcium: (n_neurons, n_frames).
    """
    rng = np.random.default_rng(seed)
    n_neurons, n_frames = calcium.shape
    out = np.empty_like(calcium)
    for i in range(n_neurons):
        shift = int(rng.integers(low=1, high=n_frames - 1))
        out[i] = np.roll(calcium[i], shift)
    return out


def zscore_split(T: np.ndarray, n_tr: int):
    """Per-neuron z-score with train-fold statistics; T is (n_frames, n_neurons)."""
    T_tr, T_te = T[:n_tr], T[n_tr:]
    mu = T_tr.mean(axis=0, keepdims=True)
    sd = T_tr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((T_tr - mu) / sd).astype(np.float32), ((T_te - mu) / sd).astype(np.float32)


# ----------------------------------------------------------------- estimators
def bin_2d(Y: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    bx = np.clip(np.floor(Y[:, 0] * n_bins), 0, n_bins - 1).astype(int)
    by = np.clip(np.floor(Y[:, 1] * n_bins), 0, n_bins - 1).astype(int)
    return bx * n_bins + by


def h_discrete(bid: np.ndarray, n_classes: int = N_CLASSES) -> float:
    counts = np.bincount(bid, minlength=n_classes)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def confusion_mi_from_predictions(pred: np.ndarray, true: np.ndarray) -> float:
    """Plug-in MI (bits) of the joint table of predicted and true labels."""
    n = len(true)
    joint = Counter(zip(pred.tolist(), true.tolist()))
    p_pred = Counter(pred.tolist())
    p_true = Counter(true.tolist())
    mi = 0.0
    for (yp, yt), c in joint.items():
        pj = c / n
        mi += pj * np.log2(pj / (p_pred[yp] / n * p_true[yt] / n + 1e-30))
    return max(mi, 0.0)


def _decodable(bid_tr, bid_te):
    """Test frames whose bin occurs in the training fold, or None if too few."""
    if len(np.unique(bid_tr)) < 2:
        return None
    keep = np.isin(bid_te, np.unique(bid_tr))
    return keep if keep.sum() >= 20 else None


def confusion_mi(Z_tr, bid_tr, Z_te, bid_te, n_classes: int = N_CLASSES):
    """Relevance estimate I(Z; Y_bin): confusion-matrix MI of a logistic decoder.

    Multinomial logistic regression (C=0.1, lbfgs) on standardized Z of the
    train fold, evaluated on test frames whose bin was seen in training.
    The confusion-matrix MI lower-bounds I(Z; Y_bin) (data processing) and,
    unlike H(Y) - cross-entropy, is not broken by over-confident wrong
    predictions. Returns (mi, H(Y_bin) of the kept test frames).
    """
    keep = _decodable(bid_tr, bid_te)
    if keep is None:
        return float("nan"), float("nan")
    sc = StandardScaler().fit(Z_tr)
    try:
        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs").fit(
            sc.transform(Z_tr), bid_tr)
    except Exception:
        return float("nan"), float("nan")
    pred = clf.predict(sc.transform(Z_te[keep]))
    return confusion_mi_from_predictions(pred, bid_te[keep]), h_discrete(bid_te[keep], n_classes)


def jitter(arr: np.ndarray, rng, scale: float = JITTER) -> np.ndarray:
    """Add Gaussian noise of scale * per-column SD; 1-D input becomes one column."""
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    s = arr.std(axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return arr + rng.standard_normal(arr.shape).astype(arr.dtype) * scale * s


def safe_gcmi(x, y) -> float:
    # Imported here so that the statistics and figure scripts, which never call
    # it, run without loading driada.
    from ib.mi_estimators import mi_gcmi_cc

    try:
        return float(mi_gcmi_cc(x, y))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------- statistics
def per_mouse_slopes(df: pd.DataFrame, col: str, min_days: int = 2) -> pd.Series:
    """OLS slope of col against day, per mouse with at least min_days sessions."""
    out = {}
    for m, g in df.groupby("mouse"):
        g = g[np.isfinite(g[col])]
        if len(g) >= min_days:
            out[m] = stats.linregress(g["day"].values.astype(float), g[col].values).slope
    return pd.Series(out, dtype=float)


def wilcoxon_p(x, alternative="two-sided") -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(stats.wilcoxon(x, alternative=alternative).pvalue)


def table_path(name: str) -> Path:
    """Per-session table shipped under results/hippocampus/."""
    return results_dir("hippocampus") / name


def stats_dir() -> Path:
    return results_dir("hippocampus", "stats")
