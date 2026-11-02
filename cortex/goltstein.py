"""Loading the Goltstein et al. (2021) chronic imaging sessions.

Each ``*TrainingImaging*.mat`` file holds one session of two-photon calcium
imaging during the visual category-learning task (MATLAB v7.3 / HDF5):

  rec.L[level, 0] -> group with dRoR (frames x neurons) and nROI
  rec.X           -> stimulus info: catId, onFrames, uStimId
  rec.stim.S      -> physical grating parameters per category side
  rec.behav       -> FirstLickOutcomeIx (1 correct, 0 incorrect, NaN no response)
  rec.imag        -> SamplingFreq

Two directory layouts are accepted under ``ib.paths.data_dir('cortex')``:

  deposit layout  <AREA>/<MOUSE>/<MOUSE>-spike_dRoR-Loc*-[<AREA>-]<yyyymmdd>-Exp1 (TrainingImaging).mat
                  (the ``data/chronicrecordings`` folder of the public dataset)
  flat copy       <MOUSE>/<AREA>/<same file name>

The area is taken from the directory, not the file name, because the K-mice
file names do not carry it.
"""
import glob
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

# The five mice analysed in the paper; the other five mice of the dataset
# were not analysed.
PAPER_MICE = ("F02", "F03", "F04", "K01", "K02")

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_window(activity_3d, frame_start, frame_end, pre_frames):
    """Mean activity over frames [frame_start, frame_end] relative to onset."""
    idx_start = pre_frames + frame_start
    idx_end = pre_frames + frame_end + 1
    return activity_3d[:, idx_start:idx_end, :].mean(axis=1)


@dataclass
class GoltsteinSession:
    """One session of chronic imaging during category learning."""
    file: str
    mouse_id: str
    area: str
    date: str
    activity: np.ndarray      # (n_trials, n_timepoints, n_neurons), peri-stimulus dRoR
    labels: np.ndarray        # (n_trials,) 0 = category A, 1 = category B
    outcomes: np.ndarray      # (n_trials,) 1 = correct, 0 = incorrect
    accuracy: float           # fraction correct among responded trials
    n_neurons: int
    n_trials: int             # responded trials only
    n_trials_total: int       # all trials including no-response
    sampling_rate: float      # Hz
    orientation: np.ndarray   # (n_trials,) grating orientation, degrees
    spatial_freq: np.ndarray  # (n_trials,) spatial frequency, cycles/degree
    stim_phys: np.ndarray     # (n_trials, 3) (sin 2theta, cos 2theta, log2 SF)
    uStimId: np.ndarray       # (n_trials,) within-session stimulus index 1..K
    K: int                    # number of distinct stimuli among valid trials (12 / 10 / 2)


def _area_from_path(path: Path, mouse_id: str) -> str:
    parent = path.parent
    return parent.parent.name if parent.name == mouse_id else parent.name


def load_session(mat_path, pre_frames=15, post_frames=30, level=0):
    """Load one session file; returns GoltsteinSession or None if unusable."""
    f = h5py.File(mat_path, "r")
    rec = f["rec"]

    sr = float(rec["imag"]["SamplingFreq"][0, 0])

    path = Path(mat_path)
    fname = path.name
    mouse_id = fname.split("-")[0]
    area = _area_from_path(path, mouse_id)
    m = re.search(r"(\d{8})", fname)
    date = m.group(1) if m else ""

    n_levels = int(rec["nLevels"][0, 0])
    if level >= n_levels:
        f.close()
        return None
    L = f[rec["L"][level, 0]]
    dRoR = L["dRoR"]  # (total_frames, n_neurons), read lazily
    n_neurons = int(L["nROI"][0, 0])
    total_frames = dRoR.shape[0]

    stim_on = rec["X"]["onFrames"][:].flatten().astype(int)
    cat_id = rec["X"]["catId"][:].flatten().astype(int)
    uStimId_all = rec["X"]["uStimId"][:].flatten().astype(int)  # 1-indexed
    n_trials_total = len(stim_on)

    # uStimId in [1..n_left] indexes LeftCat (category 1), in
    # [n_left+1..n_left+n_right] indexes RightCat (category 2).
    left_ang = rec["stim"]["S"]["LeftCat"]["Angles"][:].flatten().astype(float)
    left_sf = rec["stim"]["S"]["LeftCat"]["spatialF"][:].flatten().astype(float)
    right_ang = rec["stim"]["S"]["RightCat"]["Angles"][:].flatten().astype(float)
    right_sf = rec["stim"]["S"]["RightCat"]["spatialF"][:].flatten().astype(float)
    n_left = len(left_ang)
    n_right = len(right_ang)

    ori_all = np.empty(n_trials_total, dtype=float)
    sf_all = np.empty(n_trials_total, dtype=float)
    for i, u in enumerate(uStimId_all):
        if 1 <= u <= n_left:
            ori_all[i] = left_ang[u - 1]
            sf_all[i] = left_sf[u - 1]
        elif n_left < u <= n_left + n_right:
            ori_all[i] = right_ang[u - n_left - 1]
            sf_all[i] = right_sf[u - n_left - 1]
        else:
            ori_all[i] = np.nan
            sf_all[i] = np.nan

    first_lick_outcome = rec["behav"]["FirstLickOutcomeIx"][:].flatten()

    responded = ~np.isnan(first_lick_outcome)
    valid_window = (stim_on - pre_frames >= 0) & (stim_on + post_frames <= total_frames)
    valid_phys = ~np.isnan(ori_all) & ~np.isnan(sf_all) & (sf_all > 0)
    valid = responded & valid_window & valid_phys

    if valid.sum() < 10:
        f.close()
        return None

    trial_indices = np.where(valid)[0]
    n_timepoints = pre_frames + post_frames
    activity = np.empty((len(trial_indices), n_timepoints, n_neurons), dtype=np.float32)
    for i, trial_idx in enumerate(trial_indices):
        onset = stim_on[trial_idx]
        activity[i] = dRoR[onset - pre_frames:onset + post_frames, :]

    labels = (cat_id[valid] - 1).astype(int)
    outcomes = first_lick_outcome[valid].astype(int)
    uStimId_valid = uStimId_all[valid]
    orientation = ori_all[valid]
    spatial_freq = sf_all[valid]

    # Orientation enters as (sin 2theta, cos 2theta) because gratings are
    # 180-degree periodic; spatial frequency on a log scale.
    ori_rad = np.deg2rad(orientation)
    stim_phys = np.column_stack([
        np.sin(2 * ori_rad),
        np.cos(2 * ori_rad),
        np.log2(spatial_freq),
    ]).astype(np.float32)

    # Each stimulus must belong to one category; anything else means a data
    # or indexing error, which must not pass silently.
    for u in np.unique(uStimId_valid):
        cats_here = np.unique(labels[uStimId_valid == u])
        if len(cats_here) > 1:
            f.close()
            raise ValueError(
                f"uStimId {u} maps to multiple categories {cats_here.tolist()} "
                f"in {path.name}")

    # K counts stimuli among valid trials, so a short session can show fewer
    # stimuli than its task stage presented.
    K = int(len(np.unique(uStimId_valid)))
    accuracy = int(outcomes.sum()) / len(outcomes)

    f.close()

    return GoltsteinSession(
        file=str(mat_path),
        mouse_id=mouse_id,
        area=area,
        date=date,
        activity=activity,
        labels=labels,
        outcomes=outcomes,
        accuracy=accuracy,
        n_neurons=n_neurons,
        n_trials=len(trial_indices),
        n_trials_total=n_trials_total,
        sampling_rate=sr,
        orientation=orientation,
        spatial_freq=spatial_freq,
        stim_phys=stim_phys,
        uStimId=uStimId_valid.astype(int),
        K=K,
    )


def find_session_files(data_dir, mice=PAPER_MICE):
    """Training-session files of the given mice, in either layout."""
    pattern = str(Path(data_dir) / "*" / "*" / "*TrainingImaging*.mat")
    files = []
    for path in sorted(glob.glob(pattern)):
        mouse = Path(path).name.split("-")[0]
        if mice is None or mouse in mice:
            files.append(path)
    return files


def discover_sessions(data_dir, mice=PAPER_MICE, min_trials=20, min_neurons=50):
    """Load all sessions that pass the trial and neuron filters.

    Returns a list of GoltsteinSession sorted by (mouse, area, date).
    """
    sessions = []
    for path in find_session_files(data_dir, mice):
        try:
            s = load_session(path)
        except Exception as e:
            print(f"  SKIP {Path(path).name}: {e}")
            continue
        if s is None or s.n_trials < min_trials or s.n_neurons < min_neurons:
            continue
        sessions.append(s)
    sessions.sort(key=lambda s: (s.mouse_id, s.area, s.date))
    return sessions


def ae_config_kwargs(cfg, class_weight=None, seed=None):
    """AEConfig keyword arguments from config.yaml, with optional overrides."""
    kw = dict(cfg["ae"])
    if class_weight is not None:
        kw["class_weight"] = class_weight
    if seed is not None:
        kw["seed"] = seed
    return kw
