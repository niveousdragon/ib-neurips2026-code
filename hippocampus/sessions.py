"""Session sets used by the hippocampus statistics.

16 mice x 4 days = 64 recorded sessions. The paper analyses 61 of them.

Excluded sessions (Supplementary "Session exclusion"): three sessions showed
near-immobility throughout the recording (severe freezing), identified by a
collapse of the egocentric feature vector to effective dimensionality <= 3;
their distance travelled is in the lowest 5 % of the cohort (11.84, 13.85 and
17.29 m). Effective dimensionality here is the number of non-constant
components of X_ego over the session (see 10_session_summary.py); an object
distance is constant when the animal never entered that object's zone.
All 16 mice keep at least three days.
"""
import pandas as pd

DROP3 = frozenset({("H32", 4), ("H06", 4), ("H33", 3)})

# Sensitivity cut only (Supplementary "Aggregate session-inclusion sensitivity"):
# DROP3 plus the next two sessions by distance travelled.
DROP5 = DROP3 | {("H06", 3), ("H39", 4)}

COHORTS = {"all64": frozenset(), "drop3": DROP3, "drop5": DROP5}


def cohort(df: pd.DataFrame, name: str = "drop3") -> pd.DataFrame:
    """Rows of a per-session table (columns mouse, day) kept by a cohort."""
    excl = COHORTS[name]
    keep = [(str(m), int(d)) not in excl for m, d in zip(df["mouse"], df["day"])]
    return df[keep].copy()


def is_kept(mouse: str, day: int, name: str = "drop3") -> bool:
    return (str(mouse), int(day)) not in COHORTS[name]
