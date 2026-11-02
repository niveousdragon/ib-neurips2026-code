"""Thin wrappers over DRIADA MI estimators with a uniform interface.

Both wrappers accept x, y as (n_samples, n_features) arrays (or 1D for a
single feature) and return scalar MI in bits.
"""
import numpy as np
from driada.information.gcmi import cmi_ggg, copnorm, gcmi_cc
from driada.information.ksg import nonparam_mi_cc


def _ensure_2d_samples_features(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    return a


def mi_gcmi_cc(x, y):
    """GCMI (Gaussian copula MI) between continuous x and continuous y."""
    x = _ensure_2d_samples_features(x)
    y = _ensure_2d_samples_features(y)
    # DRIADA gcmi_cc expects (n_features, n_samples)
    return float(gcmi_cc(x.T, y.T))


def mi_ksg_cc(x, y, k: int = 5):
    """KSG nonparametric continuous MI. Returns bits."""
    x = _ensure_2d_samples_features(x)
    y = _ensure_2d_samples_features(y)
    # DRIADA nonparam_mi_cc expects (n_samples, n_features)
    return float(nonparam_mi_cc(x, y, k=k, base=2.0))


def cmi_gcmi_ccc(x, y, z):
    """Conditional GCMI: I(X;Y|Z) for three continuous variables.

    Inputs are (n_samples, n_features) or 1D. Returns bits.
    """
    x = _ensure_2d_samples_features(x)
    y = _ensure_2d_samples_features(y)
    z = _ensure_2d_samples_features(z)
    # copnorm + cmi_ggg both expect (n_features, n_samples)
    cx = copnorm(x.T)
    cy = copnorm(y.T)
    cz = copnorm(z.T)
    return float(max(0.0, cmi_ggg(cx, cy, cz)))
