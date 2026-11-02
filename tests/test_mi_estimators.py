"""Sanity: MI estimators recover known Gaussian MI on synthetic data.

For jointly Gaussian X, Y with correlation rho:
    I(X;Y) = -0.5 * log2(1 - rho^2)  bits
"""
import numpy as np

from ib.mi_estimators import cmi_gcmi_ccc, mi_gcmi_cc, mi_ksg_cc


def _gaussian_pair(n, rho, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    xy = z @ L.T
    return xy[:, 0:1], xy[:, 1:2]


def test_gcmi_gaussian_rho_0_5():
    x, y = _gaussian_pair(n=5000, rho=0.5)
    truth = -0.5 * np.log2(1 - 0.5 ** 2)
    est = mi_gcmi_cc(x, y)
    assert abs(est - truth) < 0.1, f"gcmi={est:.3f}, truth={truth:.3f}"


def test_ksg_gaussian_rho_0_5():
    x, y = _gaussian_pair(n=5000, rho=0.5)
    truth = -0.5 * np.log2(1 - 0.5 ** 2)
    est = mi_ksg_cc(x, y, k=5)
    assert abs(est - truth) < 0.1, f"ksg={est:.3f}, truth={truth:.3f}"


def test_independent_gaussians_give_near_zero_mi():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 1))
    y = rng.standard_normal((5000, 1))
    assert abs(mi_gcmi_cc(x, y)) < 0.05
    assert abs(mi_ksg_cc(x, y, k=5)) < 0.05


# ----- Conditional MI tests -----------------------------------------------
# Sanity for cmi_gcmi_ccc against analytically-known Gaussian CMI values.

def test_conditional_independence_gives_zero_cmi():
    """Common cause Z creates I(X;Y) > 0, but I(X;Y|Z) = 0 exactly.

    Setup: Z ~ N(0,1), X = a*Z + nx, Y = b*Z + ny, with nx, ny independent.
    Marginally X and Y correlate via Z. Conditionally on Z they are independent.
    """
    rng = np.random.default_rng(0)
    n = 5000
    z = rng.standard_normal((n, 1))
    a, b = 0.8, 0.9
    x = a * z + rng.standard_normal((n, 1))
    y = b * z + rng.standard_normal((n, 1))
    # Sanity: marginal MI should be > 0
    assert mi_gcmi_cc(x, y) > 0.1, "common-cause sanity"
    # Conditional MI given Z should be near zero
    cmi = cmi_gcmi_ccc(x, y, z)
    assert cmi < 0.05, f"expected ~0, got {cmi:.3f}"


def test_partial_correlation_recovers_analytic_cmi():
    """X, Z independent. Y = alpha*X + beta*Z + noise.

    Analytic: I(X;Y|Z) = 0.5 * log2(1 + alpha^2 / sigma^2) bits.
    """
    rng = np.random.default_rng(0)
    n = 5000
    alpha, beta, sigma = 1.0, 1.0, 1.0
    x = rng.standard_normal((n, 1))
    z = rng.standard_normal((n, 1))
    y = alpha * x + beta * z + sigma * rng.standard_normal((n, 1))
    truth = 0.5 * np.log2(1 + (alpha ** 2) / (sigma ** 2))  # = 0.5 bits
    est = cmi_gcmi_ccc(x, y, z)
    assert abs(est - truth) < 0.1, f"cmi={est:.3f}, truth={truth:.3f}"


def test_cmi_zero_on_jointly_independent():
    """All three independent => I(X;Y|Z) ~ 0."""
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.standard_normal((n, 1))
    y = rng.standard_normal((n, 1))
    z = rng.standard_normal((n, 1))
    cmi = cmi_gcmi_ccc(x, y, z)
    assert cmi < 0.05, f"expected ~0, got {cmi:.3f}"
