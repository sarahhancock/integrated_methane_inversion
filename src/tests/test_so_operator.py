"""The observation-error So is a single, solver-independent operator (compute_so_normal_equations).

Normal, softplus, and lognormal all assemble their K^T So^-1 K / K^T So^-1 dy / dy^T So^-1 dy through
this one function, so they see identical observation weighting. The diagonal path must reproduce the
plain K^T diag(1/so) K normal equations exactly; the two-exponential off-diagonal path requires
per-observation lat/lon/dates (the exact day-blocked solve)."""
import numpy as np
import pytest
from invert import compute_so_normal_equations


def test_diagonal_so_matches_manual():
    rng = np.random.RandomState(1)
    m, n = 150, 20
    K = rng.randn(m, n)
    dy = rng.randn(m)
    so = np.abs(rng.rand(m)) + 0.5          # positive per-observation diagonal So
    KTinvSoK, KTinvSoy, yty = compute_so_normal_equations(K, dy, so, None, None, None, None)
    inv = 1.0 / so
    assert np.allclose(KTinvSoK, K.T @ (K * inv[:, None]))     # K^T diag(1/so) K
    assert np.allclose(KTinvSoy, K.T @ (dy * inv))             # K^T diag(1/so) dy
    assert np.isclose(yty, float(dy @ (dy * inv)))             # dy^T diag(1/so) dy


def test_no_corr_params_is_plain_diagonal():
    # so_corr_params=None must give exactly the diagonal result (no off-diagonal correction).
    rng = np.random.RandomState(2)
    K = rng.randn(50, 8)
    dy = rng.randn(50)
    so = np.abs(rng.rand(50)) + 1.0
    a = compute_so_normal_equations(K, dy, so, None, None, None, None)
    b = compute_so_normal_equations(K, dy, so, None, None, None, {})   # empty dict -> no form -> diagonal
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and np.isclose(a[2], b[2])


def test_two_exponential_requires_metadata():
    # the exact day-blocked two-exponential solve needs lat/lon/dates; without them it must raise.
    K = np.ones((4, 2)); dy = np.ones(4); so = np.ones(4)
    params = {
        "form": "two_exponential", "corr_amplitude1": 0.385, "corr_length1_km": 26.0,
        "corr_amplitude2": 0.459, "corr_length2_km": 398.0, "corr_cutoff_km": 1500.0, "temporal_rho": 0.19,
    }
    with pytest.raises(ValueError):
        compute_so_normal_equations(K, dy, so, None, None, None, params)
