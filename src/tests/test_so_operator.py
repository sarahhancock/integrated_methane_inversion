"""The observation-error So is a single, solver-independent operator (compute_so_normal_equations).

Normal, softplus, and lognormal all assemble their K^T So^-1 K / K^T So^-1 dy / dy^T So^-1 dy through
this one function, so they see identical observation weighting. The diagonal path must reproduce the
plain K^T diag(1/so) K normal equations exactly; the two-exponential off-diagonal path requires
per-observation lat/lon/dates (the exact day-blocked solve)."""
import numpy as np
import pytest
from invert import compute_so_normal_equations, _great_circle_km


# Shared two-exponential parameters (South America fit).
_SA_PARAMS = {
    "form": "two_exponential",
    "corr_amplitude1": 0.385, "corr_length1_km": 26.0,
    "corr_amplitude2": 0.459, "corr_length2_km": 398.0,
    "corr_cutoff_km": 1500.0, "temporal_rho": 0.0,
}


def _dense_so(lat, lon, day_index, obs_error, params, temporal_rho):
    """Assemble the FULL So = D (I + P) D exactly as build_offdiag_so_normal_equations models it:
    within-day two-exponential correlation, adjacent-day (lag-1) coupling temporal_rho * P, and zero
    beyond a one-day lag.  Distances use the same great-circle metric as the operator."""
    A1, L1 = params["corr_amplitude1"], params["corr_length1_km"]
    A2, L2 = params["corr_amplitude2"], params["corr_length2_km"]
    cut = params["corr_cutoff_km"]
    n = len(lat)
    R = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = _great_circle_km(lat[i], lon[i], lat[j], lon[j])
            if d > cut:
                continue
            s = A1 * np.exp(-d / L1) + A2 * np.exp(-d / L2)
            lag = abs(int(day_index[i]) - int(day_index[j]))
            if lag == 0:
                R[i, j] = s
            elif lag == 1:
                R[i, j] = temporal_rho * s
            # lag >= 2 -> 0
    D = np.sqrt(obs_error)
    return D[:, None] * R * D[None, :]


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
    params = dict(_SA_PARAMS, temporal_rho=0.19)
    with pytest.raises(ValueError):
        compute_so_normal_equations(K, dy, so, None, None, None, params)


def _obs_cluster(seed, n, day_index):
    """A small clustered set of obs (all within cutoff) on the given days, plus K/dy/so."""
    rng = np.random.RandomState(seed)
    lat = -10.0 + rng.uniform(-2.0, 2.0, n)      # ~ +/- 220 km spread -> moderate correlations, PD So
    lon = -60.0 + rng.uniform(-2.0, 2.0, n)
    K = rng.randn(n, 5)
    dy = rng.randn(n)
    so = np.abs(rng.rand(n)) + 0.5
    dates = np.array(
        [np.datetime64("2019-01-01") + np.timedelta64(int(d), "D") for d in day_index]
    )
    return lat, lon, dates, np.asarray(day_index), K, dy, so


def test_two_exponential_matches_dense_spatial_only():
    # Exact day-blocked solve (single day, no temporal coupling) must equal the dense So^-1 result.
    n = 14
    day_index = np.zeros(n, dtype=int)
    lat, lon, dates, di, K, dy, so = _obs_cluster(11, n, day_index)
    params = dict(_SA_PARAMS, temporal_rho=0.0)
    So = _dense_so(lat, lon, di, so, params, temporal_rho=0.0)
    Sinv = np.linalg.inv(So)
    KTinvSoK, KTinvSoy, ytinvSoy = compute_so_normal_equations(K, dy, so, lat, lon, dates, params)
    assert np.allclose(KTinvSoK, K.T @ Sinv @ K, atol=1e-8)
    assert np.allclose(KTinvSoy, K.T @ Sinv @ dy, atol=1e-8)
    assert np.isclose(ytinvSoy, float(dy @ Sinv @ dy), atol=1e-8)


def test_two_exponential_matches_dense_with_temporal():
    # Two consecutive days + adjacent-day (lag-1) temporal coupling must equal the dense block-
    # tridiagonal So^-1 result (validates the Thomas forward/back substitution).
    n = 16
    day_index = np.array([0] * 8 + [1] * 8)
    lat, lon, dates, di, K, dy, so = _obs_cluster(23, n, day_index)
    params = dict(_SA_PARAMS, temporal_rho=0.19)
    So = _dense_so(lat, lon, di, so, params, temporal_rho=0.19)
    Sinv = np.linalg.inv(So)
    KTinvSoK, KTinvSoy, ytinvSoy = compute_so_normal_equations(K, dy, so, lat, lon, dates, params)
    assert np.allclose(KTinvSoK, K.T @ Sinv @ K, atol=1e-8)
    assert np.allclose(KTinvSoy, K.T @ Sinv @ dy, atol=1e-8)
    assert np.isclose(ytinvSoy, float(dy @ Sinv @ dy), atol=1e-8)


def test_two_exponential_nonadjacent_days_decouple():
    # Days 2 apart have zero So coupling: the exact solve must equal two independent single-day solves.
    n = 12
    day_index = np.array([0] * 6 + [2] * 6)     # gap of 2 -> no temporal coupling
    lat, lon, dates, di, K, dy, so = _obs_cluster(31, n, day_index)
    params = dict(_SA_PARAMS, temporal_rho=0.19)
    So = _dense_so(lat, lon, di, so, params, temporal_rho=0.19)   # lag>=2 -> block-diagonal
    Sinv = np.linalg.inv(So)
    KTinvSoK, _, _ = compute_so_normal_equations(K, dy, so, lat, lon, dates, params)
    assert np.allclose(KTinvSoK, K.T @ Sinv @ K, atol=1e-8)
