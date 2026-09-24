"""Off-diagonal observation error covariance (So) operator.

The day-blocked block-Thomas solve must compute the observation-error normal-equation products
    KTinvSoK = K^T So^-1 K,  KTinvSoy = K^T So^-1 (y-F),  ytinvSoy = (y-F)^T So^-1 (y-F)
EXACTLY for So = D (I + P) D, D = diag(sqrt(obs_error)), where P is a two-exponential spatial
correlation within a day plus an adjacent-day (lag-1) temporal correlation. These tests check the
operator against a brute-force dense So^-1, and that the diagonal path (no correlation) reduces to
K^T diag(1/obs_error) K.
"""
import numpy as np

from src.inversion_scripts.so_operator import (
    build_offdiag_so_normal_equations,
    compute_so_normal_equations,
    _great_circle_km,
)

PARAMS = {
    "form": "two_exponential",
    "corr_amplitude1": 0.385, "corr_length1_km": 26.0,
    "corr_amplitude2": 0.459, "corr_length2_km": 398.0,
    "corr_cutoff_km": 3 * 398.0,
}


def _synthetic(seed=0, n_per_day=8, n_days=3, n_elem=4):
    rng = np.random.RandomState(seed)
    n = n_per_day * n_days
    K = rng.randn(n, n_elem)
    dy = rng.randn(n)
    obs_error = rng.uniform(50.0, 400.0, n)            # variance per observation
    lat = rng.uniform(-5.0, 5.0, n)                    # small domain: some pairs in, some beyond cutoff
    lon = rng.uniform(-5.0, 5.0, n)
    base = np.datetime64("2020-01-01")
    dates = np.concatenate(
        [np.full(n_per_day, base + np.timedelta64(d, "D")) for d in range(n_days)]
    )
    return K, dy, obs_error, lat, lon, dates


def _dense_products(K, dy, obs_error, lat, lon, dates, temporal_rho):
    """Brute-force K^T So^-1 K, K^T So^-1 dy, dy^T So^-1 dy with a dense So = D (I+P) D."""
    A1, L1 = PARAMS["corr_amplitude1"], PARAMS["corr_length1_km"]
    A2, L2 = PARAMS["corr_amplitude2"], PARAMS["corr_length2_km"]
    cut = PARAMS["corr_cutoff_km"]
    n = len(dy)
    day = np.asarray(dates).astype("datetime64[D]")
    day = (day - day.min()).astype(int)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dd = _great_circle_km(lat[i], lon[i], lat[j], lon[j])
            if dd > cut:
                continue
            corr = A1 * np.exp(-dd / L1) + A2 * np.exp(-dd / L2)
            if day[i] == day[j]:
                M[i, j] = corr
            elif abs(day[i] - day[j]) == 1:
                M[i, j] = temporal_rho * corr
    D = np.sqrt(obs_error)
    So = (D[:, None] * M) * D[None, :]
    So_inv = np.linalg.inv(So)
    return K.T @ So_inv @ K, K.T @ So_inv @ dy, float(dy @ So_inv @ dy)


def test_offdiag_matches_dense_spatial_only():
    K, dy, oe, lat, lon, dates = _synthetic(seed=0)
    a, b, c = build_offdiag_so_normal_equations(K, dy, oe, lat, lon, dates, PARAMS, temporal_rho=0.0)
    A, B, C = _dense_products(K, dy, oe, lat, lon, dates, 0.0)
    assert np.allclose(a, A, rtol=1e-6, atol=1e-6)
    assert np.allclose(b, B, rtol=1e-6, atol=1e-6)
    assert np.isclose(c, C, rtol=1e-6)


def test_offdiag_matches_dense_with_temporal():
    K, dy, oe, lat, lon, dates = _synthetic(seed=3)
    a, b, c = build_offdiag_so_normal_equations(K, dy, oe, lat, lon, dates, PARAMS, temporal_rho=0.17)
    A, B, C = _dense_products(K, dy, oe, lat, lon, dates, 0.17)
    assert np.allclose(a, A, rtol=1e-6, atol=1e-6)
    assert np.allclose(b, B, rtol=1e-6, atol=1e-6)
    assert np.isclose(c, C, rtol=1e-6)


def test_diagonal_path_reduces_to_inverse_variance():
    K, dy, oe, lat, lon, dates = _synthetic(seed=5)
    a, b, c = compute_so_normal_equations(K, dy, oe, lat, lon, dates, None)  # None -> diagonal So
    KTinvSo = K.T / oe
    assert np.allclose(a, KTinvSo @ K)
    assert np.allclose(b, KTinvSo @ dy)
    assert np.isclose(c, float(dy @ (dy / oe)))


def test_compute_dispatches_to_offdiag():
    K, dy, oe, lat, lon, dates = _synthetic(seed=1)
    a, b, c = compute_so_normal_equations(K, dy, oe, lat, lon, dates, PARAMS)
    A, B, C = build_offdiag_so_normal_equations(K, dy, oe, lat, lon, dates, PARAMS, temporal_rho=0.0)
    assert np.allclose(a, A) and np.allclose(b, B) and np.isclose(c, C)


def test_unsupported_form_raises():
    import pytest
    K, dy, oe, lat, lon, dates = _synthetic(seed=2)
    with pytest.raises(ValueError):
        compute_so_normal_equations(K, dy, oe, lat, lon, dates, {"form": "gaussian"})
