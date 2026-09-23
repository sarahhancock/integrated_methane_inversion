"""Cross-module great-circle distance consistency.

Sa correlation lengths (length_scale / sector_ensemble) and So correlation lengths (REM off-diagonal
in build_obs_error_covariance and the exact solve in invert) are only comparable if every module
measures distance the same way. This guards against a stray Earth radius or formula change in one
place silently rescaling one covariance relative to another.
"""
import numpy as np
import pytest

from invert import _great_circle_km
from build_obs_error_covariance import haversine_km
from build_sector_ensemble_prior_covariance import great_circle_matrix
from build_length_scale_prior_covariance import haversine_distance_km


def test_all_great_circle_implementations_agree():
    lat = np.array([-10.0, -5.0, 3.0, -20.0, 12.5])
    lon = np.array([-60.0, -55.0, -70.0, -65.0, -47.0])
    n = len(lat)
    gc = np.array([[_great_circle_km(lat[i], lon[i], lat[j], lon[j]) for j in range(n)] for i in range(n)])
    hv = np.array([[haversine_km(lat[i], lon[i], lat[j], lon[j]) for j in range(n)] for i in range(n)])
    gm = great_circle_matrix(lat, lon)
    ls = haversine_distance_km(lat, lon)
    for other in (hv, gm, ls):
        assert np.allclose(gc, other, atol=1e-6)


def test_distance_is_symmetric_with_zero_diagonal():
    lat = np.array([-10.0, 0.0, 5.0])
    lon = np.array([-60.0, -50.0, -40.0])
    d = great_circle_matrix(lat, lon)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_known_distance_one_degree_latitude():
    # 1 degree of latitude ~ 111.19 km on a 6371 km sphere.
    d = _great_circle_km(0.0, 0.0, 1.0, 0.0)
    assert np.isclose(d, np.radians(1.0) * 6371.0, rtol=1e-6)
