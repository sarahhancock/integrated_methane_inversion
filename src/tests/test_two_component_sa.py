"""Two-component prior error covariance Sa.

Within a (country, sector) group the cell-cell correlation is 1/(ratio_i*ratio_j), with
ratio_i = sqrt(1 + (R01^2 - 1)/n_eff_i). With n_eff = 1 (single native cell per element) the
correlation is 1/R01^2, and at R01 = 1 the block is fully correlated (rho = 1) -- the national
rank-1 limit that makes the R01=1 prior rank-deficient. Also checks the Saunois global-background
natural correlation terms exist and take effect."""
import numpy as np
from build_national_inventory_prior_covariance import (
    two_component_absolute,
    SAUNOIS_GLOBAL_BACKGROUND,
)


def _two_cell_correlation(R01):
    """Correlation between two equal-emission, single-native-cell elements of one country/sector."""
    rows = {("1", "Oil", 0): 1.0, ("1", "Oil", 1): 1.0}
    urows = [{"country_id": "1", "sector": "Oil", "relative_uncertainty": 0.6}]
    # empty neff dicts -> n_eff = 1 for every element -> ratio_i = R01
    Sa, _ = two_component_absolute(
        rows, {}, {}, urows, 2, grid_national_ratio=R01, min_uncertainty=0.30
    )
    return Sa[0, 1] / np.sqrt(Sa[0, 0] * Sa[1, 1])


def test_correlation_matches_formula():
    assert abs(_two_cell_correlation(2.5) - 1.0 / 2.5 ** 2) < 1e-9
    assert abs(_two_cell_correlation(2.0) - 1.0 / 2.0 ** 2) < 1e-9


def test_fully_correlated_at_ratio_one():
    # R01 = 1 -> ratio_i = 1 -> rho = 1 (fully-correlated rank-1 block; explains R01=1 degeneracy)
    assert abs(_two_cell_correlation(1.0) - 1.0) < 1e-9


def test_correlation_decreases_with_ratio():
    # larger grid:national ratio -> more independent local error -> weaker cell-cell correlation
    assert _two_cell_correlation(1.0) > _two_cell_correlation(2.0) > _two_cell_correlation(2.5)


def test_saunois_background_defined():
    assert isinstance(SAUNOIS_GLOBAL_BACKGROUND, dict) and len(SAUNOIS_GLOBAL_BACKGROUND) > 0


def test_saunois_background_is_a_floor():
    # The Saunois global background is a FLOOR: g only ADDS where u < g (aggregate -> max(u, g)),
    # and adds NOTHING where u >= g (the BTR value already covers it -- no double-counting).
    rows = {("1", "Oil", 0): 1.0, ("1", "Oil", 1): 1.0}
    urows = [{"country_id": "1", "sector": "Oil", "relative_uncertainty": 0.6}]
    Sa_plain, _ = two_component_absolute(rows, {}, {}, urows, 2, grid_national_ratio=2.5)
    # g > u: the floor engages and changes the covariance
    Sa_hi, _ = two_component_absolute(
        rows, {}, {}, urows, 2, grid_national_ratio=2.5, global_background={"Oil": 0.9}
    )
    assert not np.allclose(Sa_plain, Sa_hi)
    # g <= u: nothing is added (no double-count); Sa is identical to the plain two-component
    Sa_lo, _ = two_component_absolute(
        rows, {}, {}, urows, 2, grid_national_ratio=2.5, global_background={"Oil": 0.3}
    )
    assert np.allclose(Sa_plain, Sa_lo)
