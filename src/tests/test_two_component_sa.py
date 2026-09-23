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


def _aggregate_rel(Sa, positions, emis):
    """Emission-weighted national relative uncertainty over a country's cells:
    sqrt(1^T Sa_block 1) / sum(emis) -- the relative error of the aggregated national emission."""
    block = Sa[np.ix_(positions, positions)]
    return float(np.sqrt(block.sum()) / emis.sum())


def test_national_aggregate_equals_u_btr():
    # THE core two-component claim: for one country/sector the emission-weighted national aggregate
    # equals u_BTR EXACTLY (snat = u/sqrt(1+Q) makes national + local sum to u^2 E^2), for arbitrary
    # per-cell emissions.
    emis = np.array([3.0, 1.0, 0.5, 2.0])
    rows = {("1", "Oil", i): float(e) for i, e in enumerate(emis)}
    urows = [{"country_id": "1", "sector": "Oil", "relative_uncertainty": 0.6}]
    Sa, _ = two_component_absolute(rows, {}, {}, urows, emis.size, grid_national_ratio=2.5)
    assert np.isclose(_aggregate_rel(Sa, np.arange(emis.size), emis), 0.6, atol=1e-12)


def test_national_aggregate_independent_of_grid_ratio():
    # The national aggregate is u_BTR regardless of R01 (R01 only redistributes national vs local).
    emis = np.array([2.0, 1.0, 4.0])
    rows = {("1", "Gas", i): float(e) for i, e in enumerate(emis)}
    urows = [{"country_id": "1", "sector": "Gas", "relative_uncertainty": 0.45}]
    for R01 in (1.0, 2.0, 3.5):
        Sa, _ = two_component_absolute(rows, {}, {}, urows, emis.size, grid_national_ratio=R01)
        assert np.isclose(_aggregate_rel(Sa, np.arange(emis.size), emis), 0.45, atol=1e-12)


def test_multicountry_aggregates_independent():
    # Two countries with different u_BTR: each national aggregate matches its own u, and different
    # countries are uncorrelated (block-diagonal across countries).
    rows = {("A", "Oil", 0): 2.0, ("A", "Oil", 1): 1.0, ("B", "Oil", 2): 3.0, ("B", "Oil", 3): 1.0}
    urows = [
        {"country_id": "A", "sector": "Oil", "relative_uncertainty": 0.5},
        {"country_id": "B", "sector": "Oil", "relative_uncertainty": 0.8},
    ]
    Sa, _ = two_component_absolute(rows, {}, {}, urows, 4, grid_national_ratio=2.5)
    assert np.isclose(_aggregate_rel(Sa, np.array([0, 1]), np.array([2.0, 1.0])), 0.5, atol=1e-12)
    assert np.isclose(_aggregate_rel(Sa, np.array([2, 3]), np.array([3.0, 1.0])), 0.8, atol=1e-12)
    assert np.allclose(Sa[np.ix_([0, 1], [2, 3])], 0.0)   # cross-country covariance is zero


def test_saunois_floor_sets_aggregate_to_max_u_g():
    emis = np.array([2.0, 1.0])
    rows = {("1", "Wetlands", 0): 2.0, ("1", "Wetlands", 1): 1.0}
    urows = [{"country_id": "1", "sector": "Wetlands", "relative_uncertainty": 0.35}]
    pos = np.array([0, 1])
    # g > u -> aggregate rises to g
    Sa_hi, _ = two_component_absolute(
        rows, {}, {}, urows, 2, grid_national_ratio=2.5, global_background={"Wetlands": 0.6}
    )
    assert np.isclose(_aggregate_rel(Sa_hi, pos, emis), 0.6, atol=1e-12)
    # g <= u -> aggregate stays u (no double-count)
    Sa_lo, _ = two_component_absolute(
        rows, {}, {}, urows, 2, grid_national_ratio=2.5, global_background={"Wetlands": 0.2}
    )
    assert np.isclose(_aggregate_rel(Sa_lo, pos, emis), 0.35, atol=1e-12)


def test_domain_invariance_scales_national_offdiagonal():
    # The off-diagonal is purely the national rank-1 term, so with in-domain fraction f_C it scales by
    # 1/f_C^2 (the domain-invariant national constraint); the local diagonal excess is NOT f-scaled.
    rows = {("1", "Oil", 0): 2.0, ("1", "Oil", 1): 1.0}
    urows = [{"country_id": "1", "sector": "Oil", "relative_uncertainty": 0.5}]
    Sa_full, _ = two_component_absolute(rows, {}, {}, urows, 2, grid_national_ratio=2.5, country_fraction={"1": 1.0})
    Sa_half, _ = two_component_absolute(rows, {}, {}, urows, 2, grid_national_ratio=2.5, country_fraction={"1": 0.5})
    assert np.isclose(Sa_half[0, 1] / Sa_full[0, 1], 1.0 / 0.5 ** 2, atol=1e-9)
