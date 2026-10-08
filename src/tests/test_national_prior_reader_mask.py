"""National-inventory prior: per-(country, sector) uncertainty CSV reader + per-country mask.

Covers the "any IMI user" robustness of read_uncertainty_table (flexible column names, header
casing/whitespace, blank/non-numeric cells) and the equal-area correctness of
get_country_fraction_mask's domain-invariant fraction f_C (areas must be computed in an equal-area
CRS, not the shapefile's geographic CRS where square-degrees are latitude-biased by ~1/cos(lat)).
"""
import csv
import os
import tempfile
import warnings

import numpy as np
import pytest

import build_national_inventory_prior_covariance as B


def _write_csv(header, rows):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return path


# --------------------------- read_uncertainty_table ---------------------------

def test_domain_invariant_defaults_on_for_regional():
    # The domain-invariant national term (1/f_C^2 for partially-in-domain countries) defaults ON for
    # regional inversions and OFF for global; an explicit key overrides.
    assert B.domain_invariant_enabled({"isRegional": True}) is True
    assert B.domain_invariant_enabled({"isRegional": False}) is False
    assert B.domain_invariant_enabled({}) is True                      # isRegional defaults True
    assert B.domain_invariant_enabled({"isRegional": True, "NationalPriorDomainInvariant": False}) is False
    assert B.domain_invariant_enabled({"isRegional": False, "NationalPriorDomainInvariant": True}) is True


def test_reader_flexible_columns_and_iso3_identifier():
    path = _write_csv(["iso3", "sector", "u"], [["BRA", "Livestock", "0.5"]])
    try:
        rows = B.read_uncertainty_table(path)
    finally:
        os.remove(path)
    assert rows == [
        {
            "country_id": "BRA",
            "country_name": "BRA",
            "sector": "Livestock",
            "relative_uncertainty": 0.5,
        }
    ]


def test_reader_tolerates_header_casing_and_whitespace():
    # Header with odd casing + surrounding spaces must still be recognized.
    path = _write_csv(["  ISO3 ", " Sector ", "  U  "], [["COL", "Gas", " 0.3 "]])
    try:
        rows = B.read_uncertainty_table(path)
    finally:
        os.remove(path)
    assert len(rows) == 1
    assert rows[0]["country_id"] == "COL"
    assert rows[0]["relative_uncertainty"] == 0.3  # padded numeric parsed


def test_reader_skips_blank_and_nonnumeric_uncertainty():
    path = _write_csv(
        ["iso3", "sector", "u"],
        [
            ["BRA", "Livestock", "0.5"],
            ["BRA", "Coal", "   "],   # whitespace-only -> skipped, not a crash
            ["BRA", "Oil", "n/a"],    # non-numeric -> skipped
        ],
    )
    try:
        rows = B.read_uncertainty_table(path)
    finally:
        os.remove(path)
    assert [r["sector"] for r in rows] == ["Livestock"]


def test_reader_uses_explicit_country_name_column():
    path = _write_csv(
        ["iso3", "country", "sector", "relative_uncertainty"],
        [["BRA", "Brazil", "Livestock", "0.4"]],
    )
    try:
        rows = B.read_uncertainty_table(path)
    finally:
        os.remove(path)
    assert rows[0]["country_id"] == "BRA"
    assert rows[0]["country_name"] == "Brazil"


def test_reader_missing_sector_column_raises():
    path = _write_csv(["iso3", "u"], [["BRA", "0.5"]])
    try:
        with pytest.raises(ValueError):
            B.read_uncertainty_table(path)
    finally:
        os.remove(path)


def test_reader_no_usable_rows_raises():
    path = _write_csv(["iso3", "sector", "u"], [["BRA", "Coal", ""]])
    try:
        with pytest.raises(ValueError):
            B.read_uncertainty_table(path)
    finally:
        os.remove(path)


# --------------------------- get_country_fraction_mask ---------------------------

def _load_shapes_or_skip():
    pytest.importorskip("geopandas")
    pytest.importorskip("regionmask")
    shp = B.default_country_shapefile()
    if not shp or not os.path.exists(shp):
        pytest.skip("bundled country shapefile not available")
    return B.load_country_shapes(shp, "ISO3"), shp


def test_default_shapefile_has_name_and_iso3():
    _, shp = _load_shapes_or_skip()
    import geopandas as gpd

    gdf = gpd.read_file(shp)
    assert "NAME" in gdf.columns and "ISO3" in gdf.columns
    assert len(gdf) > 100


def test_binary_mask_lands_in_country_and_returns_unit_fc():
    shapes, _ = _load_shapes_or_skip()
    lon = np.arange(-80.0, -66.0, 0.5)
    lat = np.arange(-5.0, 13.0, 0.5)  # window over Colombia
    mask, f_c = B.get_country_fraction_mask("COL", lat, lon, shapes, "ISO3", area_weighting=False)
    assert f_c == 1.0                       # binary path: fraction unknown -> no scaling
    ii, jj = np.where(mask > 0)
    assert ii.size > 0
    # nonzero cells should cluster around Colombia (roughly lon -74, lat 4)
    assert abs(float(np.mean(lon[jj])) - (-73.0)) < 6.0
    assert abs(float(np.mean(lat[ii])) - 4.0) < 6.0


def test_area_weighted_fractions_are_valid_and_warning_free():
    shapes, _ = _load_shapes_or_skip()
    lon = np.arange(-80.0, -66.0, 1.0)
    lat = np.arange(-5.0, 13.0, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask, f_c = B.get_country_fraction_mask("COL", lat, lon, shapes, "ISO3", area_weighting=True)
    # The equal-area fix must eliminate geopandas' "Geometry is in a geographic CRS" area warning.
    assert not [w for w in caught if "geographic CRS" in str(w.message)]
    assert mask.min() >= 0.0 and mask.max() <= 1.0 + 1e-9
    assert 0.0 < f_c <= 1.0


def test_area_weighted_fc_is_one_when_country_fully_covered():
    shapes, _ = _load_shapes_or_skip()
    # Grid comfortably enclosing all of Colombia -> in-domain fraction of the whole country ~ 1.
    lon = np.arange(-80.0, -65.0, 0.5)
    lat = np.arange(-5.0, 14.0, 0.5)
    _, f_c = B.get_country_fraction_mask("COL", lat, lon, shapes, "ISO3", area_weighting=True)
    assert f_c > 0.98


def test_unknown_country_raises():
    shapes, _ = _load_shapes_or_skip()
    lon = np.arange(-80.0, -66.0, 1.0)
    lat = np.arange(-5.0, 13.0, 1.0)
    with pytest.raises(ValueError):
        B.get_country_fraction_mask("ZZZ", lat, lon, shapes, "ISO3", area_weighting=False)


def test_partial_country_warns_when_domain_invariant_off(capsys):
    # A regional domain covering only part of a country (Nigeria-style) must warn that the national
    # uncertainty is applied as if fully in-domain when NationalPriorDomainInvariant is off.
    _load_shapes_or_skip()
    xr = pytest.importorskip("xarray")
    lon = np.arange(-76.0, -72.0, 0.5)   # a small window inside Colombia -> Colombia extends beyond
    lat = np.arange(2.0, 6.0, 0.5)
    prior = xr.Dataset(coords={"lat": lat, "lon": lon})
    rows = [{"country_id": "COL", "country_name": "COL", "sector": "Oil", "relative_uncertainty": 0.3}]
    config = {"NationalPriorCountryNameColumn": "ISO3", "NationalPriorDomainInvariant": False}
    B.build_country_mask_from_shapes(rows, prior, config)
    out = capsys.readouterr().out
    assert "extend beyond the inversion domain" in out
    assert "NationalPriorDomainInvariant" in out


def test_no_partial_warning_when_domain_invariant_on(capsys):
    # With domain-invariance ON the scaling is applied, so no "should have enabled it" warning.
    _load_shapes_or_skip()
    xr = pytest.importorskip("xarray")
    lon = np.arange(-76.0, -72.0, 0.5)
    lat = np.arange(2.0, 6.0, 0.5)
    prior = xr.Dataset(coords={"lat": lat, "lon": lon})
    rows = [{"country_id": "COL", "country_name": "COL", "sector": "Oil", "relative_uncertainty": 0.3}]
    config = {"NationalPriorCountryNameColumn": "ISO3", "NationalPriorDomainInvariant": True}
    B.build_country_mask_from_shapes(rows, prior, config)
    assert "extend beyond the inversion domain" not in capsys.readouterr().out
