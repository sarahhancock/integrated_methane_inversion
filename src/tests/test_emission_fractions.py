"""Emission-fraction scaling for the prior covariance (build_global_emission_fractions).

Two quantities are derived from the user's GLOBAL HEMCO inventories so a regional inversion carries only
its share of a global sector / a country:
  * the Saunois GLOBAL-BACKGROUND amplitude for the natural sectors, sigma_sys = G_s * f_s, and
  * the domain-invariant national term's in-domain EMISSION fraction f_C (replacing the AREA fraction).

These tests cover the config-driven resolution/fallbacks (which must NEVER crash a run) and that an
emission-based f_C, when available, overrides the area fraction in build_country_mask_from_shapes.
"""
import os
import tempfile
import warnings

import numpy as np
import pytest

import build_global_emission_fractions as G
import build_national_inventory_prior_covariance as B


# --------------------------- resolve_hemco_config ---------------------------

def test_resolve_hemco_explicit_key_wins():
    fd, path = tempfile.mkstemp(suffix="HEMCO_Config.rc")
    os.close(fd)
    try:
        assert G.resolve_hemco_config({"EmissionFractionHemcoConfig": path}) == path
    finally:
        os.remove(path)


def test_resolve_hemco_none_when_unresolvable():
    assert G.resolve_hemco_config({}) is None
    assert G.resolve_hemco_config({"EmissionFractionHemcoConfig": "/no/such/HEMCO_Config.rc"}) is None
    # OutputPath+RunName that do not contain hemco_prior_emis/HEMCO_Config.rc -> None (no crash)
    assert G.resolve_hemco_config({"OutputPath": "/tmp", "RunName": "definitely_missing_run_xyz"}) is None


def test_resolve_hemco_standard_imi_location():
    with tempfile.TemporaryDirectory() as d:
        run = "myrun"
        hemco_dir = os.path.join(d, run, "hemco_prior_emis")
        os.makedirs(hemco_dir)
        hemco = os.path.join(hemco_dir, "HEMCO_Config.rc")
        open(hemco, "w").close()
        assert G.resolve_hemco_config({"OutputPath": d, "RunName": run}) == hemco


# --------------------------- emission_fractions_for_config (graceful) ---------------------------

def test_fractions_none_when_scaling_disabled():
    # A user can turn the whole thing off; must return None (callers then fall back), no parsing attempted.
    assert G.emission_fractions_for_config({"EmissionFractionScaling": False}, prior=None) is None


def test_fractions_none_when_no_hemco(capsys):
    xr = pytest.importorskip("xarray")
    prior = xr.Dataset(coords={"lat": np.arange(-5.0, 5.0, 1.0), "lon": np.arange(-75.0, -65.0, 1.0)})
    # No HEMCO config resolvable -> None + a message, never an exception.
    assert G.emission_fractions_for_config({"LonMin": -75, "LonMax": -65, "LatMin": -5, "LatMax": 5}, prior) is None
    assert "no HEMCO_Config.rc resolved" in capsys.readouterr().out


# --------------------------- _sector_fields override hook ---------------------------

def test_sector_tokens_override_is_honoured():
    # EmissionFractionSectorMap -> _SECTOR_TOKENS_OVERRIDE lets a user point a sector at their own HEMCO
    # container name. With no matching entries the generator simply yields nothing (no crash).
    G._SECTOR_TOKENS_OVERRIDE = {"Wetlands": ["MY_CUSTOM_WETLAND_TAG"]}
    try:
        fields = list(G._sector_fields("Wetlands", {}))       # empty entries dict -> nothing matches
        assert fields == []
    finally:
        if hasattr(G, "_SECTOR_TOKENS_OVERRIDE"):
            del G._SECTOR_TOKENS_OVERRIDE


# --------------------------- emission f_C overrides area f_C ---------------------------

def _shapes_or_skip():
    pytest.importorskip("geopandas")
    pytest.importorskip("regionmask")
    shp = B.default_country_shapefile()
    if not shp or not os.path.exists(shp):
        pytest.skip("bundled country shapefile not available")
    return B.load_country_shapes(shp, "ISO3")


def test_emission_country_fraction_overrides_area_fraction():
    shapes = _shapes_or_skip()
    xr = pytest.importorskip("xarray")
    # A window comfortably enclosing Colombia -> AREA f_C ~ 1.0.
    lon = np.arange(-80.0, -65.0, 0.5)
    lat = np.arange(-5.0, 14.0, 0.5)
    prior = xr.Dataset(coords={"lat": lat, "lon": lon})
    rows = [{"country_id": "COL", "country_name": "COL", "sector": "Oil", "relative_uncertainty": 0.3}]
    config = {"NationalPriorCountryNameColumn": "ISO3", "NationalPriorDomainInvariant": True}

    # area-based (no emission dict, and no HEMCO config resolvable) -> f_C ~ 1.0
    _, cf_area = B.build_country_mask_from_shapes(rows, prior, config)
    (area_val,) = cf_area.values()
    assert area_val > 0.98

    # supplying an emission-based dict overrides the area fraction for the SAME country id
    _, cf_emis = B.build_country_mask_from_shapes(
        rows, prior, config, emission_country_fraction={"COL": 0.42}
    )
    (emis_val,) = cf_emis.values()
    assert abs(emis_val - 0.42) < 1e-9


def test_emission_fraction_missing_country_falls_back_to_area():
    shapes = _shapes_or_skip()
    xr = pytest.importorskip("xarray")
    lon = np.arange(-80.0, -65.0, 0.5)
    lat = np.arange(-5.0, 14.0, 0.5)
    prior = xr.Dataset(coords={"lat": lat, "lon": lon})
    rows = [{"country_id": "COL", "country_name": "COL", "sector": "Oil", "relative_uncertainty": 0.3}]
    config = {"NationalPriorCountryNameColumn": "ISO3", "NationalPriorDomainInvariant": True}
    # emission dict present but WITHOUT this country -> area fraction is used (not KeyError, not 1.0-by-fiat)
    _, cf = B.build_country_mask_from_shapes(
        rows, prior, config, emission_country_fraction={"ARG": 0.5}
    )
    (val,) = cf.values()
    assert val > 0.98
