"""Sector routing in the sector-ensemble prior covariance.

OtherAnth (residual anthropogenic = fuel combustion, IPCC 1A) carries a reported national (BTR)
uncertainty, so it must get the two-component national covariance like the other anthropogenic
sectors, NOT the generic minor-natural block (decision 2026-09-23: BTR for OtherAnth). The generic
block is for the naturals that have no reported national total (reservoirs, seeps, termites, fires).
"""
from build_sector_ensemble_prior_covariance import DEFAULT_TWO_COMPONENT_SECTORS


def test_otheranth_is_default_two_component():
    # OtherAnth has a reported BTR uncertainty -> BTR two-component, not the generic block.
    assert "OtherAnth" in DEFAULT_TWO_COMPONENT_SECTORS


def test_all_anthropogenic_sectors_are_default_two_component():
    for s in ("Livestock", "Rice", "Landfills", "Wastewater", "Coal", "Gas", "Oil", "OtherAnth"):
        assert s in DEFAULT_TWO_COMPONENT_SECTORS, s


def test_naturals_are_not_default_two_component():
    # Naturals have no reported national total; they go to the wetland ensemble / generic block.
    for s in ("Wetlands", "Reservoirs", "Seeps", "Termites", "BiomassBurn", "Lakes"):
        assert s not in DEFAULT_TWO_COMPONENT_SECTORS, s
