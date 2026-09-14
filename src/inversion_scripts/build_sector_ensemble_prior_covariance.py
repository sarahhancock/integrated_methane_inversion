from __future__ import annotations

"""Sector-ensemble prior error covariance (PriorCovarianceMethod: sector_ensemble).

Builds the full data-driven prior error covariance Sa by summing, in absolute emission^2
units, three per-sector contributions and then decomposing the result exactly into the
standard (unit-diagonal correlation, per-element sigma) contract:

  1. Anthropogenic sectors  -> two-component national covariance (national rank-1 set so
     the per-country aggregate equals the reported inventory uncertainty u_BTR, plus a
     grid-scale local-diagonal excess). Reused from build_national_inventory_prior_covariance.
  2. Wetlands               -> ensemble covariance from a wetland model ensemble:
     diag(sigma_i E_i) exp(-d_ij / L) diag(sigma_j E_j), with a per-cell relative error
     sigma_i (inter-model spread) mapped onto the state-vector elements and a correlation
     length L fit from the ensemble variogram.  Enabled by SectorEnsembleWetlandFile.
  3. Remaining natural sectors + OtherAnth -> a generic correlated block
     diag(sigma E) [exp(-d/L) o S] diag(sigma E) with sigma=0.5, L=200 km (Yu et al. 2021)
     and S the cosine similarity of the cells' sectoral composition.

Every block is built in absolute units, summed, divided by outer(E_total, E_total), and
split exactly into (C, sigma) -- identical output contract to the other prior-covariance
builders, so invert.py needs no changes.

CLI (matches build_national_inventory_prior_covariance.py):
  build_sector_ensemble_prior_covariance.py StateVectorFile PriorEmisDir config StartDate EndDate nBufferClusters
"""

import sys

import numpy as np
import xarray as xr

try:
    from src.utilities.config_utils import load_config
except ModuleNotFoundError:
    from config_utils import load_config

from utils import ensure_float_list, get_mean_emissions

# reuse the national-inventory builder's helpers (both scripts live in the inversion run dir)
try:
    from src.inversion_scripts.build_national_inventory_prior_covariance import (
        append_buffer_elements,
        build_all_ones_country_mask,
        build_country_mask_from_shapes,
        build_saunois_default_table,
        decompose_relative_covariance,
        get_sector_fields,
        load_country_mask,
        nearest_positive_semidefinite_correlation,
        read_uncertainty_table,
        sector_key,
        select_state_vector_subset,
        state_vector_ids_and_mask,
        emission_weighted_element_table,
        two_component_absolute,
        write_diagnostics,
    )
except ModuleNotFoundError:
    from build_national_inventory_prior_covariance import (
        append_buffer_elements,
        build_all_ones_country_mask,
        build_country_mask_from_shapes,
        build_saunois_default_table,
        decompose_relative_covariance,
        get_sector_fields,
        load_country_mask,
        nearest_positive_semidefinite_correlation,
        read_uncertainty_table,
        sector_key,
        select_state_vector_subset,
        state_vector_ids_and_mask,
        emission_weighted_element_table,
        two_component_absolute,
        write_diagnostics,
    )

# Anthropogenic sectors that carry a reported national (BTR) uncertainty and so get the
# two-component national covariance. Others present in the prior are handled by the wetland
# ensemble (Wetlands) or the generic block (everything else). Overridable via config.
DEFAULT_TWO_COMPONENT_SECTORS = [
    "Livestock", "Rice", "Landfills", "Wastewater", "Coal", "Gas", "Oil",
]
EARTH_RADIUS_KM = 6371.0


def great_circle_matrix(lat, lon):
    """(n, n) great-circle distance in km between element centroids."""
    la = np.radians(np.asarray(lat, dtype=np.float64))
    lo = np.radians(np.asarray(lon, dtype=np.float64))
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2.0) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def element_geometry_and_sector_emissions(state_vector_subset, prior, sector_fields, roi_ids):
    """Per-element centroid (lat, lon) and per-element per-sector absolute emission.

    Returns (elat, elon, sector_emis) where sector_emis[sector_key] is an (n,) array of
    that sector's total emission (EmisCH4_field * AREA) summed over the element's cells.
    """
    labels = state_vector_subset.values
    lat2d, lon2d = np.meshgrid(prior.lat.values, prior.lon.values, indexing="ij")
    area = prior["AREA"].values if "AREA" in prior else np.ones_like(labels, dtype=float)
    id_to_pos = {int(label): idx for idx, label in enumerate(roi_ids)}
    n = len(roi_ids)
    elat = np.zeros(n, dtype=np.float64)
    elon = np.zeros(n, dtype=np.float64)
    sector_emis = {sector_key(f): np.zeros(n, dtype=np.float64) for f in sector_fields}
    weighted = {f: np.asarray(prior[f].values, dtype=np.float64) * area for f in sector_fields}
    for label in roi_ids:
        mask = labels == int(label)
        if not np.any(mask):
            continue
        pos = id_to_pos[int(label)]
        elat[pos] = float(np.mean(lat2d[mask]))
        elon[pos] = float(np.mean(lon2d[mask]))
        for f in sector_fields:
            sector_emis[sector_key(f)][pos] = float(np.nansum(weighted[f][mask]))
    return elat, elon, sector_emis


def load_wetland_ensemble_sigma(path, config, elat, elon):
    """Map a gridded wetland ensemble relative error onto the state-vector elements.

    The ensemble file (npz or netCDF) provides a per-cell relative 1-sigma error
    (inter-model spread; variable SectorEnsembleWetlandVar, default 'rel') on a lat/lon
    grid, and optionally a scalar correlation length 'L_km'. A 3-D (month, lat, lon) field
    is averaged over months. Returns (sigma_per_element, length_km).
    """
    var = config.get("SectorEnsembleWetlandVar", "rel")
    length_km = None
    if str(path).endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if var not in data.files:
            raise ValueError(f"SectorEnsembleWetlandFile {path} has no variable {var!r}; found {list(data.files)}")
        rel = np.asarray(data[var], dtype=np.float64)
        wlat = np.asarray(data["lat"], dtype=np.float64)
        wlon = np.asarray(data["lon"], dtype=np.float64)
        if "L_km" in data.files:
            length_km = float(data["L_km"])
    else:
        ds = xr.open_dataset(path)
        rel = np.asarray(ds[var].values, dtype=np.float64)
        wlat = np.asarray(ds["lat"].values, dtype=np.float64)
        wlon = np.asarray(ds["lon"].values, dtype=np.float64)
        if "L_km" in ds:
            length_km = float(ds["L_km"].values)
    if rel.ndim == 3:            # (month, lat, lon) -> annual mean
        rel = np.nanmean(rel, axis=0)
    finite = np.isfinite(rel)
    median = float(np.nanmedian(rel[finite])) if finite.any() else 0.5
    sigma = np.full(len(elat), median, dtype=np.float64)
    for i in range(len(elat)):
        jlat = int(np.argmin(np.abs(wlat - elat[i])))
        jlon = int(np.argmin(np.abs(wlon - elon[i])))
        value = rel[jlat, jlon]
        if np.isfinite(value):
            sigma[i] = value
    scale = float(config.get("SectorEnsembleWetlandSigmaScale", 1.0))
    floor = float(config.get("SectorEnsembleSigmaFloor", 0.15))
    cap = float(config.get("SectorEnsembleSigmaCap", 2.0))
    sigma = np.clip(sigma * scale, floor, cap)
    length_km = float(config.get("SectorEnsembleWetlandLengthKm", length_km if length_km else 161.0))
    return sigma, length_km


def main(sv_path, prior_emis_dir, config_path, start_date, end_date, nbuffer_elements):
    config = load_config(config_path)
    nbuffer_elements = int(nbuffer_elements)
    prior_sigma = float(ensure_float_list(config["PriorError"])[0])

    prior = get_mean_emissions(start_date, end_date, prior_emis_dir)
    configured_sector_fields = config.get("NationalPriorSectorFields", None)
    if configured_sector_fields:
        missing = [f for f in configured_sector_fields if f not in prior]
        if missing:
            raise ValueError(f"Configured sector fields missing from prior: {missing}")
        prior.attrs["_configured_sector_fields"] = list(configured_sector_fields)
    sector_fields = get_sector_fields(prior)
    sector_names = [sector_key(f) for f in sector_fields]

    state_vector = xr.open_dataset(sv_path)
    if "time" in state_vector.dims:
        state_vector = state_vector.isel(time=0)
    state_vector_subset = select_state_vector_subset(state_vector, prior)
    roi_ids, _ = state_vector_ids_and_mask(state_vector_subset, nbuffer_elements)
    n = len(roi_ids)

    # ---- national (BTR) uncertainties + country mask, for the two-component anthro block ----
    uncertainty_path = config.get("NationalPriorUncertaintyFile")
    country_mask_path = config.get("NationalPriorCountryMaskFile")
    country_mask_var = config.get("NationalPriorCountryMaskVariable", "country_id")
    country_shapefile = config.get("NationalPriorCountryShapefile")
    if uncertainty_path and not (country_mask_path or country_shapefile):
        raise ValueError(
            "NationalPriorUncertaintyFile was set but no country mask was provided. "
            "Set NationalPriorCountryMaskFile or NationalPriorCountryShapefile, or omit "
            "NationalPriorUncertaintyFile to use the Saunois global sectoral defaults."
        )
    using_global_defaults = not uncertainty_path
    if using_global_defaults:
        print("NationalPriorUncertaintyFile not set; using Saunois et al. global sectoral defaults for the anthro block.")
        uncertainty_rows = build_saunois_default_table()
        country_mask = build_all_ones_country_mask(prior)
    else:
        uncertainty_rows = read_uncertainty_table(uncertainty_path)
        if country_mask_path:
            country_mask = load_country_mask(country_mask_path, country_mask_var)
        else:
            country_mask = build_country_mask_from_shapes(uncertainty_rows, prior, config)

    rows, totals_by_element, neff_sum, neff_sumsq = emission_weighted_element_table(
        state_vector_subset, country_mask, prior, sector_fields, roi_ids
    )

    # geometry + per-element per-sector emission (for the ensemble / generic blocks)
    elat, elon, sector_emis = element_geometry_and_sector_emissions(
        state_vector_subset, prior, sector_fields, roi_ids
    )
    dist = great_circle_matrix(elat, elon)

    diagnostics = []

    # ---- (1) anthropogenic two-component national covariance (absolute units) ----
    two_component_sectors = list(config.get("SectorEnsembleTwoComponentSectors", DEFAULT_TWO_COMPONENT_SECTORS))
    grid_national_ratio = float(config.get("NationalPriorGridNationalRatio", 2.5))
    min_uncertainty = float(config.get("NationalPriorMinUncertainty", 0.30))
    anthro_rows = [r for r in uncertainty_rows if r["sector"] in two_component_sectors]
    Sa_abs, tc_diag = two_component_absolute(
        rows, neff_sum, neff_sumsq, anthro_rows, n, grid_national_ratio, min_uncertainty
    )
    diagnostics.extend(tc_diag)
    if not tc_diag:
        print("WARNING: no (country, sector) anthropogenic groups matched; the anthro block is empty. "
              "Check country IDs/mask and sector names.")

    # ---- (2) wetland ensemble covariance (absolute units) ----
    wetland_file = config.get("SectorEnsembleWetlandFile")
    wetland_handled = False
    if wetland_file and "Wetlands" in sector_emis:
        sigma_w, length_w = load_wetland_ensemble_sigma(wetland_file, config, elat, elon)
        e_w = sector_emis["Wetlands"]
        we = sigma_w * e_w
        Sa_abs += np.outer(we, we) * np.exp(-dist / length_w)
        wetland_handled = True
        pos = e_w > 0
        diagnostics.append({
            "sector": "Wetlands", "status": "ok", "method": "ensemble",
            "length_km": length_w,
            "sigma_median": float(np.median(sigma_w[pos])) if pos.any() else 0.0,
            "n_elements": int(pos.sum()),
        })
        print(f"Wetland ensemble covariance: L={length_w:.0f} km, "
              f"sigma median={np.median(sigma_w[pos]) if pos.any() else 0:.2f} over {int(pos.sum())} elements.")
    elif "Wetlands" in sector_emis:
        print("SectorEnsembleWetlandFile not set; Wetlands folded into the generic correlated block.")

    # ---- (3) generic correlated block for remaining sectors (naturals + OtherAnth) ----
    generic_default = [
        s for s in sector_names
        if s not in two_component_sectors and not (s == "Wetlands" and wetland_handled)
    ]
    generic_sectors = list(config.get("SectorEnsembleGenericSectors", generic_default))
    generic_sectors = [s for s in generic_sectors if s in sector_emis]
    if generic_sectors:
        generic_sigma = float(config.get("SectorEnsembleGenericSigma", 0.5))
        generic_length = float(config.get("SectorEnsembleGenericLengthKm", 200.0))
        comp = np.stack([sector_emis[s] for s in generic_sectors], axis=1)   # (n, n_generic)
        e_non = comp.sum(axis=1)
        norm = np.linalg.norm(comp, axis=1)
        unit = np.zeros_like(comp)
        nz = norm > 0
        unit[nz] = comp[nz] / norm[nz, None]
        similarity = unit @ unit.T                                          # cosine similarity (PSD)
        sb = generic_sigma * e_non
        Sa_abs += np.outer(sb, sb) * np.exp(-dist / generic_length) * similarity
        diagnostics.append({
            "sector": "+".join(generic_sectors), "status": "ok", "method": "generic",
            "length_km": generic_length, "sigma": generic_sigma,
            "n_elements": int((e_non > 0).sum()),
        })
        print(f"Generic correlated block: sectors={generic_sectors}, sigma={generic_sigma}, "
              f"L={generic_length:.0f} km.")

    # ---- decompose to (correlation, sigma), PSD-repair, append buffer, write ----
    covariance, sigma_vector = decompose_relative_covariance(Sa_abs, totals_by_element, prior_sigma)
    covariance, min_eig_before, min_eig_after = nearest_positive_semidefinite_correlation(covariance)
    state_vector_ids, covariance = append_buffer_elements(roi_ids, covariance, nbuffer_elements)
    sigma_scale = np.concatenate(
        [sigma_vector / prior_sigma, np.ones(nbuffer_elements)]
    ).astype(np.float32)

    np.savez(
        "prior_norm_error_covariance.npz",
        covariance=covariance.astype(np.float32),
        state_vector_ids=state_vector_ids.astype(np.int32),
        sigma_scale=sigma_scale,
    )
    write_diagnostics("sector_ensemble_prior_covariance_diagnostics.csv", diagnostics)
    print(
        f"Wrote sector-ensemble prior covariance with shape {covariance.shape}; "
        f"anthro groups={len(tc_diag)}; wetland_ensemble={wetland_handled}; "
        f"generic_sectors={len(generic_sectors)}; "
        f"min_eig_before={min_eig_before:.3e}; min_eig_after={min_eig_after:.3e}"
    )


if __name__ == "__main__":
    main(
        sv_path=sys.argv[1],
        prior_emis_dir=sys.argv[2],
        config_path=sys.argv[3],
        start_date=sys.argv[4],
        end_date=sys.argv[5],
        nbuffer_elements=int(sys.argv[6]),
    )
