from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from src.utilities.config_utils import load_config
except ModuleNotFoundError:
    from config_utils import load_config

from utils import ensure_float_list, get_mean_emissions


AGGREGATE_FIELDS = {"EmisCH4_Total", "EmisCH4_Total_ExclSoilAbs"}
EXCLUDED_SECTORS = {"EmisCH4_SoilAbsorb"}

# Fallback sectoral uncertainties (fraction) for a (country, sector) with NO reported
# BTR value. ANTHROPOGENIC defaults are the CROSS-COUNTRY MEDIAN of reported BTR CH4
# uncertainties (118-country compilation), which are 2-6x larger than the old Saunois
# bottom-up lows (e.g. oil 15%->50%, livestock 5%->30%); the inversion chi^2 shows the
# lows are far too tight (see methods_Sa.tex Sect. "Prior calibration"). NATURAL defaults
# stay at the Saunois inter-model spread. TODO(user): replace per-country anthro values
# with detailed IPCC-methods uncertainties as they are filled in.
# Keys match HEMCO EmisCH4_* field suffixes; missing keys are silently ignored.
SAUNOIS_GLOBAL_DEFAULTS = {
    "Agriculture": 0.30,
    "Livestock": 0.30,
    "Rice": 0.45,
    "Landfills": 0.47,
    "Wastewater": 0.58,
    "FossilFuels": 0.50,
    "Coal": 0.50,
    "Oil": 0.50,
    "Gas": 0.50,
    "OilGas": 0.50,
    "Industry": 0.80,
    "Transport": 0.50,
    "Other": 0.80,
    "BiofuelBurn": 0.45,
    "BiomassBurn": 0.41,
    "Wetlands": 0.28,
    "Reservoirs": 0.80,
    "InlandWaters": 0.80,
    "Seeps": 0.60,
    "Termites": 0.60,
}


def build_saunois_default_table():
    """Synthetic uncertainty table using Saunois et al. global sectoral defaults.

    Treats the entire inversion domain as one region so that all grid cells
    sharing a sector are correlated regardless of country boundaries.
    """
    return [
        {
            "country_id": "1",
            "country_name": "global",
            "sector": sector,
            "relative_uncertainty": unc,
        }
        for sector, unc in SAUNOIS_GLOBAL_DEFAULTS.items()
    ]


def build_all_ones_country_mask(prior):
    """Country mask with value 1 everywhere — treats whole domain as one region."""
    return xr.DataArray(
        np.ones((prior.sizes["lat"], prior.sizes["lon"]), dtype=float),
        coords={"lat": prior.lat.values, "lon": prior.lon.values},
        dims=("lat", "lon"),
    )


def normalize_country_id(country_id):
    """Return a stable string key for numeric or domain-wide country IDs."""
    text = str(country_id).strip()
    try:
        value = float(text)
    except ValueError:
        return text
    if np.isfinite(value) and value.is_integer():
        return str(int(value))
    return text


def get_sector_fields(prior):
    configured = prior.attrs.get("_configured_sector_fields")
    if configured:
        return configured
    fields = []
    for name in prior.data_vars:
        if not name.startswith("EmisCH4_"):
            continue
        if name in AGGREGATE_FIELDS or name in EXCLUDED_SECTORS:
            continue
        if prior[name].dims == ("lat", "lon"):
            fields.append(name)
    if not fields:
        raise ValueError("No sector-resolved EmisCH4_* fields found in prior emissions.")
    return fields


def sector_key(field):
    return field.replace("EmisCH4_", "")


def read_uncertainty_table(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sector", "relative_uncertainty"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            country_id = str(row.get("country_id", "")).strip()
            country_name = str(row.get("country", row.get("country_name", ""))).strip()
            if not (country_id or country_name) or not row.get("sector"):
                continue
            rows.append(
                {
                    "country_id": country_id or country_name,
                    "country_name": country_name,
                    "sector": str(row["sector"]).strip(),
                    "relative_uncertainty": float(row["relative_uncertainty"]),
                }
            )
    if not rows:
        raise ValueError(f"No usable uncertainty rows found in {path}")
    return rows


def load_country_mask(path, variable):
    ds = xr.open_dataset(path)
    if variable not in ds:
        raise ValueError(f"Country mask variable {variable!r} not found in {path}")
    return ds[variable]


def load_country_shapes(path, name_column):
    import geopandas as gpd

    shapes = gpd.read_file(path)
    if name_column not in shapes.columns:
        raise ValueError(f"Country shapefile column {name_column!r} not found in {path}")
    return shapes


def get_country_fraction_mask(country, lat, lon, shapes, name_column, area_weighting=True):
    import geopandas as gpd
    import regionmask
    from shapely.geometry import box

    country_shape = shapes[shapes[name_column] == country]
    if country_shape.empty:
        raise ValueError(f"Country {country!r} not found in shapefile column {name_column!r}")

    if not area_weighting:
        mask = np.array(regionmask.mask_geopandas(country_shape, lon, lat) + 1)
        mask[np.isnan(mask)] = 0
        mask[mask > 0] = 1
        return mask.astype(float)

    dlat = float(np.abs(lat[1] - lat[0]))
    dlon = float(np.abs(lon[1] - lon[0]))
    grid_cells = []
    indices = []
    for i, y in enumerate(lat):
        for j, x in enumerate(lon):
            grid_cells.append(box(x - dlon / 2, y - dlat / 2, x + dlon / 2, y + dlat / 2))
            indices.append((i, j))

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326").to_crs(shapes.crs)
    country_projected = country_shape.to_crs(shapes.crs)
    mask = np.zeros((len(lat), len(lon)), dtype=float)
    for geom, (i, j) in zip(grid.geometry, indices):
        if geom.is_empty or geom.area == 0:
            continue
        intersection = country_projected.intersection(geom)
        if not intersection.is_empty.all():
            mask[i, j] = float(intersection.area.sum() / geom.area)
    return mask


def build_country_mask_from_shapes(uncertainty_rows, prior, config):
    shapefile = config.get("NationalPriorCountryShapefile")
    name_column = config.get("NationalPriorCountryNameColumn", "NAME")
    area_weighting = bool(config.get("NationalPriorCountryMaskAreaWeighting", True))
    if not shapefile:
        return None

    shapes = load_country_shapes(shapefile, name_column)
    mask = xr.DataArray(
        np.zeros((prior.sizes["lat"], prior.sizes["lon"]), dtype=float),
        coords={"lat": prior.lat.values, "lon": prior.lon.values},
        dims=("lat", "lon"),
    )
    country_lookup = {}
    next_id = 1
    for row in uncertainty_rows:
        country_name = row.get("country_name") or row["country_id"]
        if country_name in country_lookup:
            row["country_id"] = str(country_lookup[country_name])
            continue
        country_id = int(row["country_id"]) if str(row["country_id"]).isdigit() else next_id
        next_id = max(next_id, country_id + 1)
        fraction = get_country_fraction_mask(
            country_name,
            prior.lat.values,
            prior.lon.values,
            shapes,
            name_column,
            area_weighting=area_weighting,
        )
        current = mask.values
        replace = fraction > current
        current[replace] = country_id
        mask.values[:] = current
        country_lookup[country_name] = country_id
        row["country_id"] = str(country_id)
    return mask


def select_state_vector_subset(state_vector, prior):
    subset = state_vector["StateVector"].sel(lat=prior.lat, lon=prior.lon)
    if subset.shape != (prior.sizes["lat"], prior.sizes["lon"]):
        raise ValueError("State vector subset shape does not match prior grid shape.")
    return subset


def state_vector_ids_and_mask(state_vector_subset, nbuffer_elements):
    labels = state_vector_subset.values
    valid = np.isfinite(labels) & (labels > 0)
    if nbuffer_elements > 0:
        last_roi = int(np.nanmax(labels)) - nbuffer_elements
        valid &= labels <= last_roi
    ids = np.unique(labels[valid].astype(np.int32))
    ids.sort()
    return ids, valid


def emission_weighted_element_table(state_vector_subset, country_mask, prior, sector_fields, roi_ids):
    labels = state_vector_subset.values
    countries = country_mask.sel(lat=prior.lat, lon=prior.lon, method="nearest").values
    area = prior["AREA"].values if "AREA" in prior else np.ones_like(labels, dtype=float)

    id_to_pos = {int(label): idx for idx, label in enumerate(roi_ids)}
    rows = defaultdict(float)
    totals_by_element = np.zeros(len(roi_ids), dtype=np.float64)
    # For the two-component prior: per (sector, element) effective number of independent
    # native cells n_eff = (sum e)^2 / sum e^2, used to set the grid:national error ratio.
    neff_sum = defaultdict(float)     # (key, pos) -> sum of native-cell emissions
    neff_sumsq = defaultdict(float)   # (key, pos) -> sum of native-cell emissions squared

    for field in sector_fields:
        emis = np.asarray(prior[field].values, dtype=np.float64) * area
        key = sector_key(field)
        for label in roi_ids:
            element_mask = labels == label
            if not np.any(element_mask):
                continue
            pos = id_to_pos[int(label)]
            sector_total = float(np.nansum(emis[element_mask]))
            totals_by_element[pos] += max(sector_total, 0.0)

            country_values = countries[element_mask]
            emis_values = emis[element_mask]
            cell_emis = emis_values[np.isfinite(emis_values)]
            cell_emis = cell_emis[cell_emis > 0]
            if cell_emis.size:
                neff_sum[(key, pos)] += float(cell_emis.sum())
                neff_sumsq[(key, pos)] += float((cell_emis ** 2).sum())
            for country in np.unique(country_values[np.isfinite(country_values)]):
                country_emis = float(np.nansum(emis_values[country_values == country]))
                if country_emis > 0:
                    rows[(normalize_country_id(country), key, pos)] += country_emis

    return rows, totals_by_element, neff_sum, neff_sumsq


def solve_group_correlations(rows, uncertainty_rows, prior_sigma, sigma_by_pos=None):
    # sigma_by_pos: optional {pos: per-element sigma}. When given (per-sector amplitude in
    # effect), the group correlation rho is solved with the true per-element sigma so the
    # achieved aggregate still matches the target; otherwise a uniform prior_sigma is used.
    emissions_by_group = defaultdict(lambda: defaultdict(float))
    for (country_id, sector, pos), value in rows.items():
        emissions_by_group[(country_id, sector)][pos] += value

    group_rho = {}
    diagnostics = []
    for row in uncertainty_rows:
        group = (row["country_id"], row["sector"])
        if group not in emissions_by_group:
            diagnostics.append({**row, "status": "missing_emissions"})
            continue
        items = emissions_by_group[group]
        positions = np.array(list(items.keys()))
        emis = np.array(list(items.values()), dtype=np.float64)
        if sigma_by_pos is None:
            sig = np.full(emis.size, prior_sigma, dtype=np.float64)
        else:
            sig = np.array([sigma_by_pos.get(int(p), prior_sigma) for p in positions], dtype=np.float64)
        total = float(np.sum(emis))
        target_var = (row["relative_uncertainty"] * total) ** 2
        diag_var = float(np.sum((sig * emis) ** 2))
        full_var = float(np.sum(sig * emis) ** 2)
        denom = full_var - diag_var
        rho = 0.0 if denom <= 0 else (target_var - diag_var) / denom
        clipped = float(np.clip(rho, 0.0, 1.0))
        group_rho[group] = clipped
        achieved_var = diag_var + clipped * denom
        diagnostics.append(
            {
                **row,
                "status": "ok",
                "rho_raw": rho,
                "rho_used": clipped,
                "emissions_total": total,
                "target_relative_uncertainty": row["relative_uncertainty"],
                "independent_relative_uncertainty": np.sqrt(diag_var) / total,
                "fully_correlated_relative_uncertainty": np.sqrt(full_var) / total,
                "achieved_relative_uncertainty": np.sqrt(achieved_var) / total,
            }
        )
    return group_rho, diagnostics


def build_weighted_correlation(rows, totals_by_element, group_rho):
    n = len(totals_by_element)
    covariance = np.eye(n, dtype=np.float64)
    pair_numerator = np.zeros((n, n), dtype=np.float64)
    pair_weight = np.zeros((n, n), dtype=np.float64)

    by_group = defaultdict(dict)
    for (country_id, sector, pos), value in rows.items():
        group = (country_id, sector)
        if group in group_rho and totals_by_element[pos] > 0:
            by_group[group][pos] = value / totals_by_element[pos]

    for group, element_weights in by_group.items():
        rho = group_rho[group]
        positions = np.array(list(element_weights), dtype=np.int32)
        weights = np.array([element_weights[pos] for pos in positions], dtype=np.float64)
        shared = np.sqrt(np.outer(weights, weights))
        pair_numerator[np.ix_(positions, positions)] += rho * shared
        pair_weight[np.ix_(positions, positions)] += shared

    mask = pair_weight > 0
    covariance[mask] = pair_numerator[mask] / pair_weight[mask]
    np.fill_diagonal(covariance, 1.0)
    return np.clip(covariance, 0.0, 1.0)


def two_component_absolute(
    rows, neff_sum, neff_sumsq, uncertainty_rows, n_elements,
    grid_national_ratio=2.5, min_uncertainty=0.30,
):
    """Two-component anthropogenic prior error covariance in ABSOLUTE emission^2 units.

    Per (country, sector) group the within-country prior covariance is
        national rank-1 :  snat^2 * E_i * E_j
        local diagonal  :  snat^2 * (ratio_i^2 - 1) * E_i^2
    with ratio_i = sqrt(1 + (R01^2 - 1) / n_eff_i) the grid:national error ratio
    (n_eff_i = effective number of independent native cells in the element), and
        snat = u_BTR / sqrt(1 + Q),   Q = sum_i (ratio_i^2 - 1) E_i^2 / E_c^2,
    chosen so the national aggregate uncertainty equals u_BTR EXACTLY.  Sector
    contributions are summed (sectors are independent).  Only groups that appear in
    uncertainty_rows contribute; every other element stays zero here (its variance is
    supplied by another block, e.g. the wetland ensemble, or the uniform fallback in
    decompose_relative_covariance).  Returns (Sa_abs, diagnostics).
    """
    Sa_abs = np.zeros((n_elements, n_elements), dtype=np.float64)

    emissions_by_group = defaultdict(lambda: defaultdict(float))
    for (country_id, sector, pos), value in rows.items():
        emissions_by_group[(country_id, sector)][pos] += value
    u_by_group = {
        (row["country_id"], row["sector"]): max(float(row["relative_uncertainty"]), min_uncertainty)
        for row in uncertainty_rows
    }

    diagnostics = []
    for group, items in emissions_by_group.items():
        u = u_by_group.get(group)
        if u is None:
            continue
        country_id, sector = group
        positions = np.array(list(items.keys()), dtype=int)
        emis = np.array([items[p] for p in positions], dtype=np.float64)
        total = float(emis.sum())
        if total <= 0:
            continue
        n_eff = np.array([
            (neff_sum[(sector, int(p))] ** 2 / neff_sumsq[(sector, int(p))])
            if neff_sumsq.get((sector, int(p)), 0.0) > 0 else 1.0
            for p in positions
        ])
        n_eff = np.clip(n_eff, 1.0, None)
        r2 = (grid_national_ratio ** 2 - 1.0) / n_eff          # ratio_i^2 - 1 (local excess)
        Q = float(np.sum(r2 * emis ** 2)) / (total * total)
        snat = u / np.sqrt(1.0 + Q)
        Sa_abs[np.ix_(positions, positions)] += (snat * snat) * np.outer(emis, emis)  # national rank-1
        Sa_abs[positions, positions] += (snat * snat) * r2 * emis ** 2                # local diagonal
        achieved = snat * np.sqrt(total * total + float(np.sum(r2 * emis ** 2))) / total
        diagnostics.append({
            "country_id": country_id, "sector": sector, "status": "ok",
            "relative_uncertainty": u, "achieved_relative_uncertainty": achieved,
            "n_elements": int(positions.size), "median_ratio": float(np.median(np.sqrt(1.0 + r2))),
        })
    return Sa_abs, diagnostics


def decompose_relative_covariance(Sa_abs, totals_by_element, prior_sigma):
    """Convert an ABSOLUTE emission^2 covariance to a scale-factor covariance and split it
    exactly into a unit-diagonal correlation C and a per-element sigma:
        Sa_rel = Sa_abs / outer(E_total, E_total),   Sa_rel = sigma * C * sigma.
    Elements with zero emission (hence zero variance) fall back to the uniform
    prior_sigma, uncorrelated.  Returns (C, sigma)."""
    total_emis = np.asarray(totals_by_element, dtype=np.float64)
    total_emis_floored = np.where(total_emis > 0, total_emis, 1.0)
    Sa_rel = Sa_abs / np.outer(total_emis_floored, total_emis_floored)   # scale-factor covariance
    sigma = np.sqrt(np.clip(np.diag(Sa_rel), 0.0, None))
    sig_safe = np.where(sigma > 0, sigma, 1.0)
    C = Sa_rel / np.outer(sig_safe, sig_safe)
    np.fill_diagonal(C, 1.0)
    sigma_out = np.where(sigma > 0, sigma, prior_sigma)
    return C, sigma_out


def build_two_component_covariance(
    rows, neff_sum, neff_sumsq, uncertainty_rows, totals_by_element,
    prior_sigma, grid_national_ratio=2.5, min_uncertainty=0.30,
):
    """Exact two-component prior error covariance (returns correlation C, per-element sigma,
    diagnostics).  Thin wrapper: assemble the absolute covariance (two_component_absolute)
    then decompose it exactly into (C, sigma) (decompose_relative_covariance).  Fits the
    standard (correlation, sigma_scale) output contract with no approximation."""
    n = len(totals_by_element)
    Sa_abs, diagnostics = two_component_absolute(
        rows, neff_sum, neff_sumsq, uncertainty_rows, n, grid_national_ratio, min_uncertainty
    )
    C, sigma_out = decompose_relative_covariance(Sa_abs, totals_by_element, prior_sigma)
    return C, sigma_out, diagnostics


def append_buffer_elements(state_vector_ids, covariance, nbuffer_elements):
    if nbuffer_elements == 0:
        return state_vector_ids, covariance
    original_size = covariance.shape[0]
    expanded_size = original_size + nbuffer_elements
    expanded = np.zeros((expanded_size, expanded_size), dtype=np.float64)
    expanded[:original_size, :original_size] = covariance
    expanded[original_size:, original_size:] = np.eye(nbuffer_elements)
    extra_ids = np.arange(original_size + 1, expanded_size + 1, dtype=state_vector_ids.dtype)
    return np.concatenate([state_vector_ids, extra_ids]), expanded


def nearest_positive_semidefinite_correlation(covariance, floor=1.0e-6):
    sym = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    min_before = float(np.min(eigvals))
    eigvals = np.clip(eigvals, floor, None)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    diag = np.sqrt(np.clip(np.diag(repaired), floor, None))
    repaired = repaired / diag[:, None] / diag[None, :]
    np.fill_diagonal(repaired, 1.0)
    min_after = float(np.min(np.linalg.eigvalsh(repaired)))
    return repaired, min_before, min_after


def write_diagnostics(path, diagnostics):
    if not diagnostics:
        return
    fieldnames = sorted({key for row in diagnostics for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)


def main(sv_path, prior_emis_dir, config_path, start_date, end_date, nbuffer_elements):
    config = load_config(config_path)
    uncertainty_path = config.get("NationalPriorUncertaintyFile")
    country_mask_path = config.get("NationalPriorCountryMaskFile")
    country_mask_var = config.get("NationalPriorCountryMaskVariable", "country_id")
    country_shapefile = config.get("NationalPriorCountryShapefile")
    if uncertainty_path and not (country_mask_path or country_shapefile):
        raise ValueError(
            "NationalPriorUncertaintyFile was set but no country mask was provided. "
            "Set NationalPriorCountryMaskFile or NationalPriorCountryShapefile in config.yml, "
            "or omit NationalPriorUncertaintyFile to use Saunois et al. global defaults."
        )
    using_global_defaults = not uncertainty_path

    prior_sigma = float(ensure_float_list(config["PriorError"])[0])
    prior = get_mean_emissions(start_date, end_date, prior_emis_dir)
    configured_sector_fields = config.get("NationalPriorSectorFields", None)
    if configured_sector_fields:
        missing = [field for field in configured_sector_fields if field not in prior]
        if missing:
            raise ValueError(f"Configured sector fields missing from prior: {missing}")
        prior.attrs["_configured_sector_fields"] = list(configured_sector_fields)

    sector_fields = get_sector_fields(prior)
    state_vector = xr.open_dataset(sv_path)
    if "time" in state_vector.dims:                      # some StateVector files have no time dim
        state_vector = state_vector.isel(time=0)
    state_vector_subset = select_state_vector_subset(state_vector, prior)
    roi_ids, _ = state_vector_ids_and_mask(state_vector_subset, int(nbuffer_elements))

    if using_global_defaults:
        print("NationalPriorUncertaintyFile not set; using Saunois et al. global sectoral defaults.")
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
    two_component = str(config.get("NationalPriorTwoComponent", False)).strip().lower() in ("true", "1", "yes")
    if two_component:
        # EXACT two-component prior: per (country, sector) national rank-1 + local-diagonal excess,
        # snat set so the national aggregate = u_BTR, summed over sectors, decomposed exactly into a
        # unit-diagonal correlation + per-element sigma.
        grid_national_ratio = float(config.get("NationalPriorGridNationalRatio", 2.5))
        min_uncertainty = float(config.get("NationalPriorMinUncertainty", 0.30))
        covariance, sigma_vector, diagnostics = build_two_component_covariance(
            rows, neff_sum, neff_sumsq, uncertainty_rows, totals_by_element,
            prior_sigma, grid_national_ratio, min_uncertainty,
        )
        if not diagnostics:
            raise ValueError(
                "Two-component prior: no (country, sector) groups matched sector emissions. "
                "Check country IDs/mask values and sector names in NationalPriorUncertaintyFile."
            )
        ratios = [d["achieved_relative_uncertainty"] / d["relative_uncertainty"]
                  for d in diagnostics if d["relative_uncertainty"] > 0]
        print(f"Two-component prior: {len(diagnostics)} (country, sector) groups; national "
              f"aggregate / u_BTR median={np.median(ratios):.4f} [{min(ratios):.4f}, {max(ratios):.4f}] "
              f"(should be 1.0); R01={grid_national_ratio}, BTR floor={min_uncertainty}")
        covariance, min_eig_before, min_eig_after = nearest_positive_semidefinite_correlation(covariance)
        state_vector_ids, covariance = append_buffer_elements(roi_ids, covariance, int(nbuffer_elements))
        # sigma_scale so invert.py forms Sa = (PriorError*sigma_scale) C (PriorError*sigma_scale) = the
        # exact two-component covariance (PriorError cancels; magnitudes come from the BTR-set sigma).
        sigma_scale = np.concatenate(
            [sigma_vector / prior_sigma, np.ones(int(nbuffer_elements))]
        ).astype(np.float32)
    else:
        # One-component path: a single correlation rho per (country, sector) matching the u_BTR aggregate.
        # Optional per-SECTOR per-element amplitude on sigma (e.g. {Wetlands: 5.0}). Emission-weighted
        # over the sectors present in each element; default 1.0.
        sector_amplitude = config.get("NationalPriorSectorAmplitude", {}) or {}
        n_roi = len(totals_by_element)
        amp = np.ones(n_roi, dtype=np.float64)
        if sector_amplitude:
            numer = np.zeros(n_roi); denom = np.zeros(n_roi)
            for (cid, sector, pos), val in rows.items():
                numer[pos] += float(sector_amplitude.get(sector, 1.0)) * val
                denom[pos] += val
            m = denom > 0
            amp[m] = numer[m] / denom[m]
            print(f"NationalPriorSectorAmplitude active: {sector_amplitude}; "
                  f"per-element sigma amplitude range [{amp.min():.2f}, {amp.max():.2f}]")
        sigma_by_pos = ({int(p): prior_sigma * float(amp[int(p)]) for p in range(n_roi)}
                        if sector_amplitude else None)
        group_rho, diagnostics = solve_group_correlations(
            rows, uncertainty_rows, prior_sigma, sigma_by_pos=sigma_by_pos
        )
        ok_rows = [row for row in diagnostics if row.get("status") == "ok"]
        if not ok_rows:
            raise ValueError(
                "No national-inventory prior covariance groups matched sector emissions. "
                "Check country IDs/mask values and sector names in NationalPriorUncertaintyFile. "
                "For domain-wide Saunois defaults this indicates an internal grouping bug."
            )
        covariance = build_weighted_correlation(rows, totals_by_element, group_rho)
        covariance, min_eig_before, min_eig_after = nearest_positive_semidefinite_correlation(
            covariance
        )
        state_vector_ids, covariance = append_buffer_elements(
            roi_ids, covariance, int(nbuffer_elements)
        )
        # per-element sigma amplitude aligned to state_vector_ids (buffer elements = 1.0)
        sigma_scale = np.concatenate([amp, np.ones(int(nbuffer_elements))]).astype(np.float32)

    np.savez(
        "prior_norm_error_covariance.npz",
        covariance=covariance.astype(np.float32),
        state_vector_ids=state_vector_ids.astype(np.int32),
        sigma_scale=sigma_scale,
    )
    write_diagnostics("national_inventory_prior_covariance_diagnostics.csv", diagnostics)
    source = "Saunois global defaults" if using_global_defaults else uncertainty_path
    print(
        f"Wrote national inventory prior covariance (source: {source}) "
        f"with shape {covariance.shape}; diagnostics rows={len(diagnostics)}; "
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
