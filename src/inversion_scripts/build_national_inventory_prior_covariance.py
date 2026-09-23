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

# GENUINE Saunois et al. global (bottom-up) 1-sigma relative uncertainties per source. These are the
# TRUE global sectoral uncertainties (much lower than the inflated cross-country-BTR-median fallback
# above): being <= the mean national uncertainty, they are always reachable by the cross-country
# correlation without capping. Used as the default GLOBAL BACKGROUND (NationalPriorGlobalBackground);
# override per sector with NationalPriorGlobalBackgroundValues. Keys = HEMCO EmisCH4_* suffixes.
SAUNOIS_GLOBAL_BACKGROUND = {
    "Livestock": 0.05, "Rice": 0.22, "Landfills": 0.19, "Wastewater": 0.19,
    "Coal": 0.10, "Gas": 0.15, "Oil": 0.15, "OilGas": 0.15, "OtherAnth": 0.80,
    "Wetlands": 0.28, "Reservoirs": 0.80, "Lakes": 0.80, "InlandWaters": 0.80,
    "Seeps": 0.60, "Termites": 0.60, "BiomassBurn": 0.41,
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
    """Read a per-(country, sector) relative-uncertainty CSV. Columns are flexible so any IMI user's
    file works: the uncertainty column may be 'relative_uncertainty' | 'u' | 'uncertainty'; the country
    identifier may be 'country_id' | 'iso3' | 'country' | 'country_name'. ISO3 codes are matched to the
    bundled shapefile's ISO3 column (set NationalPriorCountryNameColumn: ISO3); country names match its
    NAME column. Sector is 'sector'. Returns rows {country_id, country_name, sector, relative_uncertainty}."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        # Map normalized (stripped, lowercased) header -> original header, so header casing/whitespace
        # (e.g. "ISO3, Sector, U") is tolerated. Row access uses the original field names.
        norm = {}
        for fn in (reader.fieldnames or []):
            norm.setdefault(str(fn).strip().lower(), fn)
        unc_key = next((k for k in ("relative_uncertainty", "u", "uncertainty") if k in norm), None)
        id_keys = [k for k in ("country_id", "iso3", "country", "country_name") if k in norm]
        if "sector" not in norm or unc_key is None:
            raise ValueError(
                f"{path} must have a 'sector' column and an uncertainty column "
                f"(relative_uncertainty | u | uncertainty); found {list(reader.fieldnames or [])}"
            )
        if not id_keys:
            raise ValueError(
                f"{path} must have a country identifier (country_id | iso3 | country | country_name); "
                f"found {list(reader.fieldnames or [])}"
            )
        unc_col, sector_col = norm[unc_key], norm["sector"]
        id_cols = [norm[k] for k in id_keys]
        name_col = norm.get("country", norm.get("country_name"))
        for row in reader:
            ident = next((str(row.get(c, "")).strip() for c in id_cols if str(row.get(c, "")).strip()), "")
            country_name = (str(row.get(name_col, "")).strip() if name_col else "") or ident
            unc = str(row.get(unc_col, "")).strip()
            if not ident or not str(row.get(sector_col, "")).strip() or unc == "":
                continue
            try:
                unc_val = float(unc)
            except ValueError:
                print(f"  {path}: non-numeric uncertainty {unc!r} for {ident}/{row.get(sector_col)}; skipping.")
                continue
            rows.append(
                {
                    "country_id": ident,
                    "country_name": country_name,
                    "sector": str(row[sector_col]).strip(),
                    "relative_uncertainty": unc_val,
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


def indomain_area_fraction(country_shape, lat, lon):
    """Fraction of a country's area inside the inversion domain, via a SINGLE polygon intersection with
    the domain's lon/lat bounding box in an equal-area CRS -- O(1) per country, no per-cell loop. This is
    the in-domain AREA fraction; it equals the in-domain EMISSION fraction the domain-invariant national
    term wants only under a uniform-emission-density assumption -- the best estimate available without
    out-of-domain (global) emissions, which a regional run does not have. Equal-area (EPSG:6933) is used
    because square-degrees over-weight high latitudes by ~1/cos(lat)."""
    import geopandas as gpd
    from shapely.geometry import box

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = float(np.abs(lat[1] - lat[0])) if lat.size > 1 else 0.0
    dlon = float(np.abs(lon[1] - lon[0])) if lon.size > 1 else 0.0
    domain_box = gpd.GeoDataFrame(
        geometry=[box(lon.min() - dlon / 2, lat.min() - dlat / 2,
                      lon.max() + dlon / 2, lat.max() + dlat / 2)],
        crs="EPSG:4326",
    ).to_crs("EPSG:6933")
    cp = country_shape.to_crs("EPSG:6933")
    full = float(cp.area.sum())
    if full <= 0:
        return 1.0
    indomain = float(cp.intersection(domain_box.geometry.iloc[0]).area.sum())
    return min(max(indomain / full, 0.0), 1.0)


def get_country_fraction_mask(country, lat, lon, shapes, name_column, area_weighting=True):
    """Binary in-domain mask for a country (regionmask cell-center assignment) plus the in-domain area
    fraction f_C for the domain-invariant national term. area_weighting=False returns f_C=1 (national
    term treated as fully in-domain); True computes f_C via indomain_area_fraction (a cheap polygon clip)."""
    import regionmask
    import warnings

    country_shape = shapes[shapes[name_column] == country]
    if country_shape.empty:
        raise ValueError(f"Country {country!r} not found in shapefile column {name_column!r}")

    with warnings.catch_warnings():
        # a country outside the inversion domain legitimately covers no gridpoints (e.g. a global BTR
        # CSV run over a regional domain) -- silence regionmask's expected all-NaN warning.
        warnings.filterwarnings("ignore", message="No gridpoint belongs to any region")
        mask = np.array(regionmask.mask_geopandas(country_shape, lon, lat) + 1)
    mask[np.isnan(mask)] = 0
    mask[mask > 0] = 1
    mask = mask.astype(float)

    if not area_weighting:
        return mask, 1.0
    return mask, indomain_area_fraction(country_shape, lat, lon)


def default_country_shapefile():
    """Path to the global admin-0 country shapefile bundled with the IMI (resources/countries/, with
    NAME + ISO3 columns). Resolved from IMI_COUNTRY_SHAPEFILE or relative to this file's repo location;
    returns None if not found (e.g. the script was copied to a run dir without resources/)."""
    env = os.environ.get("IMI_COUNTRY_SHAPEFILE")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "resources", "countries", "imi_country_boundaries.shp"),   # copied beside a run-dir script
        os.path.normpath(os.path.join(here, "..", "..", "resources", "countries", "imi_country_boundaries.shp")),  # in-repo
    ):
        if os.path.exists(cand):
            return cand
    return None


def domain_invariant_enabled(config):
    """Whether the domain-invariant national term is on -- scale the national variance by 1/f_C^2 for a
    country only partly in the domain. Defaults ON for regional inversions (config isRegional): it is the
    faithful treatment of whole-country BTR uncertainties and a no-op for fully-in-domain countries
    (f_C=1). An explicit NationalPriorDomainInvariant overrides the default."""
    val = config.get("NationalPriorDomainInvariant", None)
    if val is None:
        return str(config.get("isRegional", True)).strip().lower() in ("true", "1", "yes")
    return str(val).strip().lower() in ("true", "1", "yes")


def build_country_mask_from_shapes(uncertainty_rows, prior, config):
    # Default to the bundled global shapefile so any IMI user gets per-country masks with no setup.
    shapefile = config.get("NationalPriorCountryShapefile") or default_country_shapefile()
    name_column = config.get("NationalPriorCountryNameColumn", "NAME")
    # Compute the in-domain area fraction f_C (a cheap polygon clip per country) when the domain-invariant
    # national term is on (default on for regional inversions); f_C=1 otherwise.
    domain_invariant = domain_invariant_enabled(config)
    area_weighting = bool(config.get("NationalPriorCountryMaskAreaWeighting", domain_invariant))
    if not shapefile:
        return None, None

    shapes = load_country_shapes(shapefile, name_column)
    mask = xr.DataArray(
        np.zeros((prior.sizes["lat"], prior.sizes["lon"]), dtype=float),
        coords={"lat": prior.lat.values, "lon": prior.lon.values},
        dims=("lat", "lon"),
    )
    country_lookup = {}
    country_fraction = {}                                      # {country_id(str): in-domain area fraction}
    best_fraction = np.zeros((prior.sizes["lat"], prior.sizes["lon"]), dtype=float)  # best per-cell coverage so far
    next_id = 1
    for row in uncertainty_rows:
        country_name = row.get("country_name") or row["country_id"]
        if country_name in country_lookup:
            row["country_id"] = str(country_lookup[country_name])
            continue
        country_id = int(row["country_id"]) if str(row["country_id"]).isdigit() else next_id
        try:
            fraction, f_c = get_country_fraction_mask(
                country_name,
                prior.lat.values,
                prior.lon.values,
                shapes,
                name_column,
                area_weighting=area_weighting,
            )
        except ValueError as exc:
            print(f"  Country mask: {exc}; skipping {country_name!r}.")
            continue
        if not np.any(fraction > 0):
            continue                                          # country has no cells in the domain -> ignore it
        next_id = max(next_id, country_id + 1)
        current = mask.values
        replace = fraction > best_fraction                    # assign each cell to the country covering the MOST of it
        current[replace] = country_id
        best_fraction[replace] = fraction[replace]
        mask.values[:] = current
        country_lookup[country_name] = country_id
        country_fraction[str(country_id)] = f_c               # str key: two_component_absolute looks up by str id
        row["country_id"] = str(country_id)

    # Warn if a matched country extends beyond the inversion domain while domain-invariance is OFF:
    # its national uncertainty is then applied as if the whole country were in-domain, which over-
    # constrains the in-domain part (the Nigeria-style partially-in-domain case). A cheap geometry-
    # bbox vs grid-bounds check -- no extra area computation.
    domain_invariant_on = domain_invariant_enabled(config)
    if not domain_invariant_on and country_lookup:
        lon = prior.lon.values
        lat = prior.lat.values
        dlon = float(abs(lon[1] - lon[0])) if lon.size > 1 else 0.0
        dlat = float(abs(lat[1] - lat[0])) if lat.size > 1 else 0.0
        lon_lo, lon_hi = float(lon.min()) - dlon / 2, float(lon.max()) + dlon / 2
        lat_lo, lat_hi = float(lat.min()) - dlat / 2, float(lat.max()) + dlat / 2
        partial = []
        for country_name in country_lookup:
            geom = shapes[shapes[name_column] == country_name]
            if geom.empty:
                continue
            minx, miny, maxx, maxy = geom.total_bounds
            if (minx < lon_lo - 1e-6 or maxx > lon_hi + 1e-6
                    or miny < lat_lo - 1e-6 or maxy > lat_hi + 1e-6):
                partial.append(str(country_name))
        if partial:
            shown = ", ".join(partial[:6]) + (" ..." if len(partial) > 6 else "")
            print(f"  NOTE: {len(partial)} matched country(ies) extend beyond the inversion domain "
                  f"({shown}); with NationalPriorDomainInvariant off their national uncertainty is "
                  f"applied as if fully in-domain. Set NationalPriorDomainInvariant: true to scale the "
                  f"national term by the in-domain fraction (1/f_C^2).")
    return mask, country_fraction


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
    grid_national_ratio=2.5, min_uncertainty=0.30, global_background=None,
    country_fraction=None,
):
    """Prior error covariance in ABSOLUTE emission^2 units, as a sum of independent error components
    (a variance-components / random-effects model). For two cells i, j of one sector within a country:

        NATIONAL  snat^2  * E_i E_j              (fully correlated across the country)  <- BTR national error
        LOCAL     snat^2 (ratio_i^2-1) * E_i^2   on the diagonal only                   <- grid-scale wiggle
        FLOOR     g_floor^2 * E_i E_j            (only where u_BTR < g_s)               <- Saunois global floor

    ratio_i = sqrt(1 + (R01^2 - 1)/n_eff_i) is the grid:national error ratio (n_eff_i = effective number
    of independent native cells in element i). The national+local part uses the FULL u_BTR (no peel):
        snat = u_BTR / sqrt(1 + Q),   Q = sum_i (ratio_i^2 - 1) E_i^2 / E_c^2,
    so NATIONAL + LOCAL give a national aggregate of exactly u_BTR. The Saunois global value
    g_s = global_background[sector] is a FLOOR: it ADDS a within-country rank-1 with amplitude
    g_floor = sqrt(max(g_s^2 - u_BTR^2, 0)) ONLY where u_BTR < g_s, raising the national aggregate to
    max(u_BTR, g_s) and never reducing the national/local grid structure (where u_BTR >= g_s nothing is
    added, so no double-counting). global_background=None (or every g_s <= u_BTR) recovers the plain
    two-component. Sectors are independent (summed). Returns (Sa_abs, diagnostics).

    DOMAIN-INVARIANCE (country_fraction): the NATIONAL rank-1 term encodes a WHOLE-COUNTRY-total
    constraint (snat set so the in-domain aggregate == u_BTR), valid only if the country is fully in
    the domain. For a country with only fraction f_C of its emissions in-domain (the rest held at
    prior), a coherent in-domain shift moves the national total by f_C, so to keep the implied national
    uncertainty == u_BTR the in-domain national variance is scaled by 1/f_C^2. country_fraction =
    {country_id: f_C in (0,1]}; None or a missing key -> f_C=1 (unchanged, the current behaviour). The
    LOCAL diagonal term is a per-cell allocation error and is NOT scaled. f_C -> 0 makes the national
    term vanish (a sliver cannot be pinned to a national total)."""
    bg = global_background or {}
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
        g = float(bg.get(sector, 0.0))                         # Saunois global background FLOOR for this sector
        n_eff = np.array([
            (neff_sum[(sector, int(p))] ** 2 / neff_sumsq[(sector, int(p))])
            if neff_sumsq.get((sector, int(p)), 0.0) > 0 else 1.0
            for p in positions
        ])
        n_eff = np.clip(n_eff, 1.0, None)
        r2 = (grid_national_ratio ** 2 - 1.0) / n_eff          # ratio_i^2 - 1 (local excess)
        Q = float(np.sum(r2 * emis ** 2)) / (total * total)
        snat = u / np.sqrt(1.0 + Q)                            # NATIONAL amplitude from the FULL u_BTR (no peel)
        fC = float(country_fraction.get(country_id, 1.0)) if country_fraction else 1.0  # in-domain emission fraction
        fC = min(max(fC, 1.0e-3), 1.0)                         # clip (1e-3 -> national term ~vanishes for a sliver)
        Sa_abs[np.ix_(positions, positions)] += (snat * snat / (fC * fC)) * np.outer(emis, emis)  # NATIONAL rank-1 /f_C^2 (domain-invariant)
        Sa_abs[positions, positions] += (snat * snat) * r2 * emis ** 2                # LOCAL diagonal (per-cell allocation; NOT f-scaled)
        # Saunois FLOOR (global background, applied to ALL sectors): g only ADDS where the BTR aggregate is
        # short of g (u < g); it NEVER peels down the national/local grid structure.  Where u >= g nothing is
        # added -- the BTR value already covers the Saunois background, so there is no double-counting.  The
        # national aggregate therefore rises to max(u, g) and the grid-scale structure is always preserved.
        g_floor2 = max(g * g - u * u, 0.0)                     # (g^2 - u^2) where u < g, else 0
        if g_floor2 > 0.0:
            Sa_abs[np.ix_(positions, positions)] += g_floor2 * np.outer(emis, emis)   # within-country systematic floor -> aggregate = g
        # national aggregate^2 = NATIONAL snat^2 E_c^2 + LOCAL snat^2 sum(r2 E^2) + FLOOR g_floor2 E_c^2
        #                      = (u^2 + max(g^2 - u^2, 0)) E_c^2 = max(u, g)^2 E_c^2
        achieved = np.sqrt((snat * snat) * (total * total + float(np.sum(r2 * emis ** 2)))
                           + g_floor2 * total * total) / total
        diagnostics.append({
            "country_id": country_id, "sector": sector, "status": "ok",
            "relative_uncertainty": u, "global_background": g,
            "saunois_floor_added": float(np.sqrt(g_floor2)),    # extra systematic added (0 unless u < g)
            "achieved_relative_uncertainty": float(achieved),   # == max(u_BTR, g_s)
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
    prior_sigma, grid_national_ratio=2.5, min_uncertainty=0.30, global_background=None,
    country_fraction=None,
):
    """Exact three-component prior error covariance (returns correlation C, per-element sigma,
    diagnostics).  Thin wrapper: assemble the absolute covariance (two_component_absolute, which adds
    the Saunois global value as a within-country FLOOR raising each national aggregate to max(u_BTR, g_s)
    -- it only adds where u_BTR < g_s), then decompose it exactly into (C, sigma)
    (decompose_relative_covariance).  Fits the standard (correlation, sigma_scale) output contract with
    no approximation. global_background=None recovers the plain two-component."""
    n = len(totals_by_element)
    Sa_abs, diagnostics = two_component_absolute(
        rows, neff_sum, neff_sumsq, uncertainty_rows, n,
        grid_national_ratio, min_uncertainty, global_background,
        country_fraction=country_fraction,
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
    # Fall back to the bundled global shapefile so any IMI user gets per-country masks with no setup.
    country_shapefile = config.get("NationalPriorCountryShapefile") or default_country_shapefile()
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

    country_fraction = None
    if using_global_defaults:
        print("NationalPriorUncertaintyFile not set; using Saunois et al. global sectoral defaults.")
        uncertainty_rows = build_saunois_default_table()
        country_mask = build_all_ones_country_mask(prior)
    else:
        uncertainty_rows = read_uncertainty_table(uncertainty_path)
        if country_mask_path:
            country_mask = load_country_mask(country_mask_path, country_mask_var)
        else:
            country_mask, country_fraction = build_country_mask_from_shapes(uncertainty_rows, prior, config)
    # DOMAIN-INVARIANT national term: scale each country's national rank-1 by 1/f_C^2 so a partially-in-
    # domain country is not pinned to its whole-country total. Needs the shapefile mask (country_fraction);
    # defaults ON for regional inversions (a no-op where f_C=1), OFF for global.
    if not domain_invariant_enabled(config):
        country_fraction = None
    elif country_fraction is not None:
        print(f"Domain-invariant national term ON: f_C for {len(country_fraction)} countries "
              f"(min {min(country_fraction.values()):.2f}, max {max(country_fraction.values()):.2f}); "
              f"national var scaled by 1/f_C^2.")

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
        global_background = None
        if str(config.get("NationalPriorGlobalBackground", True)).strip().lower() in ("true", "1", "yes"):   # default ON
            # Saunois GLOBAL BACKGROUND applied as a FLOOR: a within-country rank-1 (magnitude g_s) added
            # ONLY where u_BTR < g_s, raising that national aggregate to max(u_BTR, g_s) without touching
            # the national/local grid structure (where u_BTR >= g_s nothing is added -- no double-count).
            # Defaults to SAUNOIS_GLOBAL_BACKGROUND; override per sector with NationalPriorGlobalBackgroundValues.
            global_background = dict(SAUNOIS_GLOBAL_BACKGROUND)   # genuine Saunois global values
            global_background.update(config.get("NationalPriorGlobalBackgroundValues", {}) or {})
            print(f"Global background (Saunois) ON: {len(global_background)} sectors "
                  f"(e.g. Wetlands={global_background.get('Wetlands')}, Coal={global_background.get('Coal')}); "
                  f"floor -> national aggregates rise to max(u_BTR, g_s) only where u_BTR < g_s.")
        covariance, sigma_vector, diagnostics = build_two_component_covariance(
            rows, neff_sum, neff_sumsq, uncertainty_rows, totals_by_element,
            prior_sigma, grid_national_ratio, min_uncertainty, global_background,
            country_fraction=country_fraction,
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
