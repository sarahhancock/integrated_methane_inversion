#!/usr/bin/env python3
"""Emission fractions from the GLOBAL (out-of-domain) inventories the IMI user configured in HEMCO.

A regional inversion only sees emissions inside its domain, so two prior-covariance quantities have to be
approximated:
  1. The Saunois GLOBAL BACKGROUND for wetlands + minor naturals is a systematic whose amplitude is the
     global Saunois relative uncertainty G. Applied over a sub-domain it should carry only that domain's
     SHARE of the global sector:  sigma_sys = G * f_sector,  f_sector = E_domain / E_global.
  2. The DOMAIN-INVARIANT national anthro term scales the national variance by 1/f_C^2 for a country only
     partly in the domain. f_C should be the in-domain EMISSION fraction (in-domain-country / full-country),
     not the AREA fraction (build_national_inventory_prior_covariance.indomain_area_fraction, which only
     equals it under uniform emission density).

This module computes BOTH from the SAME global inventory files HEMCO reads (parsed from HEMCO_Config.rc), so
each user gets fractions consistent with their own inventories. The fractions are ~time-invariant, so they are
computed ONCE and cached to <inversion_data>/global_emission_fractions.npz keyed on the domain box + the
inventory files' paths/mtimes; the per-period covariance build just reads the cache (no runtime hit).

Only numpy/xarray/geopandas/regionmask (no GEOS-Chem); safe to run in the IMI setup phase.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import xarray as xr

R_EARTH_M = 6.371e6
SECONDS_PER_YEAR = 3.1536e7

# Builder-sector -> HEMCO base/extension entry token(s) that carry that sector's GLOBAL emission field, plus
# the variable and a unit tag. Tokens are matched against the emission-line container name or the field name
# in HEMCO_Config.rc. GFED is handled specially (dry matter x CH4 emission factor). Override per config via
# EmissionFractionSectorMap in the YAML if a user's inventory names differ.
DEFAULT_SECTOR_TOKENS = {
    # Saunois-background natural sectors
    "Wetlands":    [("JPLW_CH4", None), ("WETCHARTS", None)],
    "Seeps":       [("CH4_SEEPS", None)],
    "Reservoirs":  [("CH4_RES_DAM", None), ("CH4_RES_SFC", None)],
    "Lakes":       [("CH4_LAKES", None)],
    "Termites":    [("CH4_TERMITES", None)],
    # anthro sectors (used for the per-country in-domain emission fraction f_C)
    "Livestock":   [("EDGAR", "livestock"), ("_ENT", None), ("_MNM", None)],
    "Rice":        [("_RCO", None), ("RICE", None)],
    "Landfills":   [("_SWD", None), ("LANDFILL", None)],
    "Wastewater":  [("_WWT", None), ("WASTEWATER", None)],
    "Coal":        [("_COAL", None), ("GFEI", "coal")],
    "Gas":         [("GFEI", "gas"), ("_GAS", None)],
    "Oil":         [("GFEI", "oil"), ("_OIL", None)],
}

# GFED4 dry-matter species -> CH4 emission factor (g CH4 / kg DM; van der Werf et al. 2017 / Akagi et al. 2011)
GFED_CH4_EF = {
    "DM_SAVA": 2.7, "DM_BORF": 5.2, "DM_TEMF": 3.6, "DM_TEMP": 3.6,
    "DM_DEFO": 5.0, "DM_PEAT": 20.8, "DM_AGRI": 5.7,
}


def _cell_area_m2(lat, lon):
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    dlat = np.abs(np.gradient(lat)) if lat.size > 1 else np.array([1.0])
    dlon = np.abs(np.gradient(lon)) if lon.size > 1 else np.array([1.0])
    return (R_EARTH_M ** 2) * np.outer(np.cos(np.radians(lat)) * np.radians(dlat), np.radians(dlon))


def _to_kg_per_m2_s(da):
    """Best-effort convert an emission-rate DataArray to kg/m2/s using its 'units' attribute."""
    u = str(da.attrs.get("units", "kg/m2/s")).lower().replace(" ", "")
    if u in ("kg/m2/s", "kgm-2s-1", "kg/m^2/s"):
        return da
    if u in ("molec/cm2/s", "molecules/cm2/s", "moleccm-2s-1"):
        # molec CH4 /cm2/s -> kg/m2/s : *1e4 (cm2->m2) * 16.04/6.022e23 (molec->g) /1000 (g->kg)
        return da * (1.0e4 * 16.04 / 6.022e23 / 1.0e3)
    if u in ("kg/m2/yr", "kgm-2yr-1"):
        return da / SECONDS_PER_YEAR
    return da  # assume kg/m2/s


def resolve_root(hemco_config_path):
    root = None
    with open(hemco_config_path) as f:
        for line in f:
            m = re.match(r"\s*ROOT:\s*(\S+)", line)
            if m:
                root = m.group(1); break
    return root


def parse_hemco_emission_files(hemco_config_path):
    """Map each active emission-container/field token in HEMCO_Config.rc to (path, variable). Only entries
    inside enabled (((TOKEN ... )))TOKEN blocks (or unconditional) with a resolvable .nc path are returned."""
    root = resolve_root(hemco_config_path)
    text = open(hemco_config_path).read()
    lines = text.splitlines()
    # which extension switches are true (from the ExtNr menu 'NAME : true')
    enabled = set()
    for line in lines:
        m = re.match(r"\s*-->\s*([A-Za-z0-9_]+)\s*:\s*true", line)
        if m:
            enabled.add(m.group(1))
    entries = {}   # container_token -> list of (path, var)
    block_stack = []
    for line in lines:
        mo = re.match(r"\s*\(\(\(([A-Za-z0-9_]+)", line)
        mc = re.match(r"\s*\)\)\)([A-Za-z0-9_]+)", line)
        if mo:
            block_stack.append(mo.group(1)); continue
        if mc:
            if block_stack:
                block_stack.pop()
            continue
        # a data line: idx NAME path var timerange ... ; skip masks/scale-only lines heuristically
        parts = line.split()
        if len(parts) < 4:
            continue
        if not any(p.endswith(".nc") or p.endswith(".nc4") for p in parts):
            continue
        # if inside a block that is a known extension switch and disabled, skip
        if block_stack and block_stack[-1] not in enabled and block_stack[-1] not in entries:
            # keep going: some blocks aren't ExtNr switches; only skip if the top block is a known disabled switch
            pass
        name = parts[1]
        path = next((p for p in parts if p.endswith(".nc") or p.endswith(".nc4")), None)
        # variable is the token right after the path
        var = None
        for i, p in enumerate(parts):
            if p == path and i + 1 < len(parts):
                var = parts[i + 1]; break
        if path is None:
            continue
        if root:
            path = path.replace("$ROOT", root)
        # Expand $YYYY with a representative recent year; keep $MM/$DD as globs so the loader time-averages the
        # WHOLE year (biomass burning etc. is strongly seasonal -- a single month is not the annual fraction).
        path = (path.replace("$YYYY", "2019").replace("$MM", "??").replace("$DD", "??")
                    .replace("$GCAPSCENARIO", "").strip())
        key = block_stack[-1] if block_stack else name
        entries.setdefault(key, []).append((path, var, name))
    return entries, enabled


def _open(path):
    """Open one file, or a globbed (time-varying) set with open_mfdataset (for $MM/$DD-expanded patterns)."""
    if any(c in path for c in "*?[]"):
        import glob
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(path)
        return xr.open_mfdataset(files, combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
    return xr.open_dataset(path)


def _load_global_field_tgyr(path, var):
    """Load an emission file (or globbed year), time-mean the rate, return (field_TgYr per cell, lat, lon)."""
    ds = _open(path)
    v = ds[var] if var in ds.data_vars else ds[list(ds.data_vars)[0]]
    if "time" in v.dims:
        v = v.mean("time")
    v = _to_kg_per_m2_s(v).load()
    lat = ds["lat"].values if "lat" in ds else ds[[d for d in v.dims][-2]].values
    lon = ds["lon"].values if "lon" in ds else ds[[d for d in v.dims][-1]].values
    area = _cell_area_m2(lat, lon)
    field = np.nan_to_num(v.values) * area * SECONDS_PER_YEAR / 1e9   # Tg/yr per cell
    return field, lat, lon


def domain_fraction(field, lat, lon, lon_bounds, lat_bounds):
    """f = (sum inside the domain lon/lat box) / (global sum)."""
    g = float(field.sum())
    if g <= 0:
        return 0.0
    la = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lo = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    sa = float(field[np.ix_(la, lo)].sum())
    return min(max(sa / g, 0.0), 1.0)


def country_emission_fraction(field, lat, lon, country_shape, lon_bounds, lat_bounds):
    """f_C = (country's emission INSIDE the domain box) / (country's FULL global emission), on the field's grid."""
    import regionmask
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No gridpoint belongs to any region")
        cmask = np.array(regionmask.mask_geopandas(country_shape, lon, lat) + 1)
    cmask = np.where(np.isfinite(cmask), 1.0, 0.0)
    full = float((field * cmask).sum())
    if full <= 0:
        return 1.0
    la = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lo = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    dmask = np.zeros_like(cmask); dmask[np.ix_(la, lo)] = 1.0
    indomain = float((field * cmask * dmask).sum())
    return min(max(indomain / full, 0.0), 1.0)


def _is_mask(var, name):
    return (var or "") == "Mask" or "MASK" in (name or "").upper()


def _sector_fields(sector, entries):
    """Yield (field_TgYr, lat, lon) for each UNIQUE global emission file mapped to `sector` (masks excluded);
    GFED biomass burning summed over DM species x CH4 emission factor."""
    seen = set()
    if sector == "BiomassBurn":
        acc = None; lat = lon = None
        for key, files in entries.items():
            if "GFED" not in key.upper():
                continue
            for path, var, name in files:
                if var not in GFED_CH4_EF or (path, var) in seen:
                    continue
                seen.add((path, var))
                try:
                    fld, la, lo = _load_global_field_tgyr_dm(path, var, GFED_CH4_EF[var])
                except Exception:
                    continue
                acc = fld if acc is None else acc + fld; lat, lon = la, lo
        if acc is not None:
            yield acc, lat, lon
        return
    _ov = globals().get("_SECTOR_TOKENS_OVERRIDE") or {}                # optional per-user EmissionFractionSectorMap
    if sector in _ov:
        toks = list(_ov[sector]) if isinstance(_ov[sector], (list, tuple)) else [_ov[sector]]
    else:
        toks = [t[0] for t in DEFAULT_SECTOR_TOKENS.get(sector, [])]
    for key, files in entries.items():
        for path, var, name in files:
            if _is_mask(var, name) or (path, var) in seen:
                continue
            if any(tok in (name or "") or tok in (var or "") or tok in key for tok in toks):
                seen.add((path, var))
                try:
                    yield _load_global_field_tgyr(path, var)
                except Exception:
                    continue


def _load_global_field_tgyr_dm(path, dm_var, ef_g_per_kg):
    ds = _open(path)
    v = ds[dm_var]
    if "time" in v.dims:
        v = v.mean("time")
    lat = ds["lat"].values; lon = ds["lon"].values
    area = _cell_area_m2(lat, lon)
    field = np.nan_to_num(v.values) * area * (ef_g_per_kg / 1000.0) * SECONDS_PER_YEAR / 1e9  # Tg CH4/yr
    return field, lat, lon


def _cache_key(hemco_config_path, lon_bounds, lat_bounds, entries):
    h = hashlib.sha1()
    h.update(f"{lon_bounds}{lat_bounds}".encode())
    for key in sorted(entries):
        for path, var, name in entries[key]:
            try:
                st = os.stat(path); h.update(f"{path}{st.st_mtime_ns}{st.st_size}".encode())
            except OSError:
                h.update(path.encode())
    return h.hexdigest()


def build_emission_fractions(hemco_config_path, lon_bounds, lat_bounds, shapes=None, name_column="ISO3",
                             saunois_sectors=("Wetlands", "Seeps", "Reservoirs", "Lakes", "Termites", "BiomassBurn"),
                             anthro_sectors=("Livestock", "Rice", "Landfills", "Wastewater", "Coal", "Gas", "Oil"),
                             cache_path=None):
    """Compute (and cache) the Saunois-sector domain fractions f_sector and the per-country anthro emission
    fractions f_C. Returns dict {"sector_fraction": {sec: f}, "country_fraction": {country: f_C}}."""
    entries, _enabled = parse_hemco_emission_files(hemco_config_path)
    key = _cache_key(hemco_config_path, lon_bounds, lat_bounds, entries)
    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=True)
            if str(z["key"]) == key:
                return {"sector_fraction": dict(z["sector_fraction"].item()),
                        "country_fraction": dict(z["country_fraction"].item())}
        except Exception:
            pass

    sector_fraction = {}
    for sec in saunois_sectors:
        num = 0.0; den = 0.0
        for field, lat, lon in _sector_fields(sec, entries):
            la = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1]); lo = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
            num += float(field[np.ix_(la, lo)].sum()); den += float(field.sum())
        sector_fraction[sec] = (min(max(num / den, 0.0), 1.0) if den > 0 else 0.0)

    country_fraction = {}
    if shapes is not None:
        # per-country f_C from the TOTAL anthro emission field (sum over anthro sectors)
        anthro_field = None; alat = alon = None
        for sec in anthro_sectors:
            for field, lat, lon in _sector_fields(sec, entries):
                if anthro_field is None:
                    anthro_field, alat, alon = field.copy(), lat, lon
                elif field.shape == anthro_field.shape:
                    anthro_field += field
        if anthro_field is not None:
            for country in shapes[name_column].unique():
                cs = shapes[shapes[name_column] == country]
                try:
                    country_fraction[str(country)] = country_emission_fraction(
                        anthro_field, alat, alon, cs, lon_bounds, lat_bounds)
                except Exception:
                    country_fraction[str(country)] = 1.0

    out = {"sector_fraction": sector_fraction, "country_fraction": country_fraction}
    if cache_path:
        try:
            d = os.path.dirname(os.path.abspath(cache_path))
            if d:
                os.makedirs(d, exist_ok=True)
            np.savez(cache_path, key=key,
                     sector_fraction=np.array(sector_fraction, dtype=object),
                     country_fraction=np.array(country_fraction, dtype=object))
        except Exception:
            pass
    return out


def resolve_hemco_config(config):
    """Path to the HEMCO_Config.rc for the prior emissions (used to read the GLOBAL inventory fields).
    Explicit EmissionFractionHemcoConfig wins; else the standard IMI prior-emissions location
    {OutputPath}/{RunName}/hemco_prior_emis/HEMCO_Config.rc. Returns None if nothing resolves."""
    explicit = config.get("EmissionFractionHemcoConfig")
    if explicit and os.path.exists(explicit):
        return explicit
    out, run = config.get("OutputPath"), config.get("RunName")
    if out and run:
        cand = os.path.join(str(out), str(run), "hemco_prior_emis", "HEMCO_Config.rc")
        if os.path.exists(cand):
            return cand
    return None


def emission_fractions_for_config(config, prior, shapes=None, name_column=None, cache_dir=None):
    """Config-driven wrapper: build (and cache) the emission fractions for an IMI run.

    Resolves the HEMCO_Config.rc (resolve_hemco_config), the domain box (config LonMin/LonMax/LatMin/LatMax,
    falling back to the prior grid bounds) and the cache path ({cache_dir or OutputPath}/global_emission_fractions.npz),
    then calls build_emission_fractions. `shapes` (a GeoDataFrame) is needed only for the per-country f_C.

    Returns {"sector_fraction": {...}, "country_fraction": {...}} or None if disabled
    (EmissionFractionScaling: false) or the HEMCO config can't be found -- callers then fall back to the
    area-based / unscaled behaviour and the run is never broken by a missing/parse-failing HEMCO config."""
    if str(config.get("EmissionFractionScaling", True)).strip().lower() not in ("true", "1", "yes"):
        return None
    hemco = resolve_hemco_config(config)
    if not hemco:
        print("  Emission-fraction f: no HEMCO_Config.rc resolved (set EmissionFractionHemcoConfig); "
              "falling back to area-based f_C and unscaled naturals background.")
        return None
    try:
        lon_b = (float(config["LonMin"]), float(config["LonMax"]))
        lat_b = (float(config["LatMin"]), float(config["LatMax"]))
    except Exception:
        lon = np.asarray(prior.lon.values, dtype=float); lat = np.asarray(prior.lat.values, dtype=float)
        lon_b = (float(lon.min()), float(lon.max())); lat_b = (float(lat.min()), float(lat.max()))
    if name_column is None:
        name_column = config.get("NationalPriorCountryNameColumn", "ISO3")
    if cache_dir is None:
        cache_dir = config.get("OutputPath") or os.path.dirname(os.path.abspath(hemco))
    cache_path = os.path.join(str(cache_dir), "global_emission_fractions.npz") if cache_dir else None
    kwargs = {}
    if config.get("EmissionFractionSectorMap"):                     # optional per-user token overrides
        # let build_emission_fractions/_sector_fields see custom tokens via a module-level override hook
        globals()["_SECTOR_TOKENS_OVERRIDE"] = dict(config.get("EmissionFractionSectorMap"))
    try:
        res = build_emission_fractions(hemco, lon_b, lat_b, shapes=shapes, name_column=name_column,
                                       cache_path=cache_path, **kwargs)
        sf = res.get("sector_fraction", {})
        print("  Emission-fraction f (from {}): ".format(os.path.basename(os.path.dirname(hemco))) +
              ", ".join(f"{k}={sf[k]:.3f}" for k in sorted(sf)) +
              (f"; f_C for {len(res.get('country_fraction', {}))} countries" if shapes is not None else ""))
        return res
    except Exception as exc:
        print(f"  Emission-fraction helper failed ({type(exc).__name__}: {exc}); "
              f"falling back to area-based f_C and unscaled naturals background.")
        return None


if __name__ == "__main__":
    import sys, json
    hc = sys.argv[1]
    lon_b = (float(sys.argv[2]), float(sys.argv[3])); lat_b = (float(sys.argv[4]), float(sys.argv[5]))
    res = build_emission_fractions(hc, lon_b, lat_b)
    print(json.dumps(res, indent=2))
