import datetime
import glob
import os
import re
import warnings

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from src.inversion_scripts.operators.satellite_operator import (
    average_satellite_observations,
)
from src.inversion_scripts.utils import read_and_filter_satellite, get_strdate


def _atomic_to_netcdf(ds, path, encoding):
    """Write a cache file ATOMICALLY: temp file in the same dir, then os.replace().

    Without this, a job killed mid-write (SLURM timeout / scancel) leaves a
    half-written file at the FINAL path. Every later attempt then hits the
    `if not os.path.isfile(...)` guard, SKIPS it, declares the build complete, and the
    raw GEOS-Chem output gets pruned -- leaving a corrupt cache as the only copy and
    forcing a full ~9h simulation re-run. os.replace() is atomic on POSIX, so a killed
    job leaves only a stray .tmp that the next run rewrites.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        ds.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def normalize_satellite_product(satellite_product):
    if satellite_product in ("True", "true", True):
        return "BlendedTROPOMI"
    if satellite_product in ("False", "false", False):
        return "TROPOMI"
    return satellite_product


def source_gc_files(gc_source_path, day):
    species_standard = f"{gc_source_path}/GEOSChem.SpeciesConc.{day}_0000z.nc4"
    pedge_standard = f"{gc_source_path}/GEOSChem.StateMetLevEdge.{day}_0000z.nc4"
    # Perturbation runs that start from a 1 ppb restart only write the _0005z
    # split file (hours 1-23) for the first day, with no separate hour-0 _0000z
    # snapshot. Treat the presence of either file as a valid "standard" source.
    species_split = species_standard.replace("_0000z.nc4", "_0005z.nc4")
    if (
        os.path.exists(species_standard)
        or os.path.exists(species_split)
        or not os.path.isdir(gc_source_path)
    ):
        return (
            "standard",
            species_standard,
            pedge_standard,
        )

    satdiagn_matches = sorted(
        glob.glob(f"{gc_source_path}/{day}_*.nc4")
        + glob.glob(f"{gc_source_path}/*SatDiagn*{day}*.nc4")
    )
    satdiagn_species = None
    satdiagn_pedge = None
    for path in satdiagn_matches:
        try:
            with xr.open_dataset(path) as ds:
                if satdiagn_species is None and any(
                    var.startswith("SatDiagnConc_") for var in ds.data_vars
                ):
                    satdiagn_species = path
                if satdiagn_pedge is None and "SatDiagnPEDGE" in ds.data_vars:
                    satdiagn_pedge = path
        except OSError:
            continue
        if satdiagn_species is not None and satdiagn_pedge is not None:
            break
    if satdiagn_species is not None and satdiagn_pedge is not None:
        return ("satdiagn", satdiagn_species, satdiagn_pedge)

    return (
        "standard",
        species_standard,
        pedge_standard,
    )


def load_gc_hourly_dataset(first_path):
    # GEOS-Chem's first-day output is split: the _0000z file holds only hour 0
    # and the _0005z file holds hours 1-23. Perturbation runs that start from a
    # 1 ppb restart only produce the _0005z file (no hour-0 snapshot), so load
    # whichever of the two actually exist instead of assuming _0000z is present.
    split_path = first_path.replace("_0000z.nc4", "_0005z.nc4")
    candidate_paths = [first_path]
    if split_path != first_path:
        candidate_paths.append(split_path)
    paths = [p for p in candidate_paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(
            f"No GEOS-Chem hourly file found for {first_path} (or its _0005z split)"
        )
    datasets = [xr.load_dataset(path) for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time").sortby("time")


def _select_satdiagn_overpass(ds, hour):
    if "time" not in ds.dims:
        return ds.expand_dims(time=[np.datetime64("NaT")])
    if ds.sizes.get("time", 0) == 1:
        return ds

    times = ds["time"].values
    if np.issubdtype(times.dtype, np.datetime64):
        hours = np.array([int(str(t).split("T")[1][:2]) for t in times])
        idx = int(np.argmin(np.abs(hours - hour)))
    else:
        idx = 0
    return ds.isel(time=slice(idx, idx + 1))


def _merge_hourly_targets(existing, new_i, new_j):
    pairs = np.column_stack((new_i, new_j)).astype(np.int32, copy=False)
    if existing is not None:
        prev = np.column_stack((existing["iGC"], existing["jGC"])).astype(
            np.int32, copy=False
        )
        pairs = np.vstack((prev, pairs))
    if pairs.size == 0:
        return {"iGC": np.empty(0, dtype=np.int32), "jGC": np.empty(0, dtype=np.int32)}
    pairs = np.unique(pairs, axis=0)
    return {
        "iGC": pairs[:, 0].astype(np.int32, copy=False),
        "jGC": pairs[:, 1].astype(np.int32, copy=False),
    }


def _matching_satellite_files(startday, endday, satellite_cache):
    start = np.datetime64(datetime.datetime.strptime(startday, "%Y%m%d"))
    end = np.datetime64(
        datetime.datetime.strptime(endday, "%Y%m%d") - datetime.timedelta(days=1)
    )

    files = []
    for filename in sorted(glob.glob(f"{satellite_cache}/*.nc")):
        shortname = re.split(r"\/", filename)[-1]
        shortname = re.split(r"\.", shortname)[0]
        strdate = re.split(r"\.|_+|T", shortname)[4]
        file_date = datetime.datetime.strptime(strdate, "%Y%m%d")
        if start <= np.datetime64(file_date) <= end:
            files.append(filename)
    return files


def build_hourly_obs_targets(
    startday,
    endday,
    satellite_cache,
    satellite_product,
    species,
    lon_bounds,
    lat_bounds,
    use_water_obs,
    gc_source_path,
    use_gchp=False,
):
    """
    Build a mapping from YYYYMMDD_HH timestamps to the GEOS-Chem columns that
    are actually touched by filtered/averaged satellite super-observations.
    """
    if use_gchp:
        return {}

    product = normalize_satellite_product(satellite_product)
    start = np.datetime64(datetime.datetime.strptime(startday, "%Y%m%d"))
    end = np.datetime64(
        datetime.datetime.strptime(endday, "%Y%m%d") - datetime.timedelta(seconds=1)
    )
    date_after_inversion = str(end + np.timedelta64(1, "D"))[:10].replace("-", "")
    time_threshold = f"{date_after_inversion}_00"

    _, first_gc_file, _ = source_gc_files(gc_source_path, startday)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="xarray")
        with xr.open_dataset(first_gc_file) as ds:
            gc_lat_lon = {
                "lon": ds["lon"].values,
                "lat": ds["lat"].values,
            }

    hourly_targets = {}
    satellite_files = _matching_satellite_files(startday, endday, satellite_cache)
    for filename in satellite_files:
        result = read_and_filter_satellite(
            filename,
            product,
            start,
            end,
            lon_bounds,
            lat_bounds,
            use_water_obs.lower() == "true",
        )
        if result is None:
            continue
        satellite, sat_ind = result
        if satellite is None or len(sat_ind[0]) == 0:
            continue

        obs_mapped_to_gc = average_satellite_observations(
            satellite, species, gc_lat_lon, sat_ind, time_threshold
        )
        if len(obs_mapped_to_gc) == 0:
            continue

        for strdate in np.unique(obs_mapped_to_gc["time"]):
            rows = obs_mapped_to_gc[obs_mapped_to_gc["time"] == strdate]
            hourly_targets[strdate] = _merge_hourly_targets(
                hourly_targets.get(strdate),
                rows["iGC"].astype(np.int32),
                rows["jGC"].astype(np.int32),
            )

    return hourly_targets


def build_day_to_hours(startday, endday, hourly_targets=None):
    if hourly_targets:
        day_to_hours = {}
        for stamp in sorted(hourly_targets):
            day, hour = stamp.split("_")
            day_to_hours.setdefault(day, set()).add(int(hour))
        return day_to_hours

    day_to_hours = {}
    dt = datetime.datetime.strptime(startday, "%Y%m%d")
    dt_max = datetime.datetime.strptime(endday, "%Y%m%d")
    while dt < dt_max:
        dt_str = str(dt)[0:10].replace("-", "")
        day_to_hours[dt_str] = set(range(24))
        dt += datetime.timedelta(days=1)
    return day_to_hours


def _drop_anchor(ds):
    if "anchor" in ds:
        ds = ds.drop_vars("anchor")
    return ds


def _prepare_species_dataset(ds, source_mode):
    if source_mode == "standard":
        keep_vars = [v for v in ds.data_vars if v.startswith("SpeciesConcVV_")]
        return ds[keep_vars]
    if source_mode == "satdiagn":
        rename = {
            var: var.replace("SatDiagnConc_", "SpeciesConcVV_", 1)
            for var in ds.data_vars
            if var.startswith("SatDiagnConc_")
        }
        return ds[list(rename)].rename(rename)
    raise ValueError(f"Unsupported GEOS-Chem diagnostic mode: {source_mode}")


def _prepare_pedge_dataset(ds, source_mode):
    if source_mode == "standard":
        return ds[["Met_PEDGE"]]
    if source_mode == "satdiagn":
        return ds[["SatDiagnPEDGE"]].rename({"SatDiagnPEDGE": "Met_PEDGE"})
    raise ValueError(f"Unsupported GEOS-Chem diagnostic mode: {source_mode}")


def _sample_hour_dataset(ds, time_idx, targets):
    iGC = xr.DataArray(targets["iGC"], dims="obs")
    jGC = xr.DataArray(targets["jGC"], dims="obs")
    sampled = ds.isel(time=slice(time_idx, time_idx + 1)).isel(
        lat=jGC,
        lon=iGC,
        drop=True,
    )

    out = xr.Dataset()
    for name, da in sampled.data_vars.items():
        if "time" not in da.dims:
            continue
        if "lev" in da.dims:
            out[name] = da.transpose("time", "obs", "lev")
        elif "ilev" in da.dims:
            out[name] = da.transpose("time", "obs", "ilev")
        else:
            out[name] = da.transpose("time", "obs")

    for coord in ("lat", "lon"):
        if coord in out.coords and "obs" in out.coords[coord].dims:
            out = out.drop_vars(coord)

    out = out.assign_coords(
        obs=("obs", np.arange(len(targets["iGC"]), dtype=np.int32)),
        lat=("lat", ds["lat"].values),
        lon=("lon", ds["lon"].values),
        iGC=("obs", targets["iGC"]),
        jGC=("obs", targets["jGC"]),
    )
    out.attrs["obs_cache"] = "true"
    return out


def setup_gc_cache(
    startday,
    endday,
    gc_source_path,
    gc_destination_path,
    satellite_cache=None,
    satellite_product=None,
    use_water_obs="false",
    lon_bounds=None,
    lat_bounds=None,
    species="CH4",
    obs_only=False,
    use_gchp=False,
    hourly_targets=None,
    include_pedge=True,
):
    """
    Set up a GEOS-Chem cache for inversion sampling.

    If obs_only=True, save only the GEOS-Chem columns that are actually touched
    by filtered/averaged satellite super-observations for each hour. Otherwise,
    preserve the previous behavior and save full-grid hourly files.
    """
    os.makedirs(gc_destination_path, exist_ok=True)

    if obs_only and satellite_cache and satellite_product and lon_bounds and lat_bounds:
        if hourly_targets is None:
            hourly_targets = build_hourly_obs_targets(
                startday,
                endday,
                satellite_cache,
                satellite_product,
                species,
                lon_bounds,
                lat_bounds,
                use_water_obs,
                gc_source_path,
                use_gchp=use_gchp,
            )
        if use_gchp:
            hourly_targets = {}
    else:
        hourly_targets = {}

    day_to_hours = build_day_to_hours(startday, endday, hourly_targets)
    cache_jobs = int(os.environ.get("IMI_CACHE_JOBS", "-1"))

    def process(day, hours):
        source_mode, species_source, pedge_source = source_gc_files(gc_source_path, day)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="xarray")
            species_data = load_gc_hourly_dataset(species_source)
            pedge_data = load_gc_hourly_dataset(pedge_source) if include_pedge else None

        species_data = _prepare_species_dataset(_drop_anchor(species_data), source_mode)
        if include_pedge:
            pedge_data = _prepare_pedge_dataset(_drop_anchor(pedge_data), source_mode)

        for hour in sorted(hours):
            stamp = f"{day}_{hour:02d}"
            species_save_pth = (
                f"{gc_destination_path}/GEOSChem.SpeciesConc.{day}_{hour:02d}00z.nc4"
            )
            pedge_save_pth = (
                f"{gc_destination_path}/GEOSChem.StateMetLevEdge.{day}_{hour:02d}00z.nc4"
            )

            if hourly_targets and stamp in hourly_targets:
                targets = hourly_targets[stamp]
                if source_mode == "satdiagn":
                    species_for_sample = _select_satdiagn_overpass(species_data, hour)
                    pedge_for_sample = (
                        _select_satdiagn_overpass(pedge_data, hour) if include_pedge else None
                    )
                    time_idx = 0
                else:
                    species_for_sample = species_data
                    pedge_for_sample = pedge_data
                    time_idx = hour
                if not os.path.isfile(species_save_pth):
                    sampled_species = _sample_hour_dataset(species_for_sample, time_idx, targets)
                    _atomic_to_netcdf(
                        sampled_species, species_save_pth,
                        {v: {"zlib": True, "complevel": 1} for v in sampled_species.data_vars},
                    )
                if include_pedge and not os.path.isfile(pedge_save_pth):
                    sampled_pedge = _sample_hour_dataset(pedge_for_sample, time_idx, targets)
                    _atomic_to_netcdf(
                        sampled_pedge, pedge_save_pth,
                        {v: {"zlib": True, "complevel": 1} for v in sampled_pedge.data_vars},
                    )
                continue

            if hourly_targets:
                continue

            if not os.path.isfile(species_save_pth):
                time_idx = hour if source_mode == "standard" else 0
                source_species = (
                    species_data
                    if source_mode == "standard"
                    else _select_satdiagn_overpass(species_data, hour)
                )
                species_for_hour = source_species.isel(time=slice(time_idx, time_idx + 1, 1))
                _atomic_to_netcdf(
                    species_for_hour, species_save_pth,
                    {v: {"zlib": True, "complevel": 1} for v in species_for_hour.data_vars},
                )
            if include_pedge and not os.path.isfile(pedge_save_pth):
                time_idx = hour if source_mode == "standard" else 0
                source_pedge = (
                    pedge_data
                    if source_mode == "standard"
                    else _select_satdiagn_overpass(pedge_data, hour)
                )
                pedge_for_hour = source_pedge.isel(time=slice(time_idx, time_idx + 1, 1))
                _atomic_to_netcdf(
                    pedge_for_hour, pedge_save_pth,
                    {v: {"zlib": True, "complevel": 1} for v in pedge_for_hour.data_vars},
                )

    Parallel(n_jobs=cache_jobs)(
        delayed(process)(day, hours) for day, hours in day_to_hours.items()
    )
    marker_path = os.path.join(gc_destination_path, ".setup_gc_cache_complete")
    with open(marker_path, "w", encoding="ascii") as handle:
        handle.write(
            "cache_logic_version=2\n"
            f"start={startday}\nend={endday}\nobs_only={str(obs_only).lower()}\ninclude_pedge={str(include_pedge).lower()}\n"
        )
    print(f"Set up hourly data files in {gc_destination_path}")


if __name__ == "__main__":
    import sys

    startday = sys.argv[1]
    endday = sys.argv[2]
    gc_source_path = sys.argv[3]
    gc_destination_path = sys.argv[4]
    satellite_cache = sys.argv[5] if len(sys.argv) > 5 else None
    satellite_product = sys.argv[6] if len(sys.argv) > 6 else None
    use_water_obs = sys.argv[7] if len(sys.argv) > 7 else "false"
    lon_bounds = [float(sys.argv[8]), float(sys.argv[9])] if len(sys.argv) > 9 else None
    lat_bounds = [float(sys.argv[10]), float(sys.argv[11])] if len(sys.argv) > 11 else None
    species = sys.argv[12] if len(sys.argv) > 12 else "CH4"
    obs_only = sys.argv[13].lower() == "true" if len(sys.argv) > 13 else False
    use_gchp = sys.argv[14].lower() == "true" if len(sys.argv) > 14 else False

    setup_gc_cache(
        startday,
        endday,
        gc_source_path,
        gc_destination_path,
        satellite_cache,
        satellite_product,
        use_water_obs,
        lon_bounds,
        lat_bounds,
        species,
        obs_only,
        use_gchp,
    )
