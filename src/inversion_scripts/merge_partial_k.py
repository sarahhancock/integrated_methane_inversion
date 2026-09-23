import os
import sys
import pickle as pickle
import numpy as np
import xarray as xr
from pathlib import Path
from src.inversion_scripts.utils import (
    load_obj,
    calculate_superobservation_error,
    ensure_float_list,
    map_files_to_reference,
)
from src.utilities.config_utils import load_config


def calc_so(obs_error, obs_GC):
    """Calculate the superobservation error for each observation given the observation error"""
    # calculate superobservation error
    s_superO_1 = calculate_superobservation_error(obs_error, 1)
    s_superO_p = np.array(
        [
            calculate_superobservation_error(obs_error, p) if p >= 1 else s_superO_1
            for p in obs_GC[:, 4]
        ]
    )
    # scale error variance by gP value following Chen et al. 2023
    gP = s_superO_p**2 / s_superO_1**2
    obs_error = obs_error**2
    obs_error = gP * obs_error

    # check to make sure obs_err isn't negative, set 1 as default value
    obs_error = [obs if obs > 0 else 1 for obs in obs_error]
    return obs_error

from functools import partial
print = partial(print, flush = True)


def save_npz_unless_preserving_so(path, preserve_existing_so, **kwargs):
    if preserve_existing_so and os.path.exists(path):
        print(f"Preserving existing So file {path}")
        return
    np.savez(path, **kwargs)

def merge_partial_k(satdat_dir, lat_bounds, lon_bounds, obs_errs, precomp_K, allow_missing_k=False):
    """
    Description:
        This function is used to generate the full jacobian matrix (K), observations (y),
        background vector (y_bkgd), and observational error (So) for the lognormal inversion.

        The normal inversion script of the IMI reads in the jacobian matrix, observations,
        and observational error piece by piece in order to avoid loading the full jacobian
        matrix into memory (which can be quite large). The lognormal inversion script
        requires the full form of these variables to iteratively solve for the posterior.
        Here we load in the partial jacobian matrices and observations from each satellite
        data file and concatenate them into the full jacobian matrix and observation vector,
        for use in the lognormal inversion script. We also calculate the observational error
        and background vector.

    Parameters:
        satdat_dir    [str]: path to directory containing satellite data files
        lat_bounds   [list]: list of latitude bounds to consider each bound is a tuple
        lon_bounds   [list]: list of longitude bounds to consider each bound is a tuple
        obs_errs     [list]: list observational error values
        precomp_K [boolean]: whether or not to use precomputed jacobian matrices
    """
    # Get observed and GEOS-Chem-simulated TROPOMI columns
    files = [f for f in np.sort(os.listdir(satdat_dir)) if "Satellite" in f]

    # Initialize dictionary to store observational errors
    so_dict = {}
    for obs_err in obs_errs:
        key = f"so_{obs_err}"
        so_dict[key] = [None for i in range(len(files))]
    satellite_list = [None for i in range(len(files))]
    geos_prior_list = [None for i in range(len(files))]
    lon_list = [None for i in range(len(files))]
    lat_list = [None for i in range(len(files))]
    observation_count_list = [None for i in range(len(files))]
    dates_list = [None for i in range(len(files))]   # YYYYMMDD per super-ob (overpass date)
    K_list = [None for i in range(len(files))]
    
    # If using precomputed jacobian, get the mappings to reference jacobian files
    if precomp_K:
        ref_dir = satdat_dir.replace("data_converted", "data_converted_reference")
        K_ref_file_mappings = map_files_to_reference(satdat_dir, ref_dir)

    for i, f in enumerate(files):
        # Get paths
        pth = os.path.join(satdat_dir, f)
        # Get same file from bc folder
        # Load satellite/GEOS-Chem and Jacobian matrix data from the .pkl file
        obj = load_obj(pth)
        # If there aren't any satellite observations on this day, skip
        if obj["obs_GC"].shape[0] == 0:
            continue
        # Otherwise, grab the satellite/GEOS-Chem data
        obs_GC = obj["obs_GC"]
        # Only consider data within latitude and longitude bounds
        ind = np.where(
            (obs_GC[:, 2] >= lon_bounds[0])
            & (obs_GC[:, 2] <= lon_bounds[1])
            & (obs_GC[:, 3] >= lat_bounds[0])
            & (obs_GC[:, 3] <= lat_bounds[1])
        )
        if len(ind[0]) == 0:  # Skip if no data in bounds
            continue
        obs_GC = obs_GC[ind[0], :]  # satellite and GEOS-Chem data within bounds

        # concatenate full jacobian, obs, so, and prior
        satellite_list[i] = np.asarray(obs_GC[:, 0], dtype=np.float32)
        geos_prior_list[i] = np.asarray(obs_GC[:, 1], dtype=np.float32)
        lon_list[i] = np.asarray(obs_GC[:, 2], dtype=np.float32)
        lat_list[i] = np.asarray(obs_GC[:, 3], dtype=np.float32)
        observation_count_list[i] = np.asarray(obs_GC[:, 4], dtype=np.float32)
        # Overpass date (YYYYMMDD) parsed from the pkl filename, one value per
        # super-ob -- same as merge_gc_run.py. Guarded so a non-standard name
        # never breaks the merge.
        try:
            date_val = f.split("____")[1][:8]
        except (IndexError, AttributeError):
            date_val = "00000000"
        dates_list[i] = np.repeat(date_val, obs_GC.shape[0])

        # read K from reference dir if precomp_K is true
        if precomp_K:
            # Get Jacobian from reference inversion
            fi_ref = K_ref_file_mappings.get(Path(pth))   # may be None (str() would hide it -> "None")
            if fi_ref is None:
                print(f"No reference file found for {pth}. Skipping this file.")
                continue
            dat_ref = load_obj(str(fi_ref))
            K_temp = dat_ref["K"][ind[0]]
        else:
            K_temp = obj["K"][ind[0]] if "K" in obj else None
        
        # add K_temp to K_list
        if K_temp is not None:
            K_list[i] = np.asarray(K_temp, dtype=np.float32)

        for obs_err in obs_errs:
            key = f"so_{obs_err}"
            obs_error = calc_so(obs_err, obs_GC)
            so_dict[key][i] = np.asarray(obs_error, dtype=np.float32)

    K_list = [arr for arr in K_list if arr is not None]
    geos_prior_list = [arr for arr in geos_prior_list if arr is not None]
    satellite_list = [arr for arr in satellite_list if arr is not None]
    lon_list = [arr for arr in lon_list if arr is not None]
    lat_list = [arr for arr in lat_list if arr is not None]
    observation_count_list = [arr for arr in observation_count_list if arr is not None]
    if len(satellite_list) == 0:
        raise ValueError("No valid observation chunks found for the requested month/domain.")

    geos_prior = np.concatenate(geos_prior_list, axis=0)
    satellite = np.concatenate(satellite_list, axis=0)
    lon = np.concatenate(lon_list, axis=0)
    lat = np.concatenate(lat_list, axis=0)
    observation_count = np.concatenate(observation_count_list, axis=0)
    dates_list = [arr for arr in dates_list if arr is not None]
    dates = np.concatenate(dates_list, axis=0) if dates_list else np.array([], dtype="<U8")
    for k,v in so_dict.items():
        v = [arr for arr in v if arr is not None]
        so_dict[k] = np.concatenate(v, axis=0)

    # Store merged monthly products compactly; the solver promotes to float64
    # internally before matrix algebra.
    gc_prior = np.asarray(geos_prior, dtype=np.float32)
    obs_satellite = np.asarray(satellite, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float32)
    observation_count = np.asarray(observation_count, dtype=np.float32)
    if len(K_list) == 0 and allow_missing_k:
        K = None
    elif len(K_list) == 0:
        existing_k_path = Path("full_jacobian_K.npz")
        if not existing_k_path.exists():
            raise ValueError(
                "No Jacobian chunks were found in the observation-space files, and "
                f"{existing_k_path} does not exist. Re-run jacobian.py with "
                "build_jacobian=True, or run this from an inversion directory with "
                "an existing full_jacobian_K.npz."
            )
        with np.load(existing_k_path) as existing_k:
            K = np.asarray(existing_k["K"], dtype=np.float32)
        if K.shape[0] != obs_satellite.size:
            raise ValueError(
                "Existing full_jacobian_K row count does not match merged "
                f"observation metadata: {K.shape[0]} vs {obs_satellite.size}."
            )
    else:
        K = np.asarray(np.concatenate(K_list, axis=0), dtype=np.float32)
    for key, value in so_dict.items():
        so_dict[key] = np.asarray(value, dtype=np.float32)

    obs_metadata = {
        "lon": lon,
        "lat": lat,
        "obs_tropomi": obs_satellite,
        "gc_ch4_prior": gc_prior,
        "observation_count": observation_count,
        "dates": dates,
    }

    return gc_prior, obs_satellite, K, so_dict, obs_metadata


if __name__ == "__main__":
    # read in arguments
    satdat_dir = sys.argv[1]
    state_vector_filepath = sys.argv[2]
    config_path = sys.argv[3]
    precomputed_jacobian = sys.argv[4].lower() == "true"
    allow_missing_k = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
    preserve_existing_so = os.environ.get("IMI_PRESERVE_EXISTING_SO", "false").lower() == "true"

    # Load config file
    config = load_config(config_path)

    # Ensure obs_error is a list of floats
    obs_errors = ensure_float_list(config["ObsError"])

    # directory containing partial K matrices
    state_vector = xr.load_dataset(state_vector_filepath)
    state_vector_labels = state_vector["StateVector"]
    if ~config['UseGCHP']:
        lon_bounds = [np.min(state_vector.lon.values), np.max(state_vector.lon.values)]
        lat_bounds = [np.min(state_vector.lat.values), np.max(state_vector.lat.values)]
    else:
        lon_bounds = [-180, 180]
        lat_bounds = [-90, 90]

    # Paths to GEOS/satellite data
    gc_bkgd, obs_satellite, jacobian_K, so_dict, obs_metadata = merge_partial_k(
        satdat_dir, lat_bounds, lon_bounds, obs_errors, precomputed_jacobian, allow_missing_k
    )

    if jacobian_K is not None:
        np.savez("full_jacobian_K.npz", K=jacobian_K)
    np.savez("obs_satellite.npz", obs_satellite=obs_satellite)
    np.savez("gc_bkgd.npz", gc_bkgd=gc_bkgd)
    np.savez("obs_metadata.npz", **obs_metadata)
    save_npz_unless_preserving_so("so_super.npz", preserve_existing_so, **so_dict)

    start = str(config["StartDate"])
    end = str(config["EndDate"])
    inversion_data_path = os.path.join(config["OutputPath"], config["RunName"], "inversion_data")
    os.makedirs(os.path.join(inversion_data_path, "K"), exist_ok=True)
    os.makedirs(os.path.join(inversion_data_path, "obs_ch4_tropomi"), exist_ok=True)
    os.makedirs(os.path.join(inversion_data_path, "gc_ch4_prior"), exist_ok=True)
    os.makedirs(os.path.join(inversion_data_path, "observations"), exist_ok=True)
    os.makedirs(os.path.join(inversion_data_path, "y"), exist_ok=True)
    os.makedirs(os.path.join(inversion_data_path, "xch4_0"), exist_ok=True)
    os.makedirs(os.path.join(inversion_data_path, "so"), exist_ok=True)
    if jacobian_K is not None:
        np.savez(os.path.join(inversion_data_path, "K", f"K_{start}_{end}.npz"), K=jacobian_K)
    np.savez(
        os.path.join(inversion_data_path, "obs_ch4_tropomi", f"obs_ch4_tropomi_{start}_{end}.npz"),
        obs_tropomi=obs_satellite.reshape(1, -1),
    )
    np.savez(
        os.path.join(inversion_data_path, "gc_ch4_prior", f"gc_ch4_prior_{start}_{end}.npz"),
        gc_ch4_prior=gc_bkgd,
        gc_ch4=gc_bkgd,
    )
    np.savez(
        os.path.join(inversion_data_path, "observations", f"observations_{start}_{end}.npz"),
        **obs_metadata,
    )
    # Standalone date file matching the structure of the other per-super-ob arrays
    os.makedirs(os.path.join(inversion_data_path, "date"), exist_ok=True)
    np.savez(
        os.path.join(inversion_data_path, "date", f"date_{start}_{end}.npz"),
        dates=obs_metadata["dates"],
    )
    np.savez(os.path.join(inversion_data_path, "y", f"y_{start}_{end}.npz"), y=obs_satellite)
    np.savez(
        os.path.join(inversion_data_path, "xch4_0", f"xch4_0_{start}_{end}.npz"),
        xch4_0=gc_bkgd,
        gc_ch4=gc_bkgd,
    )
    save_npz_unless_preserving_so(
        os.path.join(inversion_data_path, "so", f"so_{start}_{end}.npz"),
        preserve_existing_so,
        **so_dict,
    )
