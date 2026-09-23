#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import glob
import numpy as np
import re
import os
import datetime
import gc
import xarray as xr
from src.inversion_scripts.utils import save_obj
from src.utilities.config_utils import load_config
from src.inversion_scripts.operators.satellite_operator import (
    apply_average_satellite_operator,
    apply_satellite_operator,
)
from joblib import Parallel, delayed


def apply_operator(operator, params, config):
    """
    Run the chosen operator based on selected instrument

    Arguments
        operator [str]    : Data conversion operator to use
        params   [dict]   : parameters to run the given operator
    Returns
        output   [dict]   : Dictionary with:
                            - obs_GC : GEOS-Chem and satellite column data
                            - satellite columns
                            - GEOS-Chem columns
                            - satellite lat, lon
                            - satellite lat index, lon index
                              If build_jacobian=True, also include:
                                - K      : Jacobian matrix
    """
    if operator == "satellite_average":
        return apply_average_satellite_operator(
            params["filename"],
            params["species"],
            params["satellite_product"],
            params["n_elements"],
            params["gc_startdate"],
            params["gc_enddate"],
            params["xlim"],
            params["ylim"],
            params["gc_cache"],
            params["build_jacobian"],
            params["period_i"],
            config,
            params["use_water_obs"],
        )
    elif operator == "satellite":
        return apply_satellite_operator(
            params["filename"],
            params["species"],
            params["satellite_product"],
            params["n_elements"],
            params["gc_startdate"],
            params["gc_enddate"],
            params["xlim"],
            params["ylim"],
            params["gc_cache"],
            params["period_i"],
            config,
            params["use_water_obs"],
        )
    else:
        raise ValueError("Error: invalid operator selected.")


def file_has_usable_obs(
    filename,
    satellite_product,
    gc_startdate,
    gc_enddate,
    xlim,
    ylim,
    use_water_obs,
):
    """Cheap prefilter to skip satellite files with no valid in-domain observations."""
    if satellite_product != "BlendedTROPOMI":
        return True

    try:
        with xr.open_dataset(filename) as ds:
            longitude = ds["longitude"].values
            latitude = ds["latitude"].values
            time_utc = np.array(
                [d.replace("Z", "") for d in ds["time_utc"].values]
            ).astype("datetime64[ns]")
            longitude_bounds = ds["longitude_bounds"].values
            surface_classification_raw = ds["surface_classification"].values.astype("uint8")
            surface_classification = (surface_classification_raw & 0x03).astype(int)
            surface_classification_0xF9 = (surface_classification_raw & 0xF9).astype(int)
            chi_square_swir = ds["chi_square_SWIR"].values

        valid_idx = (
            (longitude > xlim[0])
            & (longitude < xlim[1])
            & (latitude > ylim[0])
            & (latitude < ylim[1])
            & (time_utc >= gc_startdate)
            & (time_utc <= gc_enddate)
            & (longitude_bounds.ptp(axis=1) < 100)
            & ~(
                (surface_classification == 3)
                | ((surface_classification == 2) & (chi_square_swir > 20000))
            )
            & ~(surface_classification_0xF9 == 184)
            & (latitude > -60)
        )
        if not use_water_obs:
            valid_idx = valid_idx & (surface_classification != 1)
        return bool(np.any(valid_idx))
    except Exception as exc:
        print(f"Prefilter failed for {filename}: {exc}. Processing file anyway.", flush=True)
        return True


if __name__ == "__main__":

    workdir = sys.argv[1]
    config = load_config(sys.argv[2])
    startday = sys.argv[3]
    endday = sys.argv[4]
    lonmin = float(sys.argv[5])
    lonmax = float(sys.argv[6])
    latmin = float(sys.argv[7])
    latmax = float(sys.argv[8])
    n_elements = int(sys.argv[9])
    species = sys.argv[10]
    satellite_cache = sys.argv[11]
    satellite_product = sys.argv[12]
    use_water_obs = sys.argv[13]
    isPost = sys.argv[14]
    period_i = int(sys.argv[15])
    build_jacobian = sys.argv[16]
    viz_prior = sys.argv[17]

    # Reformat start and end days for datetime in configuration
    start = f"{startday[0:4]}-{startday[4:6]}-{startday[6:8]} 00:00:00"
    end = f"{endday[0:4]}-{endday[4:6]}-{endday[6:8]} 23:59:59"

    # Configuration
    if build_jacobian.lower() == "true":
        build_jacobian = True
    else:
        build_jacobian = False
    obs_only_cache = bool(config.get("ObservationOnlyJacobianCache", False))
    # The residual-error (REM) diagonal + off-diagonal So need the unaveraged individual pixels
    # (viz_output), so still produce them even in obs-only-cache mode when either is requested.
    rem_needs_viz = (
        str(config.get("OffDiagonalObsCov", True)).strip().lower() in ("true", "1", "yes")
        or str(config.get("UseResidualObsError", False)).strip().lower() in ("true", "1", "yes")
    )
    if isPost.lower() == "false":  # if sampling prior simulation
        gc_cache = f"{workdir}/data_geoschem"
        outputdir = f"{workdir}/data_converted"
        vizdir = f"{workdir}/data_visualization"

        # for lognormal, we also sample the prior simulation in a
        # separate call to jacobian.py solely for visualization purposes
        if viz_prior.lower() == "true":
            gc_cache = f"{gc_cache}_prior"
            outputdir = f"{outputdir}_prior"
            vizdir = f"{vizdir}_prior"

    else:  # if sampling posterior simulation
        gc_cache = f"{workdir}/data_geoschem_posterior"
        outputdir = f"{workdir}/data_converted_posterior"
        vizdir = f"{workdir}/data_visualization_posterior"

    xlim = [lonmin, lonmax]
    ylim = [latmin, latmax]
    gc_startdate = np.datetime64(datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S"))
    gc_enddate = np.datetime64(
        datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        - datetime.timedelta(days=1)
    )
    print("Start:", gc_startdate)
    print("End:", gc_enddate)
    default_jobs = min(8, os.cpu_count() or 1)
    n_parallel_jobs = int(config.get("InversionParallelJobs", default_jobs))
    n_parallel_jobs = max(1, n_parallel_jobs)

    # Get satellite data filenames for the desired date range
    allfiles = glob.glob(f"{satellite_cache}/*.nc")
    sat_files = []
    for index in range(len(allfiles)):
        filename = allfiles[index]
        shortname = re.split(r"\/", filename)[-1]
        shortname = re.split(r"\.", shortname)[0]
        strdate = re.split(r"\.|_+|T", shortname)[4]
        strdate = datetime.datetime.strptime(strdate, "%Y%m%d")
        if (strdate >= gc_startdate) and (strdate <= gc_enddate):
            sat_files.append(filename)
    sat_files.sort()
    print("Found", len(sat_files), "satellite data files before prefilter.")

    filtered_sat_files = [
        filename
        for filename in sat_files
        if file_has_usable_obs(
            filename,
            satellite_product,
            gc_startdate,
            gc_enddate,
            xlim,
            ylim,
            use_water_obs.lower() == "true",
        )
    ]
    skipped_prefilter = len(sat_files) - len(filtered_sat_files)
    sat_files = filtered_sat_files
    print(
        "Keeping",
        len(sat_files),
        "satellite data files after prefilter; skipped",
        skipped_prefilter,
        "with no usable observations.",
    )

    # Map GEOS-Chem to satellite observation space
    # Also return Jacobian matrix if build_jacobian=True
    def process(filename):

        # Check if satellite file has already been processed
        print("========================")
        shortname = re.split(r"\/", filename)[-1]
        print(shortname)
        date = re.split(r"\.", shortname)[0]

        # If not yet processed, run apply_average_satellite_operator()
        if not os.path.isfile(f"{outputdir}/{date}_GCtoSatellite.pkl"):
            print("Applying satellite operator...")

            output = apply_operator(
                "satellite_average",
                {
                    "filename": filename,
                    "species" : species,
                    "satellite_product": satellite_product,
                    "n_elements": n_elements,
                    "gc_startdate": gc_startdate,
                    "gc_enddate": gc_enddate,
                    "xlim": xlim,
                    "ylim": ylim,
                    "gc_cache": gc_cache,
                    "build_jacobian": build_jacobian,
                    "period_i": period_i,
                    "use_water_obs": use_water_obs,
                },
                config,
            )

            # we also save out the unaveraged satellite operator for visualization purposes
            # (also the individual-pixel source the REM diagonal / off-diagonal So consume)
            viz_output = None
            if (not obs_only_cache) or rem_needs_viz:
                viz_output = apply_operator(
                    "satellite",
                    {
                        "filename": filename,
                        "species" : species,
                        "satellite_product": satellite_product,
                        "n_elements": n_elements,
                        "gc_startdate": gc_startdate,
                        "gc_enddate": gc_enddate,
                        "xlim": xlim,
                        "ylim": ylim,
                        "gc_cache": gc_cache,
                        "build_jacobian": False,
                        "period_i": period_i,
                        "use_water_obs": use_water_obs,
                    },
                    config,
                )

            if output == None:
                return 0
        else:
            return 0

        if output["obs_GC"].shape[0] > 0:
            print("Saving .pkl file")
            save_obj(output, f"{outputdir}/{date}_GCtoSatellite.pkl")
            if viz_output is not None:
                save_obj(viz_output, f"{vizdir}/{date}_GCtoSatellite.pkl")

        #Clean up to reduce memory use
        del output, viz_output
        gc.collect()

        return 0

    results = Parallel(n_jobs=n_parallel_jobs)(
        delayed(process)(filename) for filename in sat_files
    )
    print(f"Wrote files to {outputdir}")
