#!/usr/bin/env python3
"""Create monthly nudged prior scale factors for a sequential inversion."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--initial-start", required=True)
    parser.add_argument("--prior-cache", required=True)
    parser.add_argument("--state-vector", required=True)
    parser.add_argument("--inversion-utils-dir", required=True)
    parser.add_argument("--previous-scale-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--nudge-factor", type=float, default=0.2)
    return parser.parse_args()


def make_unit_scale_factors(state_vector_path):
    state = xr.load_dataset(state_vector_path)
    state_vector = state["StateVector"]
    return xr.Dataset(
        {"ScaleFactor": (("lat", "lon"), np.ones(state_vector.shape, dtype=np.float32))},
        coords={"lat": state["lat"].values, "lon": state["lon"].values},
    )


def emission_field(ds):
    if "EmisCH4_Total_ExclSoilAbs" in ds:
        return ds["EmisCH4_Total_ExclSoilAbs"]
    return ds["EmisCH4_Total"]


def save_scale_factors(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.lat.attrs["units"] = "degrees_north"
    ds.lat.attrs["long_name"] = "Latitude"
    ds.lon.attrs["units"] = "degrees_east"
    ds.lon.attrs["long_name"] = "Longitude"
    ds.ScaleFactor.attrs["units"] = "1"
    ds.to_netcdf(
        path,
        encoding={
            "ScaleFactor": {"zlib": True, "complevel": 1},
            "lat": {"dtype": "float32"},
            "lon": {"dtype": "float32"},
        },
    )


def main():
    args = parse_args()
    sys.path.insert(0, args.inversion_utils_dir)
    from utils import get_mean_emissions, get_posterior_emissions, sum_total_emissions

    scale_factors = make_unit_scale_factors(args.state_vector)
    state = xr.load_dataset(args.state_vector)
    labels = state["StateVector"]
    mask = labels > 0
    out_path = os.path.join(args.outdir, f"ScaleFactors_{args.start}.nc")

    if args.start == args.initial_start:
        save_scale_factors(scale_factors, out_path)
        print(f"[{args.start}] first period: wrote unit nudged scale factors to {out_path}")
        return

    prev_start = (pd.to_datetime(args.start) - pd.DateOffset(months=1)).strftime("%Y%m%d")
    prev_end = args.start
    prev_sf_path = os.path.join(args.previous_scale_dir, f"ScaleFactors_{prev_start}.nc")
    if not os.path.exists(prev_sf_path):
        raise FileNotFoundError(f"Missing previous posterior scale factor file: {prev_sf_path}")

    prior_current = get_mean_emissions(args.start, args.end, args.prior_cache)
    prior_previous = get_mean_emissions(prev_start, prev_end, args.prior_cache)
    emis_current = emission_field(prior_current)
    emis_previous = emission_field(prior_previous)
    areas = prior_current["AREA"]
    posterior_sf_previous = xr.load_dataset(prev_sf_path)["ScaleFactor"]
    posterior_previous = get_posterior_emissions(prior_previous, posterior_sf_previous)
    posterior_previous_emis = emission_field(posterior_previous).where(emis_previous.notnull(), 0)
    posterior_previous_emis = posterior_previous_emis.where(posterior_previous_emis > 0, 0)

    nudged_raw = args.nudge_factor * emis_current + (1.0 - args.nudge_factor) * posterior_previous_emis
    previous_total = sum_total_emissions(posterior_previous_emis, areas, mask)
    nudged_total = sum_total_emissions(nudged_raw, areas, mask)
    total_preserving_scale = previous_total / nudged_total if nudged_total != 0 else 1.0

    scaled = nudged_raw.where(~mask, 0) + nudged_raw.where(mask, 0) * total_preserving_scale
    sf_vals = xr.where(emis_current != 0, scaled / emis_current, 1.0)
    scale_factors["ScaleFactor"] = sf_vals.where(mask).fillna(1).astype("float32")
    save_scale_factors(scale_factors, out_path)
    print(f"[{args.start}] wrote nudged scale factors to {out_path}")
    print(f"[{args.start}] nudge_factor={args.nudge_factor}, total_preserving_scale={total_preserving_scale}")


if __name__ == "__main__":
    main()
