#!/usr/bin/env python3
"""Save posterior scale factors relative to the original prior."""

import argparse
import os

import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gridded-posterior", required=True)
    parser.add_argument("--nudged-scale-factors", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    posterior = xr.load_dataset(args.gridded_posterior)["ScaleFactor"]
    nudged = xr.load_dataset(args.nudged_scale_factors)["ScaleFactor"]
    cumulative = (posterior * nudged).astype("float32")
    ds = xr.Dataset(
        {"ScaleFactor": cumulative},
        coords={"lat": posterior["lat"].values, "lon": posterior["lon"].values},
    )
    ds.lat.attrs["units"] = "degrees_north"
    ds.lat.attrs["long_name"] = "Latitude"
    ds.lon.attrs["units"] = "degrees_east"
    ds.lon.attrs["long_name"] = "Longitude"
    ds.ScaleFactor.attrs["units"] = "1"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds.to_netcdf(
        args.out,
        encoding={
            "ScaleFactor": {"zlib": True, "complevel": 1},
            "lat": {"dtype": "float32"},
            "lon": {"dtype": "float32"},
        },
    )
    print(f"Saved cumulative posterior scale factors to {args.out}")


if __name__ == "__main__":
    main()
