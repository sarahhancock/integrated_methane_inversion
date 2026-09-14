#!/usr/bin/env python3

import glob
import xarray as xr
import os
import re
import sys

from joblib import Parallel, delayed

from src.inversion_scripts.setup_gc_cache import (
    build_hourly_obs_targets,
    setup_gc_cache,
)


def _prune_raw_output(output_dir):
    """
    Delete the bulky raw GEOS-Chem SpeciesConc / StateMetLevEdge files for a
    perturbation run once its small observation-only cache has been written.
    Files inside the obs_cache/ subdirectory are NOT matched by these globs
    (glob does not recurse), so the cache is preserved.
    """
    removed = 0
    for pattern in ("GEOSChem.SpeciesConc.*.nc4", "GEOSChem.StateMetLevEdge.*.nc4"):
        for path in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def main():
    startday = sys.argv[1]
    endday = sys.argv[2]
    jacobian_runs_dir = sys.argv[3]
    satellite_cache = sys.argv[4]
    satellite_product = sys.argv[5]
    use_water_obs = sys.argv[6]
    lon_bounds = [float(sys.argv[7]), float(sys.argv[8])]
    lat_bounds = [float(sys.argv[9]), float(sys.argv[10])]
    species = sys.argv[11]
    use_gchp = sys.argv[12].lower() == "true"

    # Only the prior (0000) and perturbation (NNNN) runs belong in the Jacobian
    # obs-cache build. Exclude anything without a 4-digit numeric suffix --
    # notably the separate "_background" analysis run, which has no perturbation
    # output and would otherwise break the build (FileNotFoundError).
    run_dirs = sorted(glob.glob(os.path.join(jacobian_runs_dir, "*")))
    run_output_dirs = [
        os.path.join(run_dir, "OutputDir")
        for run_dir in run_dirs
        if os.path.isdir(os.path.join(run_dir, "OutputDir"))
        and re.search(r"_\d{4}$", os.path.basename(run_dir))
    ]
    if not run_output_dirs:
        print("No Jacobian OutputDir directories found.")
        return 0

    reference_source = run_output_dirs[0]
    hourly_targets = build_hourly_obs_targets(
        startday,
        endday,
        satellite_cache,
        satellite_product,
        species,
        lon_bounds,
        lat_bounds,
        use_water_obs,
        reference_source,
        use_gchp=use_gchp,
    )

    # Build the observation-only cache for one run (NO pruning here -- see below).
    def process(output_dir):
        destination = os.path.join(output_dir, "obs_cache")
        setup_gc_cache(
            startday,
            endday,
            output_dir,
            destination,
            satellite_cache,
            satellite_product,
            use_water_obs,
            lon_bounds,
            lat_bounds,
            species,
            obs_only=True,
            use_gchp=use_gchp,
            hourly_targets=hourly_targets,
            include_pedge=False,
        )

    # We parallelize over RUNS here, so force the inner setup_gc_cache to process
    # its days SERIALLY -- otherwise it is nested parallelism (runs x days), which
    # loaded hundreds of GB of daily fields at once and OOM-killed the inversion
    # job (254 GB). With the inner loop serial, each worker holds only ~1 day at a
    # time, so memory ~= n_jobs x (one daily file). Override with IMI_OBSCACHE_JOBS.
    os.environ["IMI_CACHE_JOBS"] = "1"
    # Inner loop is serial, so peak mem ~= n_jobs x (one ~0.1 GB daily file). Measured
    # only 1.6 GB total at n_jobs=4 on a 1 TB node -> use all 24 allocated cores.
    n_jobs = int(os.environ.get("IMI_OBSCACHE_JOBS", "56"))
    n_jobs = max(1, min(n_jobs, 56))
    Parallel(n_jobs=n_jobs)(delayed(process)(output_dir) for output_dir in run_output_dirs)

    # Prune bulky raw output ONLY after the ENTIRE build has succeeded above. If
    # the build raised (OOM/error), this is never reached, so every run keeps its
    # raw output and the inversion can simply be re-run -- pruning is never left in
    # a half-done state that a rebuild can't recover from. Keep raw for the prior
    # (0000) / background runs (the base GC cache needs the prior's full output).
    pruned = 0
    for output_dir in run_output_dirs:
        run_dir_name = os.path.basename(os.path.dirname(output_dir))
        if run_dir_name.endswith("_0000") or run_dir_name.endswith("_background"):
            continue
        sentinel = os.path.join(output_dir, "obs_cache", ".setup_gc_cache_complete")
        if os.path.isfile(sentinel):
            # VERIFY the cache is readable before deleting the raw output it was built
            # from. The cache becomes the ONLY copy once raw is pruned, so a corrupt
            # cache file is otherwise unrecoverable and the whole simulation must be
            # re-run (this cost a 9h rerun for 202305 run 0016, whose
            # GEOSChem.SpeciesConc.20230503_2000z.nc4 was 198KB and unreadable).
            # Opening headers only, so this is milliseconds per file.
            bad = []
            for cf in sorted(glob.glob(os.path.join(output_dir, "obs_cache", "*.nc4"))):
                try:
                    with xr.open_dataset(cf) as _ds:
                        pass
                except Exception as exc:  # noqa: BLE001 - any read failure disqualifies
                    bad.append((cf, repr(exc)))
            if bad:
                print(f"NOT pruning {output_dir}: {len(bad)} unreadable cache file(s); "
                      f"keeping raw output so the cache can be rebuilt. First: {bad[0][0]}",
                      flush=True)
                continue
            pruned += _prune_raw_output(output_dir)
    print(f"Set up observation-only Jacobian caches in {jacobian_runs_dir} "
          f"(pruned {pruned} raw files after successful build)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
