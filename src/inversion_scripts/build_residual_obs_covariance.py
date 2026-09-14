#!/usr/bin/env python3
"""Build residual-error observational error covariance (So) from merged TROPOMI observations.

Two products are generated:

  1. Diagonal So  (residual error method)
     Local spatial variance sk_i is estimated from the multi-year distribution of
     (TROPOMI - prior) residuals in each GEOS-Chem grid cell.  A superobservation
     count-scaling function g(P) (Chen et al. 2023) is then applied so that
     averaging P individual retrievals into one superobservation reduces the error
     appropriately.  The result replaces the fixed obs_err in the standard pipeline.

  2. Spatial correlation parameters  (for off-diagonal So in invert.py)
     Residual anomalies (residual minus its cell mean) are paired across months and
     binned by haversine separation distance.  The mean cross-product normalised by
     the anomaly variance gives the empirical correlation as a function of distance.
     An amplitude-exponential model rho(d) = A * exp(-d/L) is fitted.  Only the
     fraction A of the residual variance is spatially correlated; the rest is
     independent and stays on the diagonal.

Outputs written to  <run_root>/inversion_data/so_residual_error_method/:
  sk_residual_error_{start}_{end}.npz              -- local variance + fit params + anomalies
  so_residual_error_correlation_{start}_{end}.npz  -- spatial correlation fit parameters
  so_diagonal_diagnostics.png                      -- 3-panel So diagonal figure
  so_correlation_diagnostics.png                   -- 3-panel correlation figure

By default this also overwrites <run_root>/inversion_data/so/so_{start}_{end}.npz
with the residual-error diagonal So (key so_{obs_error_name}).  Use
--no-overwrite-merged-so for sensitivity tests that should leave merged inversion
products untouched.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

try:
    from src.utilities.config_utils import load_config
except ModuleNotFoundError:
    try:
        from config_utils import load_config
    except ModuleNotFoundError:
        load_config = None


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build residual-error So (diagonal + spatial correlation) "
                    "from merged monthly TROPOMI observations."
    )
    p.add_argument("--start", required=True, help="Start date YYYYMMDD")
    p.add_argument("--end", required=True, help="End date YYYYMMDD")
    p.add_argument("--run-root", required=True, help="IMI run root directory")
    p.add_argument("--state-vector", required=True, help="Path to StateVector.nc")
    p.add_argument("--obs-error-name", default="15.0",
                   help="obs_err key suffix to write into so_{start}_{end}.npz (default 15.0)")
    # Diagonal So fitting parameters (fallback if curve_fit fails)
    p.add_argument("--r-retrieval", type=float, default=0.23)
    p.add_argument("--sigma-retrieval", type=float, default=13.30)
    p.add_argument("--sigma-transport", type=float, default=4.13)
    p.add_argument("--floor-variance", type=float, default=50.0,
                   help="Minimum So variance in ppb^2 (default 50)")
    # Correlation estimation parameters
    p.add_argument("--max-pairs-per-month", type=int, default=250_000)
    p.add_argument("--max-distance-km", type=float, default=1_500.0)
    p.add_argument("--distance-bin-km", type=float, default=50.0)
    p.add_argument("--min-pair-count", type=int, default=500,
                   help="Minimum pairs per bin to include in correlation fit")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--no-overwrite-merged-so", action="store_true",
                   help="Write residual So products only under so_residual_error_method; "
                        "do not update inversion_data/so/so_*.npz")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Observation file discovery
# ---------------------------------------------------------------------------

def _parse_obs_dates(path):
    stem = os.path.basename(path).replace(".npz", "")
    parts = stem.split("_")
    return parts[-2], parts[-1]


def find_observation_files(run_root, start, end):
    obs_dir = os.path.join(run_root, "inversion_data", "observations")
    files = []
    if os.path.isdir(obs_dir):
        for name in sorted(os.listdir(obs_dir)):
            if not (name.startswith("observations_") and name.endswith(".npz")):
                continue
            path = os.path.join(obs_dir, name)
            f_start, f_end = _parse_obs_dates(path)
            if f_start >= start and f_end <= end:
                files.append(path)
    if not files:
        single = os.path.join(run_root, "inversion", "obs_metadata.npz")
        if os.path.exists(single):
            files.append(single)
    if not files:
        raise FileNotFoundError(
            f"No merged observation metadata files found in {obs_dir} for {start} -> {end}"
        )
    return files


def load_observations(files, start, end):
    """Load all monthly observation files into concatenated arrays."""
    chunks = []
    slices = []
    offset = 0
    for path in files:
        with np.load(path) as obs:
            chunk = {
                "y": np.asarray(obs["obs_tropomi"], dtype=float),
                "prior": np.asarray(obs["gc_ch4_prior"], dtype=float),
                "lat": np.asarray(obs["lat"], dtype=float),
                "lon": np.asarray(obs["lon"], dtype=float),
                "counts": np.asarray(obs["observation_count"], dtype=float),
            }
        chunk_start, chunk_end = start, end
        if os.path.basename(path).startswith("observations_"):
            chunk_start, chunk_end = _parse_obs_dates(path)
        n = chunk["y"].size
        chunks.append(chunk)
        slices.append((chunk_start, chunk_end, offset, offset + n))
        offset += n

    y = np.concatenate([c["y"] for c in chunks])
    prior = np.concatenate([c["prior"] for c in chunks])
    lat = np.concatenate([c["lat"] for c in chunks])
    lon = np.concatenate([c["lon"] for c in chunks])
    counts = np.concatenate([c["counts"] for c in chunks])
    return y, prior, lat, lon, counts, slices


# ---------------------------------------------------------------------------
# Individual pixel sk from IMI visualization pkl files
# ---------------------------------------------------------------------------

def load_individual_obs_from_visualization(viz_dir):
    """Load individual (unaveraged) TROPOMI pixels from IMI data_visualization pkl files.

    jacobian.py saves per-orbit pkl files to data_visualization/ using the
    unaveraged apply_tropomi_operator.  Each file's obs_GC array has 6 columns:
      [0] actual TROPOMI XCH4 (ppb)
      [1] GC virtual XCH4 using TROPOMI averaging kernel (ppb)
      [2] longitude
      [3] latitude
      [4] iSat (pixel index, not used here)
      [5] jSat (pixel index, not used here)

    The averaged operator (data_converted pkl files) produces 5-column arrays;
    those are skipped automatically by the column-count check.

    Returns (tropomi, gc_ch4, lat, lon) float64 arrays concatenated across all
    valid orbit files, or None if viz_dir is empty or does not exist.
    """
    import glob
    import pickle

    if not os.path.isdir(viz_dir):
        return None

    pkl_files = sorted(glob.glob(os.path.join(viz_dir, "*_GCtoTROPOMI.pkl")))
    if not pkl_files:
        return None

    trop_list, gc_list, lat_list, lon_list = [], [], [], []
    n_loaded = 0
    for fp in pkl_files:
        try:
            with open(fp, "rb") as fh:
                obj = pickle.load(fh)
            obs_gc = np.asarray(obj["obs_GC"], dtype=float)
        except Exception as e:
            print(f"  Warning: cannot read {os.path.basename(fp)}: {e}")
            continue
        # Individual-pixel pkl files have 6 columns; averaged have 5 — skip averaged.
        if obs_gc.ndim != 2 or obs_gc.shape[1] < 6 or obs_gc.shape[0] == 0:
            continue
        valid = (
            np.isfinite(obs_gc[:, 0])
            & np.isfinite(obs_gc[:, 1])
            & np.isfinite(obs_gc[:, 2])
            & np.isfinite(obs_gc[:, 3])
        )
        if not np.any(valid):
            continue
        trop_list.append(obs_gc[valid, 0])
        gc_list.append(obs_gc[valid, 1])
        lon_list.append(obs_gc[valid, 2])
        lat_list.append(obs_gc[valid, 3])
        n_loaded += 1

    if not trop_list:
        return None

    print(f"  Loaded individual pixels from {n_loaded}/{len(pkl_files)} visualization pkl files")
    return (
        np.concatenate(trop_list),
        np.concatenate(gc_list),
        np.concatenate(lat_list),
        np.concatenate(lon_list),
    )


def compute_sk_from_pixels(tropomi_ind, gc_ch4_ind, lat_ind, lon_ind,
                            state_vector_path, global_var):
    """Compute within-cell sk from individual TROPOMI pixel residuals.

    For each GC grid cell, accumulates all individual pixels across all available
    orbit files, then computes the variance of demeaned (TROPOMI - GC_virtual)
    residuals.  Cells with fewer than 3 pixels are excluded.

    The GC_virtual column from the visualization pkl already applies the TROPOMI
    averaging kernel to the GC simulation, so no separate prior approximation
    is needed.

    Returns (sk_lookup, gclat, gclon) where sk_lookup is a dict {cell_id: variance}.
    """
    from collections import defaultdict

    state = xr.load_dataset(state_vector_path)
    gclat = state["lat"].values
    gclon = state["lon"].values
    elat = abs(float(gclat[1] - gclat[0])) / 2.0
    elon = abs(float(gclon[1] - gclon[0])) / 2.0

    li = np.searchsorted(gclat, lat_ind)
    li = np.clip(li, 1, len(gclat) - 1)
    li -= np.abs(lat_ind - gclat[li - 1]) <= np.abs(lat_ind - gclat[li])
    lni = np.searchsorted(gclon, lon_ind)
    lni = np.clip(lni, 1, len(gclon) - 1)
    lni -= np.abs(lon_ind - gclon[lni - 1]) <= np.abs(lon_ind - gclon[lni])

    in_b = (
        (np.abs(lat_ind - gclat[li]) <= elat * 1.01)
        & (np.abs(lon_ind - gclon[lni]) <= elon * 1.01)
    )
    cell_id_pix = li[in_b].astype(np.int64) * len(gclon) + lni[in_b].astype(np.int64)
    resid_pix = (tropomi_ind - gc_ch4_ind)[in_b]

    cell_resid = defaultdict(list)
    for cid, r in zip(cell_id_pix, resid_pix):
        if np.isfinite(r):
            cell_resid[cid].append(r)

    sk_lookup = {}
    for cid, resids in cell_resid.items():
        if len(resids) < 3:
            continue
        r = np.asarray(resids)
        sk_lookup[cid] = float(np.var(r - r.mean()))

    n_pix = int(in_b.sum())
    print(f"  sk (individual pixels): {n_pix:,} valid pixels → "
          f"{len(sk_lookup)} grid cells with variance estimates")
    return sk_lookup, gclat, gclon


# ---------------------------------------------------------------------------
# Diagonal So: residual error method
# ---------------------------------------------------------------------------

def residual_count_variance(p, r_retrieval, sigma_retrieval, sigma_transport):
    """Chen et al. 2023 superobservation count-variance model."""
    p = np.where(np.asarray(p, dtype=float) < 1, 1, p)
    return sigma_transport**2 + sigma_retrieval**2 * ((1.0 - r_retrieval) / p + r_retrieval)


def estimate_local_variance(y, prior, lat, lon, state_vector_path):
    """Estimate per-observation local residual variance from the multi-year distribution.

    For each GEOS-Chem grid cell, variance is computed from the time-series of
    (y - prior) residuals assigned to that cell.  Observations with too few
    neighbours fall back to the global residual variance.

    Returns (sk_est, anomalies, global_var).
    """
    state = xr.load_dataset(state_vector_path)
    gclat = state["lat"].values
    gclon = state["lon"].values
    elat = abs(float(gclat[1] - gclat[0])) / 2.0
    elon = abs(float(gclon[1] - gclon[0])) / 2.0

    residual = y - prior
    global_var = float(np.nanvar(residual - np.nanmean(residual)))
    if not np.isfinite(global_var) or global_var <= 0:
        global_var = 176.89  # fallback: (13.3 ppb)^2

    lat_idx = np.searchsorted(gclat, lat)
    lat_idx = np.clip(lat_idx, 1, len(gclat) - 1)
    lat_idx -= np.abs(lat - gclat[lat_idx - 1]) <= np.abs(lat - gclat[lat_idx])
    lon_idx = np.searchsorted(gclon, lon)
    lon_idx = np.clip(lon_idx, 1, len(gclon) - 1)
    lon_idx -= np.abs(lon - gclon[lon_idx - 1]) <= np.abs(lon - gclon[lon_idx])
    in_cell = (
        (np.abs(lat - gclat[lat_idx]) <= elat * 1.01)
        & (np.abs(lon - gclon[lon_idx]) <= elon * 1.01)
    )

    cell_id = lat_idx.astype(np.int64) * len(gclon) + lon_idx.astype(np.int64)
    sk_est = np.full(y.shape, global_var, dtype=float)
    anomalies = np.full(y.shape, np.nan, dtype=float)

    for cid in np.unique(cell_id[in_cell]):
        idx = np.where((cell_id == cid) & in_cell)[0]
        if idx.size >= 2:
            cell_anom = residual[idx] - np.nanmean(residual[idx])
            cell_var = np.nanvar(cell_anom)
            anomalies[idx] = cell_anom
            sk_est[idx] = cell_var if np.isfinite(cell_var) and cell_var > 0 else global_var
        elif idx.size == 1:
            anomalies[idx] = 0.0

    return sk_est, anomalies, global_var


def fit_count_variance(counts, anomalies, fallback):
    """Fit the Chen et al. count-variance model to the binned residual anomalies."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return fallback

    p_rounded = np.round(np.asarray(counts, dtype=float))
    unique_p, variances = [], []
    for p in np.unique(p_rounded[np.isfinite(p_rounded)]):
        idx = p_rounded == p
        if np.count_nonzero(idx) >= 20:
            var = np.nanvar(anomalies[idx])
            if np.isfinite(var) and var > 0:
                unique_p.append(p)
                variances.append(var)
    unique_p = np.asarray(unique_p, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if unique_p.size < 3:
        return fallback

    try:
        popt, _ = curve_fit(
            residual_count_variance,
            unique_p,
            variances,
            bounds=([0.0, 0.0, 0.0], [1.0, np.inf, np.inf]),
            p0=fallback,
            maxfev=20_000,
        )
        return tuple(float(x) for x in popt)
    except Exception as exc:
        print(f"Count-variance fit failed ({exc}); using fallback parameters {fallback}")
        return fallback


def build_diagonal_so(sk_est, counts, r_retrieval, sigma_retrieval, sigma_transport, floor_variance):
    """Combine local spatial variance with count scaling to form diagonal So."""
    gp = residual_count_variance(counts, r_retrieval, sigma_retrieval, sigma_transport)
    gp = gp / np.nanmax(gp)
    so = sk_est * gp
    so[~np.isfinite(so)] = float(np.nanmean(so[np.isfinite(so)]))
    so = np.maximum(so, floor_variance).astype(np.float32)
    return so


# ---------------------------------------------------------------------------
# Spatial correlation estimation
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two arrays of points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def estimate_residual_correlation(
    lat, lon, anomalies, slices,
    max_pairs_per_month, max_distance_km, distance_bin_km, random_seed,
):
    """Estimate binned residual-anomaly correlation as a function of separation distance.

    Pairs are sampled randomly within each month to avoid temporal correlations
    contaminating the spatial signal.

    Returns (distance_centers_km, corr_by_distance, pair_count_by_bin).
    """
    rng = np.random.default_rng(random_seed)
    bins = np.arange(0.0, max_distance_km + distance_bin_km, distance_bin_km)
    centers = 0.5 * (bins[:-1] + bins[1:])
    sum_prod = np.zeros(centers.size, dtype=float)
    count_prod = np.zeros(centers.size, dtype=int)

    for _, _, i0, i1 in slices:
        n = i1 - i0
        if n < 2:
            continue
        monthly_anom = anomalies[i0:i1].copy()
        finite_month = np.isfinite(monthly_anom)
        if np.count_nonzero(finite_month) < 2:
            continue
        monthly_anom[finite_month] -= np.nanmean(monthly_anom[finite_month])
        n_pairs = min(max_pairs_per_month, n * (n - 1) // 2)
        ii = rng.integers(0, n, size=n_pairs)
        jj = rng.integers(0, n - 1, size=n_pairs)
        jj = jj + (jj >= ii)
        d = haversine_km(lat[ii + i0], lon[ii + i0], lat[jj + i0], lon[jj + i0])
        prod = monthly_anom[ii] * monthly_anom[jj]
        valid = np.isfinite(d) & np.isfinite(prod) & (d <= max_distance_km)
        bin_idx = np.digitize(d[valid], bins) - 1
        valid_bin = (bin_idx >= 0) & (bin_idx < centers.size)
        for b in np.unique(bin_idx[valid_bin]):
            mask = valid_bin & (bin_idx == b)
            sum_prod[b] += np.nansum(prod[valid][mask])
            count_prod[b] += int(np.count_nonzero(mask))

    demeaned = []
    for _, _, i0, i1 in slices:
        monthly_anom = anomalies[i0:i1]
        finite = np.isfinite(monthly_anom)
        if np.count_nonzero(finite) >= 2:
            demeaned.append(monthly_anom[finite] - np.nanmean(monthly_anom[finite]))
    anom_var = float(np.nanvar(np.concatenate(demeaned))) if demeaned else float(np.nanvar(anomalies))
    corr = np.full(centers.size, np.nan)
    ok = count_prod > 0
    corr[ok] = sum_prod[ok] / count_prod[ok] / anom_var
    return centers, corr, count_prod


def fit_correlation_model(distance_centers, corr_by_distance, pair_counts, min_pair_count):
    """Fit amplitude-exponential and amplitude-Gaussian models to the empirical correlation.

    Returns a dict with fitted parameters for both models.
    The amplitude A < 1 is the fraction of residual variance that is spatially correlated.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    fit_mask = (
        np.isfinite(corr_by_distance)
        & (pair_counts >= min_pair_count)
        & (distance_centers > 0)
        & (distance_centers <= 900.0)
        & (corr_by_distance > 0)
    )
    if fit_mask.sum() < 3:
        return None

    x = distance_centers[fit_mask]
    y = corr_by_distance[fit_mask]
    w = np.sqrt(pair_counts[fit_mask].astype(float))

    results = {}
    for name, func in [
        ("exponential", lambda d, A, L: A * np.exp(-d / L)),
        ("gaussian", lambda d, A, L: A * np.exp(-(d / L)**2)),
    ]:
        try:
            popt, _ = curve_fit(
                func, x, y,
                sigma=1.0 / np.maximum(w, 1.0),
                absolute_sigma=False,
                p0=(0.12, 400.0),
                bounds=([0.0, 10.0], [1.0, 3000.0]),
                maxfev=20_000,
            )
            results[name] = {"amplitude": float(popt[0]), "length_km": float(popt[1])}
        except Exception as exc:
            print(f"  {name} correlation fit failed: {exc}")

    # Two-exponential model A1 e^{-d/L1} + A2 e^{-d/L2}: a short near-field term plus a
    # broad same-day transport tail.  A single exponential cannot fit both regimes (its
    # (A, L) slides with the fit window), so this two-component form is the one used for
    # the off-diagonal So.  Fit over a wider range than the single-exponential so the
    # transport tail is captured rather than truncated.
    tail_mask = (
        np.isfinite(corr_by_distance)
        & (pair_counts >= min_pair_count)
        & (distance_centers > 0)
        & (distance_centers <= 1300.0)
        & (corr_by_distance > 0)
    )
    if tail_mask.sum() >= 4:
        xt = distance_centers[tail_mask]
        yt = corr_by_distance[tail_mask]
        wt = np.sqrt(pair_counts[tail_mask].astype(float))
        try:
            popt2, _ = curve_fit(
                lambda d, A1, L1, A2, L2: A1 * np.exp(-d / L1) + A2 * np.exp(-d / L2),
                xt, yt,
                sigma=1.0 / np.maximum(wt, 1.0),
                absolute_sigma=False,
                p0=(0.385, 26.0, 0.459, 398.0),
                bounds=([0.0, 5.0, 0.0, 100.0], [1.0, 100.0, 1.0, 2000.0]),
                maxfev=40_000,
            )
            A1, L1, A2, L2 = (float(v) for v in popt2)
            if L1 > L2:  # keep term 1 as the short-range term
                A1, L1, A2, L2 = A2, L2, A1, L1
            results["two_exponential"] = {
                "amplitude1": A1, "length1_km": L1,
                "amplitude2": A2, "length2_km": L2,
            }
            print(
                f"  two-exponential correlation fit: A1={A1:.3f} L1={L1:.0f} km, "
                f"A2={A2:.3f} L2={L2:.0f} km"
            )
        except Exception as exc:
            print(f"  two-exponential correlation fit failed: {exc}")

    # Also fit a forced unit-amplitude exponential for the before/after comparison plot
    try:
        popt_forced, _ = curve_fit(
            lambda d, L: np.exp(-d / L), x, np.clip(y, 0, 1),
            sigma=1.0 / np.maximum(w, 1.0),
            absolute_sigma=False,
            p0=(250.0,),
            bounds=([10.0], [3000.0]),
            maxfev=20_000,
        )
        results["forced_exponential"] = {"amplitude": 1.0, "length_km": float(popt_forced[0])}
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diagonal_diagnostics(
    y, prior, counts, anomalies, so, sk_est,
    r_retrieval, sigma_retrieval, sigma_transport,
    lat, lon, out_path,
):
    """3-panel figure for the diagonal So construction."""
    residual = y - prior
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: residual histogram
    ax = axes[0]
    r_finite = residual[np.isfinite(residual)]
    ax.hist(r_finite, bins=120, range=(-120, 120), color="0.35", density=True)
    mu, sigma_r = np.nanmean(r_finite), np.nanstd(r_finite)
    xg = np.linspace(-120, 120, 400)
    ax.plot(xg, np.exp(-0.5 * ((xg - mu) / sigma_r)**2) / (sigma_r * np.sqrt(2 * np.pi)),
            color="tab:red", lw=2, label=f"N({mu:.1f}, {sigma_r:.1f}²)")
    ax.axvline(mu, color="tab:red", lw=1.5, ls="--")
    ax.set_xlabel("TROPOMI − prior (ppb)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Residual distribution")
    ax.legend(fontsize=9)

    # Panel 2: count-variance fit
    ax = axes[1]
    p_rounded = np.round(counts.astype(float))
    rows = [
        (p, np.nanvar(anomalies[p_rounded == p]))
        for p in np.unique(p_rounded[np.isfinite(p_rounded)])
        if np.count_nonzero(p_rounded == p) >= 20
    ]
    if rows:
        pp = np.array([r[0] for r in rows])
        vv = np.array([r[1] for r in rows])
        valid_rows = np.isfinite(vv) & (vv > 0)
        size = np.clip(np.sqrt(np.array([np.count_nonzero(p_rounded == p) for p in pp[valid_rows]])), 3, 25)
        ax.scatter(pp[valid_rows], np.sqrt(vv[valid_rows]), s=size, alpha=0.65, label="Binned anomalies")
        p_grid = np.logspace(0, np.log10(max(pp) + 1), 400)
        ax.plot(p_grid, np.sqrt(residual_count_variance(p_grid, r_retrieval, sigma_retrieval, sigma_transport)),
                color="tab:red", lw=2.5, label="Fit")
        ax.set_xscale("log")
    ax.set_xlabel("Pixel count P")
    ax.set_ylabel("Residual anomaly std (ppb)")
    ax.set_title("(b) Count-variance fit")
    ax.legend(fontsize=9)

    # Panel 3: map of sqrt(So)
    ax = axes[2]
    vmax = float(np.nanpercentile(np.sqrt(so), 99))
    sc = ax.scatter(lon, lat, c=np.sqrt(so), s=1, cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
    plt.colorbar(sc, ax=ax, label="sqrt(So) (ppb)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("(c) Diagonal So std")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_correlation_diagnostics(
    distance_centers, corr_by_distance, pair_counts, fit_results, out_path,
    max_distance_km=1500.0,
):
    """3-panel figure for the spatial residual-error correlation."""
    d_fit = np.linspace(0, max_distance_km, 400)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax in axes[:2]:
        ax.axhline(0, color="0.75", lw=1)
        ax.scatter(
            distance_centers, corr_by_distance,
            s=np.clip(np.sqrt(np.maximum(pair_counts, 0)), 3, 25),
            alpha=0.6, color="0.35", label="Binned pairs",
        )
        ax.set_xlabel("Separation distance (km)")

    # Panel 1: before — forced ρ(0)=1 exponential
    ax = axes[0]
    if fit_results and "forced_exponential" in fit_results:
        L = fit_results["forced_exponential"]["length_km"]
        ax.plot(d_fit, np.exp(-d_fit / L), color="tab:red", lw=2.5,
                label=f"exp(−d/L),  L = {L:.0f} km")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Residual-error correlation")
    ax.set_title("(a) Standard model  [ρ(0) = 1 forced]")
    ax.legend(fontsize=9)

    # Panel 2: after — amplitude fit
    ax = axes[1]
    if fit_results:
        if "exponential" in fit_results:
            A, L = fit_results["exponential"]["amplitude"], fit_results["exponential"]["length_km"]
            ax.plot(d_fit, A * np.exp(-d_fit / L), color="tab:red", lw=2.5,
                    label=f"A exp(−d/L),  A = {A:.2f},  L = {L:.0f} km")
        if "gaussian" in fit_results:
            A, L = fit_results["gaussian"]["amplitude"], fit_results["gaussian"]["length_km"]
            ax.plot(d_fit, A * np.exp(-(d_fit / L)**2), color="tab:orange", lw=2, ls="--",
                    label=f"A exp(−(d/L)²),  A = {A:.2f},  L = {L:.0f} km")
    ylim_top = max(0.18, float(np.nanmax(corr_by_distance[np.isfinite(corr_by_distance)])) * 1.25)
    ax.set_ylim(-0.05, ylim_top)
    ax.set_ylabel("Residual-error correlation")
    ax.set_title("(b) Amplitude fit  [ρ(0) = A < 1]")
    ax.legend(fontsize=9)

    # Panel 3: what goes into So off-diagonal
    ax = axes[2]
    ax.axhline(0, color="0.75", lw=1)
    if fit_results and "exponential" in fit_results:
        A, L = fit_results["exponential"]["amplitude"], fit_results["exponential"]["length_km"]
        y_curve = A * np.exp(-d_fit / L)
        ax.plot(d_fit, y_curve, color="tab:red", lw=2.5,
                label=f"ρ(d) used for off-diagonal So\nA = {A:.2f},  L = {L:.0f} km")
        ax.fill_between(d_fit, 0, y_curve, color="tab:red", alpha=0.15)
        # Annotate amplitude interpretation
        ax.annotate(
            f"{A*100:.0f}% of obs-error variance\nis spatially correlated",
            xy=(0, A), xytext=(300, A * 0.6),
            arrowprops=dict(arrowstyle="->", color="0.4"),
            fontsize=9,
        )
    ax.set_ylim(-0.01, ylim_top)
    ax.set_xlabel("Separation distance (km)")
    ax.set_ylabel("ρ(d)")
    ax.set_title("(c) Off-diagonal So correlation function")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_root = args.run_root
    start, end = args.start, args.end
    rng_seed = args.random_seed

    print(f"\n=== build_residual_obs_covariance.py  {start} -> {end} ===")

    # ------------------------------------------------------------------
    # Load merged monthly observations
    # ------------------------------------------------------------------
    files = find_observation_files(run_root, start, end)
    print(f"Found {len(files)} monthly observation file(s)")
    y, prior, lat, lon, counts, slices = load_observations(files, start, end)
    print(f"Total superobservations: {y.size:,}")

    # ------------------------------------------------------------------
    # Diagonal So: local spatial variance + count scaling
    # ------------------------------------------------------------------
    # Step 1: temporal superobservation variance (always computed — used for
    # anomalies → count-variance fit, and as fallback where pixels are sparse).
    print("Estimating local spatial residual variance ...")
    sk_est, anomalies, global_var = estimate_local_variance(
        y, prior, lat, lon, args.state_vector
    )
    print(f"  Global residual std: {np.sqrt(global_var):.2f} ppb")

    # Step 2: override sk with individual-pixel within-cell variance when the
    # IMI visualization pkl files are present in {run_root}/inversion/data_visualization/.
    # Those pkl files are written by jacobian.py using apply_tropomi_operator
    # (unaveraged) and have the GC virtual XCH4 already computed with the
    # TROPOMI averaging kernel — no prior approximation needed.
    viz_dir = os.path.join(run_root, "inversion", "data_visualization")
    ind_result = load_individual_obs_from_visualization(viz_dir)
    if ind_result is not None:
        tropomi_ind, gc_ch4_ind, lat_ind, lon_ind = ind_result
        print(f"  Using {len(tropomi_ind):,} individual pixels for sk "
              f"(from {viz_dir})")
        sk_lookup, gclat_sv, gclon_sv = compute_sk_from_pixels(
            tropomi_ind, gc_ch4_ind, lat_ind, lon_ind, args.state_vector, global_var
        )
        # Map per-cell pixel sk onto per-observation sk_est
        state = xr.load_dataset(args.state_vector)
        elat = abs(float(state["lat"].values[1] - state["lat"].values[0])) / 2.0
        elon = abs(float(state["lon"].values[1] - state["lon"].values[0])) / 2.0
        lat_idx_sv = np.searchsorted(gclat_sv, lat)
        lat_idx_sv = np.clip(lat_idx_sv, 1, len(gclat_sv) - 1)
        lat_idx_sv -= np.abs(lat - gclat_sv[lat_idx_sv - 1]) <= np.abs(lat - gclat_sv[lat_idx_sv])
        lon_idx_sv = np.searchsorted(gclon_sv, lon)
        lon_idx_sv = np.clip(lon_idx_sv, 1, len(gclon_sv) - 1)
        lon_idx_sv -= np.abs(lon - gclon_sv[lon_idx_sv - 1]) <= np.abs(lon - gclon_sv[lon_idx_sv])
        cell_id_sv = lat_idx_sv.astype(np.int64) * len(gclon_sv) + lon_idx_sv.astype(np.int64)
        n_overridden = 0
        for i, cid in enumerate(cell_id_sv):
            if cid in sk_lookup:
                sk_est[i] = sk_lookup[cid]
                n_overridden += 1
        frac = n_overridden / max(1, len(sk_est))
        print(f"  sk overridden for {n_overridden}/{len(sk_est)} obs ({frac:.1%}) "
              f"using individual pixels; remainder uses temporal fallback")
    else:
        print(f"  No visualization pkl files found in {viz_dir}; "
              f"using temporal superobservation variance for sk")

    fallback = (args.r_retrieval, args.sigma_retrieval, args.sigma_transport)
    print("Fitting count-variance model ...")
    r_ret, sig_ret, sig_tra = fit_count_variance(counts, anomalies, fallback)
    print(f"  r_retrieval={r_ret:.4f}, sigma_retrieval={sig_ret:.4f}, sigma_transport={sig_tra:.4f}")

    so = build_diagonal_so(sk_est, counts, r_ret, sig_ret, sig_tra, args.floor_variance)
    print(f"  So range: {np.nanmin(so):.1f} – {np.nanmax(so):.1f} ppb²  "
          f"(mean sqrt = {np.sqrt(so).mean():.2f} ppb,  floor fraction = {np.mean(so == args.floor_variance):.2f})")

    # ------------------------------------------------------------------
    # Spatial correlation of residual anomalies
    # ------------------------------------------------------------------
    print("Estimating residual-anomaly spatial correlation ...")
    dist_centers, corr_vals, pair_counts = estimate_residual_correlation(
        lat, lon, anomalies, slices,
        args.max_pairs_per_month, args.max_distance_km, args.distance_bin_km, rng_seed,
    )
    print(f"  Total sampled pairs: {pair_counts.sum():,}")

    print("Fitting correlation models ...")
    fit_results = fit_correlation_model(dist_centers, corr_vals, pair_counts, args.min_pair_count)
    if fit_results:
        for name, params in fit_results.items():
            print(f"  {name}: A={params['amplitude']:.3f}, L={params['length_km']:.1f} km")
    else:
        print("  Warning: correlation fitting failed; off-diagonal So will not be saved")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    residual_dir = Path(run_root) / "inversion_data" / "so_residual_error_method"
    so_dir = Path(run_root) / "inversion_data" / "so"
    residual_dir.mkdir(parents=True, exist_ok=True)
    so_dir.mkdir(parents=True, exist_ok=True)

    # Fit diagnostics + anomalies
    np.savez(
        residual_dir / f"sk_residual_error_{start}_{end}.npz",
        sk_by_observation=sk_est.astype(np.float32),
        residual_anomaly=anomalies.astype(np.float32),
        r_retrieval=np.float32(r_ret),
        sigma_retrieval=np.float32(sig_ret),
        sigma_transport=np.float32(sig_tra),
        source_files=np.asarray(files),
    )

    # Spatial correlation parameters
    # Off-diagonal So correlation model.  When the two-exponential fit (short near-field
    # term + broad same-day transport tail) is available it is the model invert.py applies,
    # with NO taper and a cutoff of 3*L2 (captures ~95% of the correlation mass, including
    # the transport tail that a short cutoff would discard).  The smoothed empirical lookup
    # and the single-exponential fit are kept as fallbacks / diagnostics.  The inversion
    # code still checks the sparse correlation operator for positive semidefiniteness.
    EMPIRICAL_CUTOFF_KM = 375.0  # fallback cutoff for the empirical lookup only
    two_exp = fit_results.get("two_exponential") if fit_results else None
    cutoff_km = float(3.0 * two_exp["length2_km"]) if two_exp is not None else EMPIRICAL_CUTOFF_KM

    corr_smooth = corr_vals.copy()
    half = 1  # half-window for 3-bin rolling median
    for b in range(len(corr_vals)):
        window = corr_vals[max(0, b - half): b + half + 1]
        finite = window[np.isfinite(window)]
        corr_smooth[b] = float(np.median(finite)) if finite.size else np.nan
    corr_smooth = np.clip(corr_smooth, 0.0, None)
    in_cutoff = dist_centers <= cutoff_km
    emp_d = dist_centers[in_cutoff]
    emp_rho = corr_smooth[in_cutoff]

    # keys common to every save branch
    common = dict(
        empirical_d_km=emp_d.astype(np.float32),
        empirical_rho=emp_rho.astype(np.float32),
        empirical_cutoff_km=np.float32(cutoff_km),
        corr_cutoff_km=np.float32(cutoff_km),
        distance_bin_centers_km=dist_centers.astype(np.float32),
        empirical_correlation=corr_vals.astype(np.float32),
        pair_count=pair_counts.astype(np.int64),
    )
    corr_path = residual_dir / f"so_residual_error_correlation_{start}_{end}.npz"

    if two_exp is not None:
        np.savez(
            corr_path,
            functional_form=np.bytes_("two_exponential"),
            corr_amplitude1=np.float32(two_exp["amplitude1"]),
            corr_length1_km=np.float32(two_exp["length1_km"]),
            corr_amplitude2=np.float32(two_exp["amplitude2"]),
            corr_length2_km=np.float32(two_exp["length2_km"]),
            # single-exponential kept as diagnostic / fallback
            corr_amplitude=np.float32(fit_results.get("exponential", {}).get("amplitude", np.nan)),
            corr_length_km=np.float32(fit_results.get("exponential", {}).get("length_km", np.nan)),
            **common,
        )
        print(f"  Saved two-exponential off-diagonal So: "
              f"A1={two_exp['amplitude1']:.3f} L1={two_exp['length1_km']:.0f} km, "
              f"A2={two_exp['amplitude2']:.3f} L2={two_exp['length2_km']:.0f} km, "
              f"no taper, cutoff={cutoff_km:.0f} km")
    elif fit_results and "exponential" in fit_results:
        exp_params = fit_results["exponential"]
        np.savez(
            corr_path,
            functional_form=np.bytes_("empirical"),
            corr_amplitude=np.float32(exp_params["amplitude"]),
            corr_length_km=np.float32(exp_params["length_km"]),
            gaussian_amplitude=np.float32(fit_results.get("gaussian", {}).get("amplitude", np.nan)),
            gaussian_length_km=np.float32(fit_results.get("gaussian", {}).get("length_km", np.nan)),
            **common,
        )
        print(f"  Saved empirical correlation (single-exp fallback): {len(emp_d)} bins up to "
              f"{cutoff_km:.0f} km, A={exp_params['amplitude']:.3f}, L={exp_params['length_km']:.1f} km")
    elif np.any(np.isfinite(emp_rho) & (emp_rho > 0)):
        np.savez(corr_path, functional_form=np.bytes_("empirical"), **common)
        print(f"  Saved empirical correlation (no parametric fit): {len(emp_d)} bins")
    else:
        print("  Skipping correlation params save (insufficient data)")

    # Monthly diagonal So files (optionally overwrite default so_{obs_error_name} key)
    for month_start, month_end, i0, i1 in slices:
        so_month = so[i0:i1]
        np.savez(
            residual_dir / f"so_{month_start}_{month_end}.npz",
            so=so_month,
        )
        if not args.no_overwrite_merged_so:
            # Overwrite the merged so file so invert.py picks up the residual-error So
            so_npz_path = so_dir / f"so_{month_start}_{month_end}.npz"
            existing = {}
            if so_npz_path.exists():
                with np.load(so_npz_path) as f:
                    existing = {k: f[k] for k in f.files}
            existing[f"so_{args.obs_error_name}"] = so_month
            np.savez(so_npz_path, **existing)
        print(f"  {month_start} -> {month_end}: n={so_month.size:,}, "
              f"mean sqrt(So)={np.sqrt(so_month).mean():.2f} ppb"
              f"{' (merged so unchanged)' if args.no_overwrite_merged_so else ''}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    print("Writing diagnostic plots ...")
    plot_diagonal_diagnostics(
        y, prior, counts, anomalies, so, sk_est,
        r_ret, sig_ret, sig_tra, lat, lon,
        out_path=residual_dir / f"so_diagonal_diagnostics_{start}_{end}.png",
    )
    plot_correlation_diagnostics(
        dist_centers, corr_vals, pair_counts, fit_results,
        out_path=residual_dir / f"so_correlation_diagnostics_{start}_{end}.png",
        max_distance_km=args.max_distance_km,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
