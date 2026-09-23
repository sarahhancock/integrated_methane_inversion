#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import re
import datetime
import numpy as np
import xarray as xr
from itertools import product
from pathlib import Path
from collections import defaultdict, deque
try:
    import scipy.optimize as _opt
    _SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    _SCIPY_OPTIMIZE_AVAILABLE = False
from src.inversion_scripts.utils import (
    load_obj,
    calculate_superobservation_error,
    ensure_float_list,
    get_mean_emissions,
    update_prior_error_for_OptimizeSoil,
    map_files_to_reference,
)
from src.inversion_scripts.softplus_invert import (
    run_softplus,
    inversion_diagnostics,
)
from src.utilities.config_utils import load_config


# Softplus smoothing scale (x = s*log(1+e^{z/s})); 0.1 is near-ReLU and matches production.
SOFTPLUS_SCALE_DEFAULT = 0.1
# Levenberg-Marquardt damping for the positivity solvers (Chen et al., 2022).
POSITIVITY_KAPPA = 10.0


def resolve_inversion_method(config):
    """Resolve the analytical-path inversion solver from config.

    `SoftplusErrors: true` is the documented flag that selects the softplus positivity solver
    (parallel to `LognormalErrors: true`). `InversionMethod: softplus` is still honored for
    back-compat. Returns "analytical" (default) or "softplus". Lognormal is handled separately
    via `LognormalErrors` in run_inversion.sh before invert.py is called.
    """
    method = str(config.get("InversionMethod", "analytical"))
    if bool(config.get("SoftplusErrors", False)):
        method = "softplus"
    return method


def align_obs_rows_with_reference(obs_GC, obs_GC_ref):
    """Match target observations to reference rows using shared metadata columns (lat, lon, obs_count)."""
    ncols = min(obs_GC.shape[1], obs_GC_ref.shape[1])

    def make_key(row):
        # Ignore the leading observed/model xCH4 columns and match on shared metadata.
        return tuple(np.round(row[2:ncols], decimals=6))

    ref_lookup = defaultdict(deque)
    for idx, row in enumerate(obs_GC_ref):
        ref_lookup[make_key(row)].append(idx)

    obs_indices = []
    ref_indices = []
    for idx, row in enumerate(obs_GC):
        key = make_key(row)
        if ref_lookup[key]:
            obs_indices.append(idx)
            ref_indices.append(ref_lookup[key].popleft())

    return np.asarray(obs_indices, dtype=int), np.asarray(ref_indices, dtype=int)

def get_prior_sigma_vector(
    n_elements,
    prior_err,
    OptimizeSoil=False,
    prior_ds=None,
    StateVectorFile=None,
):
    """Return the prior standard deviation for each state-vector element."""
    sigma = np.full(n_elements, prior_err, dtype=float)
    if OptimizeSoil:
        prior_err_new = update_prior_error_for_OptimizeSoil(
            prior_ds, prior_err, StateVectorFile, n_elements
        )
        sigma[: len(prior_err_new)] = prior_err_new
    return sigma


def get_expected_state_vector_ids(StateVectorFile):
    """Return the sorted state-vector IDs defined in the active state-vector file."""
    state_vector = xr.load_dataset(StateVectorFile)
    state_vector_ids = state_vector["StateVector"].values.reshape(-1)
    state_vector_ids = state_vector_ids[np.isfinite(state_vector_ids)]
    state_vector_ids = state_vector_ids[state_vector_ids > 0]
    state_vector_ids = np.unique(state_vector_ids.astype(np.int32))
    return np.sort(state_vector_ids)


def build_prior_covariance(
    n_elements,
    prior_err,
    OptimizeSoil=False,
    prior_ds=None,
    StateVectorFile=None,
    prebuilt_prior_err_covariance=False,
):
    """
    Build the prior covariance in either full or diagonal form.

    Returns the covariance, the constraint covariance, and a boolean indicating
    whether the covariance should be treated as a full matrix downstream.
    """
    if prebuilt_prior_err_covariance:
        # Load prebuilt covariance matrix with off-diagonal elements
        Sa = np.zeros((n_elements, n_elements), dtype=float)
        covariance_path = Path("prior_norm_error_covariance.npz")
        if not covariance_path.exists():
            raise FileNotFoundError(f"Covariance matrix file not found: {covariance_path}")
        with np.load(covariance_path) as prebuilt:
            Sa_prebuilt = prebuilt["covariance"]
            state_vector_ids_prebuilt = prebuilt["state_vector_ids"]
            # optional per-element sigma amplitude (per-sector amplitude knob from the builder)
            sigma_scale_prebuilt = (
                np.asarray(prebuilt["sigma_scale"], dtype=float)
                if "sigma_scale" in prebuilt.files
                else None
            )

        expected_state_vector_ids = get_expected_state_vector_ids(StateVectorFile)
        state_vector_ids_prebuilt = np.asarray(state_vector_ids_prebuilt, dtype=np.int32)
        if Sa_prebuilt.shape[0] != Sa_prebuilt.shape[1]:
            raise ValueError(
                f"Prior covariance must be square, got shape {Sa_prebuilt.shape}"
            )
        if Sa_prebuilt.shape[0] != state_vector_ids_prebuilt.size:
            raise ValueError(
                "Prior covariance size does not match the saved state-vector IDs: "
                f"{Sa_prebuilt.shape[0]} vs {state_vector_ids_prebuilt.size}"
            )
        if not np.array_equal(state_vector_ids_prebuilt, expected_state_vector_ids):
            raise ValueError(
                "Saved prior covariance state-vector IDs do not match the active "
                "StateVectorFile."
            )
        if Sa_prebuilt.shape[0] > n_elements:
            raise ValueError(
                "Prior covariance block is larger than the inversion state vector: "
                f"{Sa_prebuilt.shape[0]} > {n_elements}"
            )

        Sa_prebuilt_elems = Sa_prebuilt.shape[0]
        # The prebuilt matrix can define only a leading subset of the full state vector,
        # so we scale and insert it into the top-left block.
        sigma_prebuilt = get_prior_sigma_vector(
            Sa_prebuilt_elems,
            prior_err,
            OptimizeSoil=OptimizeSoil,
            prior_ds=prior_ds,
            StateVectorFile=StateVectorFile,
        )
        # Apply the optional per-element sigma amplitude (e.g. wetland 5x) written by the builder.
        if sigma_scale_prebuilt is not None:
            sigma_prebuilt = sigma_prebuilt * sigma_scale_prebuilt[:Sa_prebuilt_elems]
        # The prebuilt matrix stores only the normalized covariance structure, so we
        # apply sigma_i * sigma_j here. This reduces to a scalar prior_err**2 factor
        # when all sigmas are the same, but supports element-wise prior_err_new values.
        Sa[:Sa_prebuilt_elems, :Sa_prebuilt_elems] = (
            sigma_prebuilt[:, None] * Sa_prebuilt * sigma_prebuilt[None, :]
        )
        return Sa, Sa.copy(), True

    # Otherwise, build only a diagonal covariance matrix
    sigma = get_prior_sigma_vector(
        n_elements,
        prior_err,
        OptimizeSoil=OptimizeSoil,
        prior_ds=prior_ds,
        StateVectorFile=StateVectorFile,
    )
    Sa_diag = sigma**2
    return Sa_diag, Sa_diag.copy(), False


def apply_diagonal_prior(matrix, indices, value):
    """Write a prior variance value onto either a 1D diagonal vector or 2D matrix."""
    if np.isscalar(matrix[0]) or getattr(matrix, "ndim", 1) == 1:
        matrix[indices] = value
    else:
        # Convert slices like [-4:] into explicit diagonal positions for 2D matrices.
        diag_indices = np.arange(matrix.shape[0])[indices]
        matrix[diag_indices, diag_indices] = value


def get_oh_index_slice(n_elements, is_Regional):
    """Return the slice occupied by OH state-vector elements."""
    return slice(-1, None) if is_Regional else slice(-2, None)


def get_bc_index_slice(optimize_oh, is_Regional):
    """Return the slice occupied by boundary-condition state-vector elements."""
    if optimize_oh:
        return slice(-5, -1) if is_Regional else slice(-6, -2)
    return slice(-4, None)


def apply_oh_prior(Sa, Sa_constraint, n_elements, prior_err_oh, is_Regional):
    """Apply OH prior variances to both the unweighted and weighted constraint priors."""
    if is_Regional:
        OH_weight = 1 / (n_elements - 1)
    else:
        OH_weight = 2 / (n_elements - 2)
    oh_slice = get_oh_index_slice(n_elements, is_Regional)
    apply_diagonal_prior(Sa, oh_slice, prior_err_oh**2)
    apply_diagonal_prior(Sa_constraint, oh_slice, OH_weight * prior_err_oh**2)


def apply_bc_prior(Sa, Sa_constraint, prior_err_bc, optimize_oh, is_Regional):
    """Apply BC prior variances to both the unweighted and weighted constraint priors."""
    bc_slice = get_bc_index_slice(optimize_oh, is_Regional)
    apply_diagonal_prior(Sa, bc_slice, prior_err_bc**2)
    apply_diagonal_prior(Sa_constraint, bc_slice, prior_err_bc**2)


def invert_prior_covariance(Sa, Sa_constraint, use_full_prior_covariance):
    """Invert either full prior covariances or diagonal prior-variance vectors."""
    if use_full_prior_covariance:
        return np.linalg.inv(Sa_constraint), np.linalg.inv(Sa)
    return np.diag(1 / Sa_constraint), np.diag(1 / Sa)


# NOTE: the off-diagonal So spatial correlation is configured directly from the OffDiagonalObsCov*
# config keys -- the fixed South America two-exponential fit (see the so_corr_params dict built in
# __main__). These config keys are the SINGLE SOURCE OF TRUTH for the correlation. The per-region
# fit that build_obs_error_covariance.py writes to obs_error_covariance/so_correlation_*.npz is a
# DIAGNOSTIC only and is intentionally not read here. The data-driven part used per region is the
# REM diagonal So (obs_error_covariance/so_*.npz), built on individual observations.


def _great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (degree inputs)."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_offdiag_so_normal_equations(
    K, delta_y_vec, obs_error, lat, lon, dates, corr_params,
    temporal_rho=0.0, shrinkage=0.0,
):
    """EXACT off-diagonal-So normal equations via a day-blocked block-Thomas solve.

    Computes, exactly,
        KTinvSoK   = K^T So^-1 K
        KTinvSoy   = K^T So^-1 (y - F(xA))
        ytinvSoy   = (y - F(xA))^T So^-1 (y - F(xA))
    for the residual-error observation covariance  So = D (I + P) D,  D = diag(sqrt(obs_error)),
        P_ij = A1 exp(-d_ij/L1) + A2 exp(-d_ij/L2)   (two-exponential, no taper, hard cutoff),
    with observations grouped by day.  Within a day the dense correlation block is factorized
    (LU); adjacent days are coupled at lag-1 correlation `temporal_rho` through a block-tridiagonal
    (Thomas) forward/back substitution, so the full n_obs x n_obs So is never assembled.

    This is the exact application used in the inversion (valid for strong correlation, ||P|| >> 1).
    It needs per-observation latitude, longitude, and date; use it on the merged monthly path (not the
    day-level streaming path).
    """
    import scipy.linalg as sla
    from scipy.spatial import cKDTree

    A1 = float(corr_params["corr_amplitude1"]); L1 = float(corr_params["corr_length1_km"])
    A2 = float(corr_params["corr_amplitude2"]); L2 = float(corr_params["corr_length2_km"])
    cut = float(corr_params["corr_cutoff_km"])

    K = np.asarray(K, dtype=float)
    obs_error = np.asarray(obs_error, dtype=float)
    delta_y_vec = np.asarray(delta_y_vec, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    D = np.sqrt(obs_error)
    W = K / D[:, None]                                   # So-normalized Jacobian
    z = (delta_y_vec / D)[:, None]
    rhs_all = np.concatenate([W, z], axis=1)             # solve (I+P)^-1 against [W | z]

    # day index (accept datetime64 or YYYYMMDD strings/ints)
    dts = np.asarray(dates)
    if np.issubdtype(dts.dtype, np.datetime64):
        day = dts.astype("datetime64[D]")
    else:
        s = dts.astype(str)
        day = np.array([f"{v[:4]}-{v[4:6]}-{v[6:8]}" for v in s], dtype="datetime64[D]")
    day_index = (day - day.min()).astype(int)
    days = np.unique(day_index)
    groups = [np.where(day_index == d)[0] for d in days]
    n_days = len(days)

    mean_lat = np.radians(float(np.mean(lat)))
    X = np.radians(lon) * np.cos(mean_lat) * 6371.0     # planar coords for the neighbor search
    Y = np.radians(lat) * 6371.0
    # inflate the planar search radius so no true great-circle pair within `cut` is missed
    # (the cos(mean_lat) planar map over-estimates E-W distance away from the mean latitude)
    inflate = max(np.cos(mean_lat) / max(np.cos(np.radians(np.abs(lat).max())), 0.2), 1.0)
    search_radius = cut * inflate

    def corr_block(a, b):
        """Dense correlation matrix between obs sets a and b (two-exponential within cutoff)."""
        Bm = np.zeros((len(a), len(b)))
        ta = cKDTree(np.c_[X[a], Y[a]]); tb = cKDTree(np.c_[X[b], Y[b]])
        for ia, nbr in enumerate(ta.query_ball_tree(tb, search_radius)):
            if not nbr:
                continue
            jb = np.array(nbr)
            d = _great_circle_km(lat[a[ia]], lon[a[ia]], lat[b[jb]], lon[b[jb]])
            keep = d <= cut
            Bm[ia, jb[keep]] = (A1 * np.exp(-d[keep] / L1) + A2 * np.exp(-d[keep] / L2)) * (1.0 - shrinkage)
        return Bm

    # forward sweep: LU-factor each day's Schur complement
    B_factor = [None] * n_days
    coupling = [None] * n_days
    g = [None] * n_days
    for d in range(n_days):
        a = groups[d]
        Bd = corr_block(a, a)
        np.fill_diagonal(Bd, 1.0)
        rhs = rhs_all[a].copy()
        C = None
        if temporal_rho > 0.0 and d > 0 and (days[d] - days[d - 1] == 1):
            C = temporal_rho * corr_block(groups[d - 1], a)     # lag-1 coupling to the previous day
            Bd = Bd - C.T @ sla.lu_solve(B_factor[d - 1], C)
            rhs = rhs - C.T @ sla.lu_solve(B_factor[d - 1], g[d - 1])
        B_factor[d] = sla.lu_factor(Bd)
        g[d] = rhs
        coupling[d] = C

    # back substitution
    solved = np.zeros_like(rhs_all)
    Xb = [None] * n_days
    for d in range(n_days - 1, -1, -1):
        rhs = g[d]
        if d + 1 < n_days and coupling[d + 1] is not None:
            rhs = rhs - coupling[d + 1] @ Xb[d + 1]
        Xb[d] = sla.lu_solve(B_factor[d], rhs)
        solved[groups[d]] = Xb[d]

    SW = solved[:, : W.shape[1]]
    Sz = solved[:, W.shape[1]]
    KTinvSoK = W.T @ SW
    KTinvSoy = W.T @ Sz
    ytinvSoy = float(z[:, 0] @ Sz)
    return KTinvSoK, KTinvSoy, ytinvSoy


def compute_so_normal_equations(K, delta_y_vec, obs_error, lat, lon, dates, so_corr_params):
    """Observation-error normal-equation products for ANY solver (normal, softplus, lognormal):

        KTinvSoK = K^T So^-1 K
        KTinvSoy = K^T So^-1 (y - F(xA))
        ytinvSoy = (y - F(xA))^T So^-1 (y - F(xA))

    So is a SINGLE, solver-independent choice: a per-observation diagonal (obs_error), optionally
    combined with a same-day two-exponential off-diagonal spatial correlation plus an adjacent-day
    (lag-1) temporal correlation (so_corr_params with form "two_exponential"), applied EXACTLY by a
    day-blocked block-Thomas solve. With so_corr_params=None it is the plain diagonal So. This is the
    ONE place So^-1 is applied, so every solver sees identical observation weighting.
    Returns (KTinvSoK, KTinvSoy, ytinvSoy).
    """
    K = np.asarray(K, dtype=float)
    delta_y_vec = np.asarray(delta_y_vec, dtype=float)
    obs_error = np.asarray(obs_error, dtype=float)
    form = so_corr_params.get("form") if so_corr_params else None
    if form is not None and form != "two_exponential":
        raise ValueError(
            f"Unsupported off-diagonal So form {form!r}. The only supported off-diagonal model is the "
            "two-exponential spatial correlation + adjacent-day temporal correlation "
            "(so_corr_params['form'] = 'two_exponential'); set OffDiagonalObsCov: false for a diagonal So."
        )
    if form == "two_exponential":
        if not (lat is not None and lon is not None and dates is not None):
            raise ValueError(
                "Off-diagonal So (two-exponential) requires per-observation lat, lon, and dates for the "
                "exact day-blocked solve. Provide observation metadata (merged monthly path)."
            )
        temporal_rho = float(so_corr_params.get("temporal_rho", 0.0))
        KTinvSoK, KTinvSoy, ytinvSoy = build_offdiag_so_normal_equations(
            K, delta_y_vec, obs_error, lat, lon, dates, so_corr_params, temporal_rho=temporal_rho,
        )
        print(f"  Off-diagonal So applied EXACTLY (two-exponential, day-blocked block-Thomas: "
              f"A1={so_corr_params['corr_amplitude1']:.3f} L1={so_corr_params['corr_length1_km']:.0f} km, "
              f"A2={so_corr_params['corr_amplitude2']:.3f} L2={so_corr_params['corr_length2_km']:.0f} km, "
              f"no taper, cutoff={so_corr_params['corr_cutoff_km']:.0f} km, temporal_rho={temporal_rho})")
    else:
        # Plain diagonal So (no off-diagonal correlation).
        KTinvSo = K.transpose() / obs_error
        KTinvSoK = KTinvSo @ K
        KTinvSoy = KTinvSo @ delta_y_vec
        ytinvSoy = float(delta_y_vec @ (delta_y_vec / obs_error))
    return KTinvSoK, KTinvSoy, ytinvSoy


def solve_inversion_from_k(
    K,
    delta_y,
    n_elements,
    prior_err=0.5,
    gamma=0.25,
    prior_err_bc=0.0,
    prior_err_oh=0.0,
    is_Regional=True,
    OptimizeSoil=False,
    prior_ds=None,
    StateVectorFile=None,
    prebuilt_prior_err_covariance=False,
    so_corr_params=None,
    inversion_method="analytical",
    max_scale_factor=None,
    scale_factor_upper_bound=None,
    softplus_scale=SOFTPLUS_SCALE_DEFAULT,
):
    """Solve the inversion once K, y-F(xA), and So are already assembled.

    When so_corr_params is provided (the two-exponential off-diagonal So model, with per-observation
    lat/lon/dates) the observation weighting So^-1 is applied EXACTLY via compute_so_normal_equations
    (day-blocked block-Thomas); otherwise So is the plain per-observation diagonal.
    """
    optimize_bc = prior_err_bc > 0.0
    optimize_oh = prior_err_oh > 0.0

    Sa, Sa_constraint, use_full_prior_covariance = build_prior_covariance(
        n_elements,
        prior_err,
        OptimizeSoil=OptimizeSoil,
        prior_ds=prior_ds,
        StateVectorFile=StateVectorFile,
        prebuilt_prior_err_covariance=prebuilt_prior_err_covariance,
    )

    scale_factor_idx = n_elements
    if optimize_oh:
        apply_oh_prior(Sa, Sa_constraint, n_elements, prior_err_oh, is_Regional)
        scale_factor_idx -= 1 if is_Regional else 2

    if optimize_bc:
        scale_factor_idx -= 4
        apply_bc_prior(Sa, Sa_constraint, prior_err_bc, optimize_oh, is_Regional)

    inv_Sa_constraint, inv_Sa = invert_prior_covariance(
        Sa, Sa_constraint, use_full_prior_covariance
    )

    obs_error = np.asarray(delta_y["obs_error"], dtype=float)
    K = np.asarray(K, dtype=float)
    delta_y_vec = np.asarray(delta_y["delta_y"], dtype=float)

    # ---- Observation-error normal-equation products (shared solver-independent So operator) ----
    lat = np.asarray(delta_y["lat"], dtype=float) if "lat" in delta_y else None
    lon = np.asarray(delta_y["lon"], dtype=float) if "lon" in delta_y else None
    dates = delta_y.get("dates", None)
    KTinvSoK, KTinvSoyKxA, ytinvSoy = compute_so_normal_equations(
        K, delta_y_vec, obs_error, lat, lon, dates, so_corr_params
    )

    method = str(inversion_method or "analytical").lower()
    n_obs = int(K.shape[0])

    # Prior state vector in scale-factor space: 1 over the region of interest (and OH),
    # 0 for the concentration-space boundary-condition elements.
    xa = np.zeros(n_elements, dtype=float)
    xa[:scale_factor_idx] = 1.0
    if optimize_oh:
        if is_Regional:
            xa[-1] = 1.0
        else:
            xa[-2:] = 1.0

    if method in ("analytical", "normal", "gaussian"):
        system_constraint = gamma * KTinvSoK + inv_Sa_constraint
        delta_optimized = np.linalg.solve(system_constraint, gamma * KTinvSoyKxA)
        xhat = xa + delta_optimized
        if optimize_oh:
            print(f"xhat[OH] = {xhat[-1] if is_Regional else xhat[-2:]}")
        S_post = np.linalg.inv(gamma * KTinvSoK + inv_Sa)
        A = np.identity(n_elements) - S_post @ inv_Sa
        # returned Ja_normalized keeps the original full-state definition (used for ensemble
        # member selection); the richer diagnostics below are reported alongside it.
        Ja_normalized = float(delta_optimized @ inv_Sa @ delta_optimized) / n_elements
        diagnostics = inversion_diagnostics(
            delta_optimized, KTinvSoK, KTinvSoyKxA, ytinvSoy, inv_Sa, gamma, n_obs, scale_factor_idx
        )
    elif method == "softplus":
        xhat, delta_optimized, S_post, A, diagnostics, n_iter = run_softplus(
            KTinvSoK, KTinvSoyKxA, ytinvSoy, inv_Sa, inv_Sa_constraint,
            n_obs, scale_factor_idx, xa,
            scale=softplus_scale, gamma=gamma, kappa=POSITIVITY_KAPPA,
        )
        Ja_normalized = diagnostics["J_A"] / n_elements
        print(f"softplus positivity solver converged in {n_iter} iterations")
    else:
        raise ValueError(
            f"Unsupported InversionMethod={inversion_method!r}. "
            "Use the analytical (default) solver, set SoftplusErrors: true for the softplus "
            "positivity solver, or LognormalErrors: true for lognormal errors."
        )

    print(
        f"hyperparameters: (prior_err: {prior_err}, obs_err: {delta_y['obs_err_name']}, gamma: {gamma}, "
        + f"prior_err_bc: {prior_err_bc}, prior_err_oh: {prior_err_oh})"
    )
    print(f"InversionMethod: {method}")
    print(
        f"Diagnostics: DOFS={diagnostics['DOFS']:.1f}, "
        f"J_A/DOFS={diagnostics['J_A_over_DOFS']:.2f}, "
        f"J_O/(m-DOFS)={diagnostics['J_O_over_m_minus_DOFS']:.3f}  "
        f"(J_A/n={Ja_normalized:.3f})"
    )
    print(
        "Min:",
        xhat[:scale_factor_idx].min(),
        "Mean:",
        xhat[:scale_factor_idx].mean(),
        "Max",
        xhat[:scale_factor_idx].max(),
    )

    return xhat, delta_optimized, KTinvSoK, KTinvSoyKxA, S_post, A, Ja_normalized


def get_gridded_sf_scalers(sf_path, state_vector_file, n_columns):
    """Return one scale factor per emission state-vector element."""
    if sf_path in (None, "", "None", "False", False):
        return None
    if state_vector_file in (None, "", "None", "False", False):
        raise ValueError("StateVectorFile is required to apply gridded K scale factors.")

    sf = xr.load_dataset(sf_path)["ScaleFactor"].transpose("lat", "lon").values
    sv = xr.load_dataset(state_vector_file)["StateVector"].values
    labels = np.nan_to_num(sv, nan=0).astype(int)
    n_emis = int(np.nanmax(labels))
    if n_columns < n_emis:
        raise ValueError(
            f"Merged K has {n_columns} columns, fewer than {n_emis} emission labels."
        )

    scalers = np.ones(n_emis, dtype=float)
    for label in range(1, n_emis + 1):
        idx = np.where(labels == label)
        if idx[0].size:
            vals = sf[idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                scalers[label - 1] = float(np.nanmean(vals))
    return scalers


def scale_merged_k_by_gridded_sf(K, scalers):
    """Scale emission columns of merged K by state-vector scale factors."""
    if scalers is None:
        return K
    K[:, : scalers.size] *= scalers
    return K


def load_merged_jacobian_products(config, StateVectorFile=None):
    """Load monthly merged inversion products if they already exist.

    Returns (K, y, prior, so_dict, obs_lat, obs_lon, obs_dates).
    obs_lat/obs_lon/obs_dates are None if their files are not found; obs_dates
    (per-observation day) is needed for the exact day-blocked off-diagonal So.
    """
    start = str(config["StartDate"])
    end = str(config["EndDate"])
    inversion_data = Path(os.path.expandvars(config["OutputPath"])) / config["RunName"] / "inversion_data"
    files = {
        "K": inversion_data / "K" / f"K_{start}_{end}.npz",
        "y": inversion_data / "y" / f"y_{start}_{end}.npz",
        "prior": inversion_data / "xch4_0" / f"xch4_0_{start}_{end}.npz",
        "so": inversion_data / "so" / f"so_{start}_{end}.npz",
    }
    if not all(path.exists() for path in files.values()):
        return None

    with np.load(files["K"]) as data:
        # Merged K files are saved in dry-air mole-fraction units by
        # merge_partial_k.py. The inversion equations use ppb, matching the
        # chunk-by-chunk path below and the Colombia normal_invert scripts.
        K = 1e9 * data["K"]
    if bool(config.get("ClipNegativeK", False)):
        K[K < 0] = 0
    sf_path = config.get("NudgedScaleFactorPath", config.get("JacobianScaleFactorPath", None))
    state_vector_file = StateVectorFile or config.get("StateVectorFile", None)
    scalers = get_gridded_sf_scalers(sf_path, state_vector_file, K.shape[1])
    with np.load(files["y"]) as data:
        y = data["y"]
    with np.load(files["prior"]) as data:
        prior = data["xch4_0"] if "xch4_0" in data.files else data["gc_ch4"]
    if scalers is not None and bool(config.get("AdjustPriorWithScaleFactors", True)):
        prior = np.asarray(prior, dtype=float) + K[:, : scalers.size] @ (scalers - 1.0)
    K = scale_merged_k_by_gridded_sf(K, scalers)
    with np.load(files["so"]) as data:
        so_dict = {key: data[key] for key in data.files}

    if bool(config.get("UseResidualObsError", False)):
        residual_so_path = (
            inversion_data
            / "obs_error_covariance"
            / f"so_{start}_{end}.npz"
        )
        if not residual_so_path.exists():
            raise FileNotFoundError(
                "UseResidualObsError=true but residual diagonal So file was not found: "
                f"{residual_so_path}"
            )
        with np.load(residual_so_path) as data:
            residual_so = data["so"] if "so" in data.files else data[data.files[0]]
        if len(residual_so) != len(y):
            raise ValueError(
                "Residual diagonal So length does not match observations: "
                f"{len(residual_so)} vs {len(y)} for {start}->{end}"
            )
        for obs_err in ensure_float_list(config["ObsError"]):
            so_dict[f"so_{obs_err}"] = np.asarray(residual_so, dtype=float)
        print(
            "Using residual-error diagonal So from "
            f"{residual_so_path} for ObsError={ensure_float_list(config['ObsError'])}"
        )

    # Load observation coordinates and dates for off-diagonal So (optional)
    obs_lat, obs_lon, obs_dates = None, None, None
    obs_file = inversion_data / "observations" / f"observations_{start}_{end}.npz"
    if obs_file.exists():
        with np.load(obs_file, allow_pickle=True) as data:
            if "lat" in data.files and "lon" in data.files:
                obs_lat = np.asarray(data["lat"], dtype=float)
                obs_lon = np.asarray(data["lon"], dtype=float)
            if "dates" in data.files:
                obs_dates = np.asarray(data["dates"])
    # dates are stored separately in most IMI layouts
    if obs_dates is None:
        date_file = inversion_data / "date" / f"date_{start}_{end}.npz"
        if date_file.exists():
            with np.load(date_file, allow_pickle=True) as data:
                key = "dates" if "dates" in data.files else data.files[0]
                obs_dates = np.asarray(data[key])

    return K, y, prior, so_dict, obs_lat, obs_lon, obs_dates


def do_inversion(
    n_elements,
    jacobian_files,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    prior_err=0.5,
    obs_err=15,
    gamma=0.25,
    res="0.25x0.3125",
    jacobian_sf=None,
    prior_err_bc=0.0,
    prior_err_oh=0.0,
    is_Regional=True,
    OptimizeSoil=False,
    prior_ds=None,
    StateVectorFile=None,
    verbose=False,
    prebuilt_prior_err_covariance=False,
):
    """
    After running jacobian.py, use this script to perform the inversion and save out results.

    Arguments
        n_elements   [int]   : Number of state vector elements
        jacobian_files [list]  : Jacobian files generated from jacobian.py
        lon_min      [float] : Minimum longitude
        lon_max      [float] : Maximum longitude
        lat_min      [float] : Minimum latitude
        lat_max      [float] : Maximum latitude
        prior_err    [float] : Prior error standard deviation (default 0.5)
        obs_err      [float] : Observational error standard deviation (default 15 ppb)
        gamma        [float] : Regularization parameter (default 0.25)
        res          [str]   : Resolution string from config.yml (default '0.25x0.3125')
        jacobian_sf  [str]   : Path to Jacobian scale factors file if using precomputed K
        prior_err_bc [float] : Prior error standard deviation (default 0.0)
        prior_err_oh [float] : Prior error standard deviation (default 0.0)
        is_Regional  [bool]  : Is this a regional simulation?
        OptimizeSoil [bool]  : Optimize soil sink?
        prior_ds     [xr.Dataset]: prior emission dataset
        StateVectorFile [str]: Path to gridded state vector file

    Returns
        xhat         [float] : Posterior scaling factors
        delta_optimized        [float] : Change from prior     [xhat = 1 + delta_optimized]
        KTinvSoK     [float] : K^T*inv(S_o)*K        [part of inversion equation]
        KTinvSoyKxA  [float] : K^T*inv(S_o)*(y-K*xA) [part of inversion equation]
        S_post       [float] : Posterior error covariance matrix
        A            [float] : Averaging kernel matrix

    """
    # make mapping of target files to reference files if using precomputed Jacobian
    if jacobian_sf is not None:
        reference_dir = jacobian_dir.replace(
            "data_converted", "data_converted_reference"
        )
        K_ref_file_mappings = map_files_to_reference(jacobian_dir, reference_dir)

    # boolean for whether we are optimizing boundary conditions
    optimize_bc = prior_err_bc > 0.0
    optimize_oh = prior_err_oh > 0.0

    # Need to ignore data in the GEOS-Chem 3 3 3 3 buffer zone
    # Shave off one or two degrees of latitude/longitude from each side of the domain
    # ~1 degree if 0.25x0.3125 resolution, ~2 degrees if 0.5x0.6125 resolution
    # This assumes 0.25x0.3125 and 0.5x0.625 simulations are always regional
    if not config['UseGCHP']:
        if "0.125x0.15625" in res:
            degx = 4 * 0.15625
            degy = 4 * 0.125
        elif "0.25x0.3125" in res:
            degx = 4 * 0.3125
            degy = 4 * 0.25
        elif "0.5x0.625" in res:
            degx = 4 * 0.625
            degy = 4 * 0.5
        else:
            degx = 0
            degy = 0
    else:
        degx = 0
        degy = 0
    xlim = [lon_min + degx, lon_max - degx]
    ylim = [lat_min + degy, lat_max - degy]

    # Read output data from jacobian.py (virtual & true satellite columns, Jacobian matrix)    
    files = jacobian_files
    
    # make mapping of target files to reference files if using precomputed Jacobian
    if jacobian_sf is not None:
        reference_dir = jacobian_dir.replace(
            "data_converted", "data_converted_reference"
        )
        K_ref_file_mappings = map_files_to_reference(jacobian_dir, reference_dir)
        
        # filter files to only read files we have reference Jacobians for
        files = [file for file in files if K_ref_file_mappings[Path(file)] is not None]

    # ==========================================================================================
    # Now we will assemble two different expressions needed for the analytical inversion.
    #
    # These expressions are from eq. (5) and (6) in Zhang et al. (2018) ACP:
    # "Monitoring global OH concentrations using satellite observations of atmospheric methane".
    #
    # Specifically, we are going to solve:
    #   xhat = xA + G*(y-K*xA)
    #        = xA + inv(gamma * K^T*inv(S_o)*K + inv(S_a)) * gamma * K^T*inv(S_o) * (y-K*xA)
    #                          (--------------)                     (-----------------------)
    #                            Expression 1                             Expression 2
    #
    # Expression 1 = "KTinvSoK"
    # Expression 2 = "KTinvSoyKxA"
    #
    # In the code below this becomes
    #   xhat = xA + inv(gamma*KTinvSoK + inv(S_a)) * gamma*KTinvSoyKxA
    #        = xA + delta_optimized
    #        = 1  + delta_optimized      [since xA=1 when optimizing scale factors]
    #
    # We build KTinvSoK and KTinvSoyKxA "piece by piece", loading one jacobian .pkl file at a
    # time. This is so that we don't need to assemble or invert the full Jacobian matrix, which
    # can be very large.
    # ==========================================================================================

    # Initialize two expressions from the inversion equation
    KTinvSoK = np.zeros(
        [n_elements, n_elements], dtype=float
    )  # expression 1: K^T * inv(S_o) * K
    KTinvSoyKxA = np.zeros(
        [n_elements], dtype=float
    )  # expression 2: K^T * inv(S_o) * (y-K*xA)

    # Initialize
    # For each .pkl file generated by jacobian.py:
    for fi in files:
        if verbose:
            print(fi)

        # Load satellite/GEOS-Chem and Jacobian matrix data from the .pkl file
        dat = load_obj(fi)

        # Skip if there aren't any satellite observations on this day
        if dat["obs_GC"].shape[0] == 0:
            continue

        # Otherwise, grab the satellite/GEOS-Chem data
        obs_GC = dat["obs_GC"]

        # Only consider data within the new latitude and longitude bounds
        ind = np.where(
            (obs_GC[:, 2] >= xlim[0])
            & (obs_GC[:, 2] <= xlim[1])
            & (obs_GC[:, 3] >= ylim[0])
            & (obs_GC[:, 3] <= ylim[1])
        )[0]

        # Skip if no data in bounds
        if len(ind) == 0:
            continue

        # Satellite and GEOS-Chem data within bounds
        obs_GC = obs_GC[ind, :]

        ref_ind = None
        if jacobian_sf is not None:
            # Precomputed Jacobians come from a reference run, so align observations
            # before indexing K to ensure each retained row maps to the same scene.
            fi_ref = str(K_ref_file_mappings.get(Path(fi)))
            if fi_ref is None:
                print(f"No reference file found for {fi} in {jacobian_dir}")
                continue
            dat_ref = load_obj(fi_ref)
            obs_ind, ref_ind = align_obs_rows_with_reference(obs_GC, dat_ref["obs_GC"])
            if len(ref_ind) == 0:
                print(f"No overlapping reference observations found for {fi_ref}")
                continue
            obs_GC = obs_GC[obs_ind, :]

        # weight obs_err based on the observation count to prevent overfitting
        # Note: weighting function defined by Zichong Chen for his
        # middle east inversions. May need to be tuned based on region.
        # From Chen et al. 2023:
        # "Satellite quantification of methane emissions and oil/gas methane
        # intensities from individual countries in the Middle East and North
        # Africa: implications for climate action"
        s_superO_1 = calculate_superobservation_error(obs_err, 1)
        s_superO_p = np.array(
            [
                calculate_superobservation_error(obs_err, p) if p >= 1 else s_superO_1
                for p in obs_GC[:, 4]
            ]
        )
        # Define observational errors (diagonal entries of S_o matrix)
        obs_error = np.power(obs_err, 2)
        gP = s_superO_p**2 / s_superO_1**2
        # scale error variance by gP
        obs_error = gP * obs_error

        # check to make sure obs_err isn't negative, set 1 as default value
        obs_error = [obs if obs > 0 else 1 for obs in obs_error]

        # Jacobian entries for observations within bounds [ppb]
        if jacobian_sf is None:
            K = 1e9 * dat["K"][ind, :]
        else:
            K = 1e9 * dat_ref["K"][ref_ind, :]

        # Number of observations
        if verbose:
            print("Sum of Jacobian entries:", np.sum(K))

        # Apply scaling matrix if using precomputed Jacobian
        if jacobian_sf is not None:
            scale_factors = np.load(jacobian_sf)
            if optimize_bc:
                # add (unit) scale factors for BCs
                # as the last 4 elements of the scaling matrix
                scale_factors = np.append(scale_factors, np.ones(4))
            reps = K.shape[0]
            scaling_matrix = np.tile(scale_factors, (reps, 1))
            if optimize_oh:
                if is_Regional:
                    K[:, :-1] *= scaling_matrix
                else:
                    K[:, :-2] *= scaling_matrix
            else:
                K *= scaling_matrix

        # Measurement-model mismatch: TROPOMI columns minus GEOS-Chem virtual TROPOMI columns
        # This is (y - F(xA)), i.e., (y - (K*xA + c)) or (y - K*xA) in shorthand
        delta_y = obs_GC[:, 0] - obs_GC[:, 1]  # [ppb]

        # If there are any nans in the data, abort
        if (
            np.any(np.isnan(delta_y))
            or np.any(np.isnan(K))
            or np.any(np.isnan(obs_error))
        ):
            print("missing values", fi)
            break

        # Define KTinvSo = K^T * inv(S_o)
        KT = K.transpose()
        KTinvSo = np.zeros(KT.shape, dtype=float)
        for k in range(KT.shape[1]):
            KTinvSo[:, k] = KT[:, k] / obs_error[k]

        # Parts of inversion equation
        partial_KTinvSoK = KTinvSo @ K  # expression 1: K^T * inv(S_o) * K
        partial_KTinvSoyKxA = (
            KTinvSo @ delta_y
        )  # expression 2: K^T * inv(S_o) * (y-K*xA)

        # Add partial expressions to sums
        KTinvSoK += partial_KTinvSoK
        KTinvSoyKxA += partial_KTinvSoyKxA

    # Build either a full precomputed prior covariance or the original diagonal form.
    Sa, Sa_constraint, use_full_prior_covariance = build_prior_covariance(
        n_elements,
        prior_err,
        OptimizeSoil=OptimizeSoil,
        prior_ds=prior_ds,
        StateVectorFile=StateVectorFile,
        prebuilt_prior_err_covariance=prebuilt_prior_err_covariance,
    )

    # Number of elements to apply scale factor to
    scale_factor_idx = n_elements

    # If optimizing OH, adjust for it in the inversion
    if optimize_oh:
        # Add prior error for OH as the last element(s) of the diagonal
        # Following Masakkers et al. (2019, ACP) weight the OH term by the
        # ratio of the number of elements (n_OH_elements/n_emission_elements)
        # use this weighted constraint matrix to calculate the solution only
        apply_oh_prior(Sa, Sa_constraint, n_elements, prior_err_oh, is_Regional)
        scale_factor_idx -= 1 if is_Regional else 2

    # If optimizing boundary conditions, adjust for it in the inversion
    if optimize_bc:
        scale_factor_idx -= 4
        apply_bc_prior(Sa, Sa_constraint, prior_err_bc, optimize_oh, is_Regional)

    # The inversion uses the weighted constraint prior for the state estimate and the
    # unweighted prior for posterior diagnostics such as S_post and the averaging kernel.
    inv_Sa_constraint, inv_Sa = invert_prior_covariance(
        Sa, Sa_constraint, use_full_prior_covariance
    )

    # Solve for posterior scale factors xhat using the weighted constraint matrix
    delta_optimized = np.linalg.inv(gamma * KTinvSoK + inv_Sa_constraint) @ (
        gamma * KTinvSoyKxA
    )

    # Update scale factors by 1 to match what GEOS-Chem expects
    # xhat = 1 + delta_optimized
    # Notes:
    #  - If optimizing BCs, the last 4 elements are in concentration space,
    #    so we do not need to add 1
    #  - If optimizing OH, the last element also needs to be updated by 1
    xhat = delta_optimized.copy()
    xhat[:scale_factor_idx] += 1
    if optimize_oh:
        if is_Regional:
            xhat[-1] += 1
            print(f"xhat[OH] = {xhat[-1]}")
        else:
            xhat[-2:] += 1
            print(f"xhat[OH] = {xhat[-2:]}")

    # Posterior error covariance matrix (use unweighted Sa)
    S_post = np.linalg.inv(gamma * KTinvSoK + inv_Sa)

    # Averaging kernel matrix (use unweighted Sa)
    A = np.identity(n_elements) - S_post @ inv_Sa

    # Calculate J_A, where delta_optimized = xhat - xA
    # J_A = (xhat - xA)^T * inv_Sa * (xhat - xA)
    delta_optimizedT = delta_optimized.transpose()
    J_A = delta_optimizedT @ inv_Sa @ delta_optimized
    Ja_normalized = J_A / n_elements

    # Print some statistics
    print(
        f"hyperparameters: (prior_err: {prior_err}, obs_err: {obs_err}, gamma: {gamma}, "
        + f"prior_err_bc: {prior_err_bc}, prior_err_oh: {prior_err_oh})"
    )
    print(f"Normalized J_A: {Ja_normalized}")  # ideal gamma is where this is close to 1
    print(
        "Min:",
        xhat[:scale_factor_idx].min(),
        "Mean:",
        xhat[:scale_factor_idx].mean(),
        "Max",
        xhat[:scale_factor_idx].max(),
    )

    return xhat, delta_optimized, KTinvSoK, KTinvSoyKxA, S_post, A, Ja_normalized


def do_inversion_ensemble(
    n_elements,
    jacobian_files,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    prior_errs,
    obs_errs,
    gammas,
    res,
    jacobian_sf,
    prior_errs_bc,
    prior_errs_oh,
    is_Regional,
    OptimizeSoil=False,
    prior_ds=None,
    StateVectorFile=None,
    prebuilt_prior_err_covariance=False,
):
    """
    Run series of inversions with hyperparameter vectors and save out the results.
    """
    hyperparam_ensemble = list(
        product(prior_errs, obs_errs, gammas, prior_errs_bc, prior_errs_oh)
    )

    results_dict = {
        "KTinvSoK": [],
        "KTinvSoyKxA": [],
        "ratio": [],
        "xhat": [],
        "S_post": [],
        "A": [],
        "Ja_normalized": [],
        "prior_err": [],
        "obs_err": [],
        "gamma": [],
        "prior_err_bc": [],
        "prior_err_oh": [],
    }
    for member in hyperparam_ensemble:
        prior_err, obs_err, gamma, prior_err_bc, prior_err_oh = member
        params = {
            "prior_err": prior_err,
            "obs_err": obs_err,
            "gamma": gamma,
            "prior_err_bc": prior_err_bc,
            "prior_err_oh": prior_err_oh,
        }
        xhat, delta_optimized, KTinvSoK, KTinvSoyKxA, S_post, A, Ja_normalized = (
            do_inversion(
                n_elements,
                jacobian_files,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                prior_err,
                obs_err,
                gamma,
                res,
                jacobian_sf,
                prior_err_bc,
                prior_err_oh,
                is_Regional,
                OptimizeSoil,
                prior_ds,
                StateVectorFile,
                verbose=False,
                prebuilt_prior_err_covariance=prebuilt_prior_err_covariance,
            )
        )
        results_dict["KTinvSoK"].append(KTinvSoK)
        results_dict["KTinvSoyKxA"].append(KTinvSoyKxA)
        results_dict["ratio"].append(delta_optimized)
        results_dict["xhat"].append(xhat)
        results_dict["S_post"].append(S_post)
        results_dict["A"].append(A)
        results_dict["Ja_normalized"].append(Ja_normalized)
        for k, v in params.items():
            results_dict[k].append(v)

    # Find the ensemble member that is closest to 1 following Lu et al. (2021)
    idx_default_Ja = np.argmin(np.abs(np.array(results_dict["Ja_normalized"]) - 1))
    print(
        f"J_A/n closest to 1: {results_dict['Ja_normalized'][idx_default_Ja]} with"
        + f" (prior_err, obs_err, gamma, prior_err_bc, prior_err_oh) = {hyperparam_ensemble[idx_default_Ja]}"
    )

    # Filter ensemble members to only members with Ja between 0.5 and 2
    filter_ens_members = True  # set to False to turn off filtering
    include_ens_members = [
        i for i, Ja in enumerate(results_dict["Ja_normalized"]) if 0.5 <= Ja <= 2.0
    ]
    if filter_ens_members and len(include_ens_members) > 0:
        for k in results_dict.keys():
            results_dict[k] = [results_dict[k][i] for i in include_ens_members]
        # map idx_default_Ja to new filtered index
        idx_default_Ja = include_ens_members.index(idx_default_Ja)
    elif len(include_ens_members) == 0:
        print(
            "Warning: No ensemble members with 0.5 <= J_A/n <= 2.0, "
            + "Returning all members in ensemble. This may lead to suboptimal results."
            + " Consider adding additional ensemble members with different hyperparameters."
        )
    else:
        print(
            "Warning: Returning all members in ensemble without filtering "
            + "Ja/n thresholds [0.5, 2.0]. This may lead to suboptimal results."
            + " Consider adding ensemble filters."
        )

    # Create an xarray.Dataset
    dataset = xr.Dataset()
    for k, v in results_dict.items():
        v = np.array(v)
        dims = ["ensemble"] + [f"nvar{i}" for i in range(1, v.ndim)]
        dataset[k] = (dims, v)

    # ensemble dimension to end
    dataset = dataset.transpose(..., "ensemble")

    # Specify attributes
    dataset.xhat.attrs["long_name"] = "Posterior scaling factors"
    dataset.xhat.attrs["units"] = "1"
    dataset.S_post.attrs["long_name"] = "Posterior error covariance matrix"
    dataset.S_post.attrs["units"] = "1"
    dataset.A.attrs["long_name"] = "Averaging kernel matrix"
    dataset.A.attrs["units"] = "1"
    dataset.Ja_normalized.attrs["long_name"] = "Normalized cost function Ja/n"
    dataset.Ja_normalized.attrs["units"] = "1"
    dataset.prior_err.attrs["long_name"] = "Prior error (Sa)"
    dataset.prior_err.attrs["units"] = "1"
    dataset.obs_err.attrs["long_name"] = "Observation error (So)"
    dataset.obs_err.attrs["units"] = "ppb"
    dataset.gamma.attrs["long_name"] = "Regularization parameter"
    dataset.gamma.attrs["units"] = "1"
    dataset.prior_err_bc.attrs["long_name"] = "Prior error for BC elements"
    dataset.prior_err_bc.attrs["units"] = "ppb"
    dataset.prior_err_oh.attrs["long_name"] = "Prior error for OH elements"
    dataset.prior_err_oh.attrs["units"] = "1"
    dataset.KTinvSoK.attrs["long_name"] = "K^T * inv(So) * K expression from inversion equation"
    dataset.KTinvSoK.attrs["units"] = "1"
    dataset.KTinvSoyKxA.attrs["long_name"] = "K^T * inv(So) * (y-K*xA) expression from inversion equation"
    dataset.KTinvSoyKxA.attrs["units"] = "1"
    dataset.ratio.attrs["long_name"] = "Change from prior (xhat - xA)"
    dataset.ratio.attrs["units"] = "1"

    # Calculate the mean of the ensemble as the main result
    dataset_mean = dataset.mean(dim="ensemble")

    dataset_mean.xhat.attrs["long_name"] = "Posterior scaling factors"
    dataset_mean.xhat.attrs["units"] = "1"
    dataset_mean.S_post.attrs["long_name"] = "Posterior error covariance matrix"
    dataset_mean.S_post.attrs["units"] = "1"
    dataset_mean.A.attrs["long_name"] = "Averaging kernel matrix"
    dataset_mean.A.attrs["units"] = "1"
    dataset_mean.Ja_normalized.attrs["long_name"] = "Normalized cost function Ja/n"
    dataset_mean.Ja_normalized.attrs["units"] = "1"
    dataset_mean.prior_err.attrs["long_name"] = "Prior error (Sa)"
    dataset_mean.prior_err.attrs["units"] = "1"
    dataset_mean.obs_err.attrs["long_name"] = "Observation error (So)"
    dataset_mean.obs_err.attrs["units"] = "ppb"
    dataset_mean.gamma.attrs["long_name"] = "Regularization parameter"
    dataset_mean.gamma.attrs["units"] = "1"
    dataset_mean.prior_err_bc.attrs["long_name"] = "Prior error for BC elements"
    dataset_mean.prior_err_bc.attrs["units"] = "ppb"
    dataset_mean.prior_err_oh.attrs["long_name"] = "Prior error for OH elements"
    dataset_mean.prior_err_oh.attrs["units"] = "1"
    dataset_mean.KTinvSoK.attrs["long_name"] = "K^T * inv(So) * K expression from inversion equation"
    dataset_mean.KTinvSoK.attrs["units"] = "1"
    dataset_mean.KTinvSoyKxA.attrs["long_name"] = "K^T * inv(So) * (y-K*xA) expression from inversion equation"
    dataset_mean.KTinvSoyKxA.attrs["units"] = "1"
    dataset_mean.ratio.attrs["long_name"] = "Change from prior (xhat - xA)"
    dataset_mean.ratio.attrs["units"] = "1"

    return dataset, dataset_mean


def do_inversion_ensemble_from_merged(
    n_elements,
    K,
    y,
    prior,
    so_dict,
    prior_errs,
    obs_errs,
    gammas,
    prior_errs_bc,
    prior_errs_oh,
    is_Regional,
    OptimizeSoil=False,
    prior_ds=None,
    StateVectorFile=None,
    prebuilt_prior_err_covariance=False,
    obs_lat=None,
    obs_lon=None,
    obs_dates=None,
    so_corr_params=None,
    inversion_method="analytical",
    max_scale_factor=None,
    scale_factor_upper_bound=None,
    softplus_scale=SOFTPLUS_SCALE_DEFAULT,
):
    """Run the inversion ensemble using monthly merged arrays instead of day-level pickles."""
    hyperparam_ensemble = list(
        product(prior_errs, obs_errs, gammas, prior_errs_bc, prior_errs_oh)
    )

    results_dict = {
        "KTinvSoK": [],
        "KTinvSoyKxA": [],
        "ratio": [],
        "xhat": [],
        "S_post": [],
        "A": [],
        "Ja_normalized": [],
        "prior_err": [],
        "obs_err": [],
        "gamma": [],
        "prior_err_bc": [],
        "prior_err_oh": [],
    }
    delta_y_base = np.asarray(y, dtype=float) - np.asarray(prior, dtype=float)

    for member in hyperparam_ensemble:
        prior_err, obs_err, gamma, prior_err_bc, prior_err_oh = member
        so_key = f"so_{obs_err}"
        if so_key not in so_dict:
            raise KeyError(f"Missing {so_key} in merged observational-error file.")
        params = {
            "prior_err": prior_err,
            "obs_err": obs_err,
            "gamma": gamma,
            "prior_err_bc": prior_err_bc,
            "prior_err_oh": prior_err_oh,
        }
        delta_y_dict = {
            "delta_y": delta_y_base,
            "obs_error": so_dict[so_key],
            "obs_err_name": obs_err,
        }
        if obs_lat is not None:
            delta_y_dict["lat"] = obs_lat
        if obs_lon is not None:
            delta_y_dict["lon"] = obs_lon
        if obs_dates is not None:
            delta_y_dict["dates"] = obs_dates
        xhat, delta_optimized, KTinvSoK, KTinvSoyKxA, S_post, A, Ja_normalized = solve_inversion_from_k(
            K,
            delta_y_dict,
            n_elements,
            prior_err=prior_err,
            gamma=gamma,
            prior_err_bc=prior_err_bc,
            prior_err_oh=prior_err_oh,
            is_Regional=is_Regional,
            OptimizeSoil=OptimizeSoil,
            prior_ds=prior_ds,
            StateVectorFile=StateVectorFile,
            prebuilt_prior_err_covariance=prebuilt_prior_err_covariance,
            so_corr_params=so_corr_params,
            inversion_method=inversion_method,
            max_scale_factor=max_scale_factor,
            scale_factor_upper_bound=scale_factor_upper_bound,
            softplus_scale=softplus_scale,
        )
        results_dict["KTinvSoK"].append(KTinvSoK)
        results_dict["KTinvSoyKxA"].append(KTinvSoyKxA)
        results_dict["ratio"].append(delta_optimized)
        results_dict["xhat"].append(xhat)
        results_dict["S_post"].append(S_post)
        results_dict["A"].append(A)
        results_dict["Ja_normalized"].append(Ja_normalized)
        for k, v in params.items():
            results_dict[k].append(v)

    idx_default_Ja = np.argmin(np.abs(np.array(results_dict["Ja_normalized"]) - 1))
    print(
        f"J_A/n closest to 1: {results_dict['Ja_normalized'][idx_default_Ja]} with"
        + f" (prior_err, obs_err, gamma, prior_err_bc, prior_err_oh) = {hyperparam_ensemble[idx_default_Ja]}"
    )

    filter_ens_members = True
    include_ens_members = [
        i for i, Ja in enumerate(results_dict["Ja_normalized"]) if 0.5 <= Ja <= 2.0
    ]
    if filter_ens_members and len(include_ens_members) > 0:
        for k in results_dict.keys():
            results_dict[k] = [results_dict[k][i] for i in include_ens_members]

    dataset = xr.Dataset()
    for k, v in results_dict.items():
        v = np.array(v)
        dims = ["ensemble"] + [f"nvar{i}" for i in range(1, v.ndim)]
        dataset[k] = (dims, v)

    dataset = dataset.transpose(..., "ensemble")
    dataset_mean = dataset.mean(dim="ensemble")
    return dataset, dataset_mean


if __name__ == "__main__":
    import sys
    import os

    config_path = sys.argv[1]
    n_elements = int(sys.argv[2])
    jacobian_dir = sys.argv[3]
    output_path = sys.argv[4]
    lon_min = float(sys.argv[5])
    lon_max = float(sys.argv[6])
    lat_min = float(sys.argv[7])
    lat_max = float(sys.argv[8])
    res = sys.argv[9]
    jacobian_sf = sys.argv[10]
    StateVectorFile = sys.argv[11]

    # read in config file
    config = load_config(config_path)

    # set parameters based on config file
    is_Regional = config["isRegional"]
    prior_err = ensure_float_list(config["PriorError"])
    obs_err = ensure_float_list(config["ObsError"])
    gamma = ensure_float_list(config["Gamma"])
    # Solver selection. `SoftplusErrors: true` is the documented flag (parallel to
    # `LognormalErrors: true`) that turns on the softplus positivity solver;
    # `InversionMethod: softplus` remains honored for back-compat.
    inversion_method = resolve_inversion_method(config)
    softplus_scale = float(config.get("SoftplusScale", SOFTPLUS_SCALE_DEFAULT))
    # NOTE: MaxScaleFactor / MaxTrueScaleFactor are read and threaded but not currently
    # applied (no clamp); left inert intentionally (uncapped solves). Not a documented key.
    max_scale_factor = config.get("MaxScaleFactor", None)
    max_true_scale_factor = config.get("MaxTrueScaleFactor", None)

    # 0.0 if not optimizing BCs or OH
    prior_err_BC = config["PriorErrorBCs"] if config["OptimizeBCs"] else 0.0
    prior_err_OH = config["PriorErrorOH"] if config["OptimizeOH"] else 0.0
    prior_err_BC = ensure_float_list(prior_err_BC)
    prior_err_OH = ensure_float_list(prior_err_OH)
    prebuilt_prior_err_covariance = config["OffDiagonalPriorCov"]
    
    OptimizeSoil = config["OptimizeSoil"]
    if OptimizeSoil:
        # prior emissions
        prior_cache = f"{os.path.expandvars(config['OutputPath']) }/{config['RunName']}/hemco_prior_emis/OutputDir/"
        start_date = config["StartDate"]
        end_date = config["EndDate"]
        prior_ds = get_mean_emissions(start_date, end_date, prior_cache)
    else:
        prior_ds = None

    # Reformat Jacobian scale factor input
    if jacobian_sf == "None":
        jacobian_sf = None

    use_offdiag_so = bool(config.get("OffDiagonalObsCov", True))  # default ON (matches config.yml + run_inversion.sh)

    merged_products = load_merged_jacobian_products(config, StateVectorFile)
    if merged_products is not None and jacobian_sf is None:
        K, y, prior, so_dict, obs_lat, obs_lon, obs_dates = merged_products
        scale_factor_upper_bound = None
        if max_true_scale_factor not in (None, "", "None", False):
            sf_path = config.get(
                "NudgedScaleFactorPath", config.get("JacobianScaleFactorPath", None)
            )
            scalers = get_gridded_sf_scalers(sf_path, StateVectorFile, K.shape[1])
            if scalers is None:
                scalers = np.ones(K.shape[1], dtype=float)
            scale_factor_upper_bound = float(max_true_scale_factor) / np.maximum(
                scalers, 1.0e-12
            )
            print(
                f"MaxTrueScaleFactor={max_true_scale_factor}: local xhat upper "
                f"min={np.nanmin(scale_factor_upper_bound):.3f}, "
                f"max={np.nanmax(scale_factor_upper_bound):.3f}"
            )
        so_corr_params = None
        if use_offdiag_so:
            # Off-diagonal observation-error correlation: a two-exponential rho(d) applied EXACTLY
            # (day-blocked block-Thomas) plus an adjacent-day (lag-1) temporal correlation.  The
            # correlation length scales, amplitudes, and temporal correlation are read from the config;
            # the defaults below are the South America values (fit to SA TROPOMI residuals).
            L2 = float(config.get("OffDiagonalObsCovL2", 398.0))
            so_corr_params = {
                "form": "two_exponential",
                "corr_amplitude1": float(config.get("OffDiagonalObsCovA1", 0.385)),
                "corr_length1_km": float(config.get("OffDiagonalObsCovL1", 26.0)),
                "corr_amplitude2": float(config.get("OffDiagonalObsCovA2", 0.459)),
                "corr_length2_km": L2,
                "corr_cutoff_km": float(config.get("OffDiagonalObsCovCutoffKm", 3.0 * L2)),
                "temporal_rho": float(config.get("OffDiagonalObsCovTemporalRho", 0.17)),
            }
            print(
                f"Off-diagonal So enabled: two-exponential "
                f"(A1={so_corr_params['corr_amplitude1']:.3f} L1={so_corr_params['corr_length1_km']:.0f} km, "
                f"A2={so_corr_params['corr_amplitude2']:.3f} L2={so_corr_params['corr_length2_km']:.0f} km), "
                f"cutoff={so_corr_params['corr_cutoff_km']:.0f} km, temporal_rho={so_corr_params['temporal_rho']}, "
                f"application=exact day-blocked block-Thomas"
            )
        out_ds, out_ds_mean = do_inversion_ensemble_from_merged(
            n_elements,
            K,
            y,
            prior,
            so_dict,
            prior_err,
            obs_err,
            gamma,
            prior_err_BC,
            prior_err_OH,
            is_Regional,
            OptimizeSoil,
            prior_ds,
            StateVectorFile,
            prebuilt_prior_err_covariance,
            obs_lat=obs_lat,
            obs_lon=obs_lon,
            obs_dates=obs_dates,
            so_corr_params=so_corr_params,
            inversion_method=inversion_method,
            max_scale_factor=max_scale_factor,
            scale_factor_upper_bound=scale_factor_upper_bound,
            softplus_scale=softplus_scale,
        )
    else:
        # Day-level streaming path: reads observations day by day and accumulates
        # K^T So^-1 K without holding all observations at once, for state vectors too
        # large for the merged path.  It solves the standard analytical (normal)
        # inversion only; the softplus positivity solver and off-diagonal So need the
        # merged path (they operate on the full assembled system).
        # NOTE: PrecomputedJacobian=true sets jacobian_sf (non-None), which routes execution here
        # (the merged path above requires jacobian_sf is None), so a precomputed-Jacobian re-run with
        # the default OffDiagonalObsCov/softplus lands on these guards.
        if use_offdiag_so:
            raise RuntimeError(
                "OffDiagonalObsCov=true requires the merged monthly inversion products with "
                "observation latitude/longitude metadata; the day-level pickle fallback path cannot "
                "apply residual-correlation So. This path is taken when PrecomputedJacobian=true "
                "(jacobian_sf is set). For a precomputed-Jacobian re-run, set OffDiagonalObsCov: false, "
                "or run the full (non-precomputed) merged path."
            )
        if str(inversion_method).lower() not in ("analytical", "normal", "gaussian"):
            raise RuntimeError(
                f"InversionMethod={inversion_method} requires the merged monthly inversion path; the "
                "day-level streaming path supports only the analytical (normal) inversion. This path is "
                "taken when PrecomputedJacobian=true (jacobian_sf is set). Set InversionMethod/solver to "
                "normal for a precomputed-Jacobian re-run, or use the full merged path for softplus/lognormal."
            )
        gc_startdate = np.datetime64(datetime.datetime.strptime(str(config['StartDate']), "%Y%m%d"))
        gc_enddate = np.datetime64(datetime.datetime.strptime(str(config['EndDate']), "%Y%m%d"))
        allfiles = glob.glob(f"{jacobian_dir}/*.pkl")
        jacobian_files = []
        for index in range(len(allfiles)):
            filename = allfiles[index]
            shortname = re.split(r"\/", filename)[-1]
            shortname = re.split(r"\.", shortname)[0]
            strdate = re.split(r"\.|_+|T", shortname)[4]
            strdate = datetime.datetime.strptime(strdate, "%Y%m%d")
            if (strdate >= gc_startdate) and (strdate < gc_enddate):
                jacobian_files.append(filename)
        jacobian_files.sort()

        out_ds, out_ds_mean = do_inversion_ensemble(
            n_elements,
            jacobian_files,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            prior_err,
            obs_err,
            gamma,
            res,
            jacobian_sf,
            prior_err_BC,
            prior_err_OH,
            is_Regional,
            OptimizeSoil,
            prior_ds,
            StateVectorFile,
            prebuilt_prior_err_covariance,
        )

    # add atributes for stretching GCHP simulation
    if config.get('STRETCH_GRID', False):
        out_ds.attrs['STRETCH_FACTOR'] = np.float32(config['STRETCH_FACTOR'])
        out_ds.attrs['TARGET_LAT'] = np.float32(config['TARGET_LAT'])
        out_ds.attrs['TARGET_LON'] = np.float32(config['TARGET_LON'])
        
        out_ds_mean.attrs['STRETCH_FACTOR'] = np.float32(config['STRETCH_FACTOR'])
        out_ds_mean.attrs['TARGET_LAT'] = np.float32(config['TARGET_LAT'])
        out_ds_mean.attrs['TARGET_LON'] = np.float32(config['TARGET_LON'])
    # Save the results of the ensemble inversion
    out_ds.to_netcdf(
        output_path.replace(".nc", "_ensemble.nc"),
        encoding={v: {"zlib": True, "complevel": 1} for v in out_ds.data_vars},
    )

    out_ds_mean.to_netcdf(
        output_path,
        encoding={v: {"zlib": True, "complevel": 1} for v in out_ds_mean.data_vars},
    )
    print(f"Saved results to {output_path}")
