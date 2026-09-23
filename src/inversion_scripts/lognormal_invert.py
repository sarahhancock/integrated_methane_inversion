# Description: Script to perform inversion using lognormal errors
# Usage: python lognormal_invert.py <path_to_config_file> <path_to_state_vector_file> <jacobian_sf>
# Inputs:
#       path_to_config_file: path to yaml config file
#       path_to_state_vector_file: path to state vector netcdf file
#       jacobian_sf: (optional) path to numpy array of scale factors for jacobian

import os
import sys
from itertools import product
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from src.inversion_scripts.invert import compute_so_normal_equations
from src.inversion_scripts.utils import ensure_float_list
from src.inversion_scripts.make_gridded_posterior import make_gridded_posterior
from src.utilities.config_utils import load_config


def build_lognormal_prior_cov(config, sa, n):
    """Log-space prior covariance for the n lognormal (ROI) state-vector elements.

    A BTR / prior uncertainty is specified in NORMAL (arithmetic) scale-factor space as a
    relative uncertainty u (a coefficient of variation). A lognormal scale factor SF = e^z,
    z ~ N(mu, Sigma_ln), reproduces that same arithmetic relative covariance exactly when

        Sigma_ln = ln(1 + Sa_rel)      (element-wise; the diagonal gives sigma_ln = sqrt(ln(1 + u^2))),

    where Sa_rel is the relative (scale-factor) covariance. We build Sa_rel from the prebuilt BTR
    correlation C and per-element sigma_scale (Sa_rel = diag(sigma) C diag(sigma),
    sigma = PriorError * sigma_scale) and convert with np.log1p, then repair to the nearest PSD.

    The prior mean scale factor is 1, but Levenberg-Marquardt optimises the MEDIAN, so the
    per-element mean->median shift is prior_scale_i = exp(-Sigma_ln,ii / 2) (median = e^{mu} with
    mu = -sigma_ln^2 / 2 keeps E[SF] = 1).

    Returns (lnSa_ROI (n, n), prior_scale (n,)). When no prebuilt BTR covariance is present it
    falls back to the uniform geometric-factor prior (sigma_ln = ln(sa), lnSa = (ln sa)^2 I),
    reproducing the previous behaviour.
    """
    use_btr = str(config.get("OffDiagonalPriorCov", False)).strip().lower() in ("true", "1", "yes")
    if use_btr and os.path.exists("prior_norm_error_covariance.npz"):
        with np.load("prior_norm_error_covariance.npz") as d:
            C = np.asarray(d["covariance"], dtype=float)                 # unit-diagonal correlation
            # length_scale builder omits sigma_scale (uniform PriorError) -> default to ones
            sigma_scale = (np.asarray(d["sigma_scale"], dtype=float)
                           if "sigma_scale" in d.files else np.ones(C.shape[0], dtype=float))
        if C.shape[0] < n or sigma_scale.shape[0] < n:
            raise ValueError(
                f"Prebuilt prior covariance ({C.shape[0]}) smaller than the lognormal ROI "
                f"block ({n}); cannot convert BTR Sa to lognormal space."
            )
        C = C[:n, :n]                                                    # ROI (non-buffer) block, sorted-id order
        sigma_rel = float(sa) * sigma_scale[:n]                          # per-element relative sigma
        Sa_rel = (sigma_rel[:, None] * C) * sigma_rel[None, :]           # relative (scale-factor) covariance
        lnSa = np.log1p(Sa_rel)                                          # Sigma_ln = ln(1 + Sa_rel)
        lnSa = 0.5 * (lnSa + lnSa.T)                                     # symmetrise
        w, V = np.linalg.eigh(lnSa)                                      # nearest-PSD repair in log space
        lnSa = (V * np.clip(w, 1.0e-10, None)) @ V.T
        print(
            "  lognormal prior: converted BTR Sa_rel -> log space via ln(1+Sa_rel) "
            f"(n={n}, median sigma_ln={float(np.median(np.sqrt(np.diag(lnSa)))):.3f})"
        )
    else:
        lnsa_val = float(np.log(float(sa)))                             # geometric-factor fallback
        lnSa = (lnsa_val ** 2) * np.eye(n)
    prior_scale = np.exp(-0.5 * np.diag(lnSa))                          # per-element mean->median shift
    return lnSa, prior_scale


def _load_obs_metadata(inversion_data, start, end):
    """Per-observation lat/lon/dates for the off-diagonal So, row-aligned with the merged K.
    merge_partial_k writes these as obs_metadata.npz (keys lon/lat/dates) in the inversion cwd (the
    same cwd lognormal reads full_jacobian_K.npz / obs_satellite.npz from) AND to
    {inversion_data}/observations/observations_{start}_{end}.npz. Returns (lat, lon, dates), or
    (None, None, None) if unavailable (-> diagonal-only So)."""
    for path in (
        "obs_metadata.npz",                                                    # merge_partial_k relative output (this cwd)
        os.path.join(inversion_data, "observations", f"observations_{start}_{end}.npz"),
    ):
        if not os.path.exists(path):
            continue
        try:
            with np.load(path, allow_pickle=True) as d:
                keys = set(d.files)
                if {"lat", "lon"} <= keys:
                    lat = np.asarray(d["lat"], dtype=float).flatten()
                    lon = np.asarray(d["lon"], dtype=float).flatten()
                    dates = next((d[k] for k in ("dates", "time", "date") if k in keys), None)
                    return lat, lon, dates
        except Exception:
            continue
    return None, None, None


def lognormal_invert(config, state_vector_filepath, jacobian_sf):
    """
    Description:
        Run inversion using lognormal errors following method from eqn 2 of
        Chen et al., 2022 https://doi.org/10.5194/acp-22-10809-2022
        Outputs inversion results to netcdf files.
    Arguments:
        config                [Dict]   : dictionary of config variables
        state_vector_filepath [String] : path to state vector netcdf file
        jacobian_sf           [String] : path to numpy array of scale factors
    """
    results_save_path = f"inversion_result_ln.nc"
    # dictionary to store inversion results
    results_dict = {
        "xhat": [],
        "lnxn": [],
        "S_post": [],
        "A": [],
        "Ja_normalized": [],
        "prior_err": [],
        "obs_err": [],
        "gamma": [],
        "prior_err_bc": [],
        "prior_err_oh": [],
        "prior_err_buffer": [],
    }

    state_vector = xr.load_dataset(state_vector_filepath).squeeze()
    state_vector_labels = state_vector["StateVector"]

    # used to determine convergence of xn 5e-3 is .5%
    convergence_threshold = 5e-3

    # Load in the observation and background data
    ds = np.load("obs_satellite.npz")
    y = np.array(ds["obs_satellite"])
    ds = np.load("gc_bkgd.npz")
    ybkg = np.array(ds["gc_bkgd"])

    # We only solve using lognormal errors for state vector elements
    # within the domain of interest, not the buffer elements, the
    # BC elements, or OH optimization. So, to do this we split K into
    # two matrices, one for the lognormal elements, and one for the
    # normal elements.
    optimize_bcs = config["OptimizeBCs"]
    optimize_oh = config["OptimizeOH"]
    is_regional = config["isRegional"]
    if optimize_oh:
        if is_regional:
            OH_element_num = 1
        else:
            OH_element_num = 2
    else:
        OH_element_num = 0
    BC_element_num = 4 if optimize_bcs else 0
    num_sv_elems = (
        int(state_vector_labels.max().item()) + BC_element_num + OH_element_num
    )
    num_buffer_elems = int(config["nBufferClusters"])
    num_normal_elems = num_buffer_elems + BC_element_num + OH_element_num
    ds = np.load("full_jacobian_K.npz")
    K_temp = np.array(ds["K"]) * 1e9

    # Apply scaling matrix if using precomputed Jacobian
    if jacobian_sf is not None:
        scale_factors = np.load(jacobian_sf)
        # apply unit scaling for BC elements and OH elements if using
        if optimize_bcs or optimize_oh:
            scale_factors = np.append(
                scale_factors, np.ones(BC_element_num + OH_element_num)
            )
        reps = K_temp.shape[0]
        scaling_matrix = np.tile(scale_factors, (reps, 1))
        K_temp *= scaling_matrix

    # Define Sa, gamma, So, and Sa_bc values to iterate through
    prior_errors = ensure_float_list(config["PriorError"])
    sa_buffer_elems = ensure_float_list(config["PriorErrorBufferElements"])
    sa_bc_vals = ensure_float_list(config["PriorErrorBCs"]) if optimize_bcs else [0.0]
    sa_oh_vals = ensure_float_list(config["PriorErrorOH"]) if optimize_oh else [0.0]
    gamma_vals = ensure_float_list(config["Gamma"])
    obs_err_keys = [
        f"so_{obs_err}" for obs_err in ensure_float_list(config["ObsError"])
    ]

    so_dict = np.load("so_super.npz")

    # Calculate the difference between tropomi and the background
    # simulation, which has no emissions
    y_ybkg_diff = y - ybkg
    # merge_partial_k stores obs_satellite / gc_bkgd as 1-D (m,) vectors; the Levenberg-Marquardt
    # code below operates on column vectors (m, 1). reshape(-1, 1) yields the column vector the old
    # np.asmatrix + swapaxes contract produced, and is robust to a legacy (1, m) save as well.
    ybkg = ybkg.reshape(-1, 1)
    y = y.reshape(-1, 1)
    y_ybkg_diff = y_ybkg_diff.reshape(-1, 1)

    # fixed kappa of 10 following Chen et al., 2022 https://doi.org/10.5194/acp-22-10809-2022
    kappa = 10

    # iterate through different combination of gamma, lnsa, and sa_bc
    # TODO: parallelize this once we allow vectorization of these values
    combinations = list(
        product(
            gamma_vals,
            prior_errors,
            obs_err_keys,
            sa_bc_vals,
            sa_buffer_elems,
            sa_oh_vals,
        )
    )

    # --- Observation-error So setup (SOLVER-INDEPENDENT: the same So the normal/softplus path uses) ---
    # Diagonal: the residual-error (REM) diagonal when UseResidualObsError, else the parametric super-ob So.
    # Off-diagonal: the two-exponential same-day correlation (config OffDiagonalObsCov*), applied EXACTLY by
    # the shared compute_so_normal_equations operator when per-observation lat/lon/dates are available.
    start_str, end_str = str(config["StartDate"]), str(config["EndDate"])
    # The REM So and obs metadata live under {OutputPath}/{RunName}/inversion_data (absolute) -- the
    # SAME location invert.py reads (load_merged_jacobian_products / build_prior_covariance).
    inversion_data = os.path.join(
        os.path.expandvars(config["OutputPath"]), str(config["RunName"]), "inversion_data"
    )
    so_rem = None
    if str(config.get("UseResidualObsError", False)).strip().lower() in ("true", "1", "yes"):
        rem_path = os.path.join(inversion_data, "obs_error_covariance", f"so_{start_str}_{end_str}.npz")
        if os.path.exists(rem_path):
            with np.load(rem_path) as d:
                so_rem = np.asarray(d["so"] if "so" in d.files else d[d.files[0]], dtype=float).flatten()
            print(f"  So diagonal = residual-error (REM) method: {rem_path} (n={so_rem.size})")
        else:
            print(f"  UseResidualObsError set but {rem_path} not found; using parametric super-ob So.")
    obs_lat = obs_lon = obs_dates = None
    so_corr_params = None
    if str(config.get("OffDiagonalObsCov", False)).strip().lower() in ("true", "1", "yes"):
        obs_lat, obs_lon, obs_dates = _load_obs_metadata(inversion_data, start_str, end_str)
        if obs_lat is not None:
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
            print("  Off-diagonal So = two-exponential (shared operator, exact day-blocked block-Thomas).")
        else:
            print("  OffDiagonalObsCov set but obs lat/lon/dates unavailable; using diagonal So only.")

    for gamma, sa, so_key, sa_bc, sa_buffer, sa_oh in combinations:
        # params dict to store hyperparameters
        params = {
            "prior_err": sa,
            "obs_err": float(so_key.removeprefix("so_")),
            "gamma": gamma,
            "prior_err_bc": sa_bc,
            "prior_err_oh": sa_oh,
            "prior_err_buffer": sa_buffer,
        }
        # split K based on whether we are solving for lognormal or normal elements
        # K_ROI is the matrix for the lognormal elements (the region of interest)
        K_ROI = K_temp[:, :-num_normal_elems]
        K_normal = K_temp[:, -num_normal_elems:]
        K_full = np.concatenate((K_ROI, K_normal), axis=1)

        m, n = np.shape(K_ROI)

        # Log-space prior covariance for the ROI (lognormal) elements and the per-element
        # mean->median shift. Levenberg-Marquardt assumes the prior is the MEDIAN, but the prior
        # emissions are the MEAN, so prior_scale_i = exp(-Sigma_ln,ii/2) converts each element.
        # Uses the prebuilt BTR Sa (Sigma_ln = ln(1+Sa_rel)) when available, else a uniform
        # geometric-factor prior (sigma_ln = ln(sa)).
        lnSa_ROI, prior_scale = build_lognormal_prior_cov(config, sa, n)

        # Create base xa and lnxa matrices
        # Note: the resulting xa vector has lognormal elements until the
        # final Buffer, BCs, and OH elements
        xa = np.ones((n, 1)) * 1.0
        lnxa = np.log(xa * prior_scale.reshape(-1, 1))  # convert to median (per element)

        # Create normal elements for buffer, BCs, and OH
        # BC elements are relative to 0 because they are in concentration space
        # Other elements are in scale factor space where 1 is the prior
        xa_normal_buffer = np.ones((num_buffer_elems, 1)) * 1.0
        xa_normal_BCs = np.ones((BC_element_num, 1)) * 0.0
        xa_normal_OH = np.ones((OH_element_num, 1)) * 1.0
        xa_normal = np.concatenate(
            (xa_normal_buffer, xa_normal_BCs, xa_normal_OH), axis=0
        )

        # concatenate normal elements to xa and lnxa
        xa = np.concatenate((xa, xa_normal), axis=0)
        lnxa = np.concatenate((lnxa, xa_normal), axis=0)

        # So diagonal: the shared residual-error (REM) diagonal when available, else the parametric
        # super-ob So. So^-1 is applied by the shared compute_so_normal_equations operator (below).
        so = so_rem if so_rem is not None else np.asarray(so_dict[so_key], dtype=float).flatten()

        # For the buffer elems, BCs, and OH elements
        # we apply a different Sa value
        # In the most basic we only generate Sa for buffer elements
        base_sa_normal = sa_buffer**2 * np.ones(
            (num_normal_elems - (BC_element_num + OH_element_num), 1)
        )

        # conditionally add BC and OH elements
        if optimize_bcs:
            bc_errors = sa_bc**2 * np.ones((BC_element_num, 1))
            base_sa_normal = np.concatenate((base_sa_normal, bc_errors), axis=0)

        if optimize_oh:
            oh_errors = sa_oh**2 * np.ones((OH_element_num, 1))
            # weight the OH term(s) following Maasakkers et al. (2019)
            oh_weight = OH_element_num / (num_normal_elems - OH_element_num)
            oh_errors_constraint = (oh_weight * sa_oh**2) * np.ones((OH_element_num, 1))
            sa_normal = np.concatenate(
                (base_sa_normal, oh_errors), axis=0
            )  # unweighted Sa vector
            sa_normal_constraint = np.concatenate(
                (base_sa_normal, oh_errors_constraint), axis=0
            )  # weighted Sa vector
        else:
            sa_normal = base_sa_normal.copy()
            sa_normal_constraint = base_sa_normal.copy()

        # Assemble the full block-diagonal log-space prior. The ROI (lognormal) block is the
        # BTR-derived Sigma_ln = ln(1+Sa_rel) (full covariance, or the uniform fallback); the
        # buffer/BC/OH block is the normal-space diagonal. Weighted vs unweighted differ only in
        # the OH block (Maasakkers et al. 2019 weighting).
        ntot = n + num_normal_elems
        norm_idx = np.arange(n, ntot)
        lnsa = np.zeros((ntot, ntot))
        lnsa_constraint = np.zeros((ntot, ntot))
        lnsa[:n, :n] = lnSa_ROI
        lnsa_constraint[:n, :n] = lnSa_ROI
        lnsa[norm_idx, norm_idx] = sa_normal.flatten()  # unweighted normal (buffer, BC, OH)
        lnsa_constraint[norm_idx, norm_idx] = sa_normal_constraint.flatten()  # weighted (OH)
        invlnsa = np.linalg.inv(lnsa)
        invlnsa_constraint = np.linalg.inv(lnsa_constraint)

        # we start with lnxa using the prior values (scale factors of ln(1))
        lnxn = lnxa

        # start with arbitrary value for xn_iteration_pct_diff above .05%
        xn_iteration_pct_diff = 1

        # Iterate for calculation of ln(xn) until convergence threshold is met (5e-3)
        # We decompose eqn 2 from chen et al into 4 terms
        # term 1: gamma*K'.T@inv(So)@K'
        # term 2: inv((1+kappa)*inv(ln(sa)))
        # term 3: gamma*K'.T@inv(So)@(y_ybkg_diff - K@xn)
        # term 4: -inv(ln(sa))@(ln(xn) - ln(xa))
        # We can then solve for xn iteratively by doing:
        # ln(xn) = x(n-1) + inv(term1+term2)@(term3 + term4)
        # where x(n-1) is the previous iteration of xn until convergence
        print("Status: Iterating to calculate ln(xn)")

        # Initializing the mean of xn
        xnmean = np.concatenate(
            (np.exp(lnxn[:-num_normal_elems]) / prior_scale.reshape(-1, 1), lnxn[-num_normal_elems:]),
            axis=0,
        )
    
        while xn_iteration_pct_diff >= convergence_threshold:

            # K_prime is the updated jacobian using the new xnmean from the previous iteration
            K_prime = np.concatenate(
                (K_ROI * xnmean[:-num_normal_elems].T, K_normal), axis=1
            )

            # Observation-error normal equations via the SHARED, solver-independent operator (identical
            # So weighting to the normal/softplus path): KTinvSoK = K'^T So^-1 K',
            # KTinvSo_resid = K'^T So^-1 (y - F(x)).  With so_corr_params=None this is the plain diagonal So.
            residual = (y_ybkg_diff - K_full @ xnmean).flatten()
            KTinvSoK, KTinvSo_resid, _ = compute_so_normal_equations(
                K_prime, residual, so, obs_lat, obs_lon, obs_dates, so_corr_params
            )
            gKTinvSoK = gamma * KTinvSoK

            # Compute the next xn_update (Chen et al. 2022, eqn 2)
            term1 = gKTinvSoK
            term2 = (1 + kappa) * invlnsa_constraint
            inv_term = np.linalg.inv(term1 + term2)

            # here xn and K need to be the mean
            term3 = gamma * KTinvSo_resid.reshape(-1, 1)
            # here lnxn and lnxa are the median
            term4 = invlnsa_constraint @ (lnxn - lnxa)

            # put it all together to calculate lnxn_update
            lnxn_update = lnxn + inv_term @ (term3 - term4)

            # Check for convergence
            xn_iteration_pct_diff = max(
                abs(
                    np.exp(lnxn_update[:-num_normal_elems])
                    - np.exp(lnxn[:-num_normal_elems])
                )
                / np.exp(lnxn[:-num_normal_elems])
            )

            lnxn = lnxn_update

            # posterior error covariance matrix (uses unweighted Sa)
            lns = np.linalg.inv(gKTinvSoK + invlnsa)

            # Calculate posterior mean xhat
            dlns = np.diag(lns[:-num_normal_elems, :-num_normal_elems])
            # this xn is the median returned by the inversion
            # needed for \hat x following Hancock et al. 2025, Eq. 6            
            xn = np.concatenate(
                (np.exp(lnxn[:-num_normal_elems]), lnxn[-num_normal_elems:]), axis=0
            )
            # Hancock et al. 2025, Eq. 6
            xnmean = np.concatenate(
                (
                    xn[:-num_normal_elems]
                    * np.expand_dims(np.exp(dlns * (0.5)) * prior_scale, axis=1),
                    xn[-num_normal_elems:],
                )
            )

        print("Status: Done Iterating")

        # NORMAL (linear) averaging kernel for the DOFS diagnostic. The data resolution is a property of
        # the PHYSICAL Jacobian, So, and prior and must NOT carry the log transform (matches the analytical
        # and softplus solvers). Use the physical K^T So^-1 K from the untransformed K_full, and the
        # physical prior covariance: the ROI relative covariance Sa_rel = e^{ln(1+Sa_rel)} - 1 recovered
        # from the log block, plus the already-physical (unweighted) buffer/BC/OH block.
        KTinvSoK_phys, _, _ = compute_so_normal_equations(
            K_full, residual, so, obs_lat, obs_lon, obs_dates, so_corr_params
        )
        Md_phys = gamma * KTinvSoK_phys
        sa_phys = np.zeros((ntot, ntot))
        sa_phys[:n, :n] = np.expm1(lnSa_ROI)
        sa_phys[norm_idx, norm_idx] = sa_normal.flatten()
        w_sp, V_sp = np.linalg.eigh(0.5 * (sa_phys + sa_phys.T))
        sa_phys = (V_sp * np.clip(w_sp, 1.0e-10, None)) @ V_sp.T
        ak = np.linalg.solve(Md_phys + np.linalg.inv(sa_phys), Md_phys)

        # Calculate Ja diagnostic only for domain of interest (ignoring buffer and BC elements)
        # Ja diagnostic is useful for determining regurlarization parameter (gamma)
        Ja = (
            np.transpose(lnxn[:-num_normal_elems] - lnxa[:-num_normal_elems])
            @ invlnsa[:-num_normal_elems, :-num_normal_elems]
            @ (lnxn[:-num_normal_elems] - lnxa[:-num_normal_elems])
        )

        print(
            f"Diagnostics:\n  (Ja: {Ja}, gamma: {gamma}, "
            + f"sa: {sa}, sa_bc: {sa_bc}, sa_oh: {sa_oh_vals} sa_buffer: {sa_buffer})"
        )

        # Append results to results_dict
        xhat = xnmean
        print(f"xhat = {xhat.sum()}")
        results_dict["xhat"].append(xhat.flatten()),
        results_dict["lnxn"].append(lnxn.flatten()),
        results_dict["S_post"].append(lns),
        results_dict["A"].append(ak),
        results_dict["Ja_normalized"].append(Ja.item() / num_sv_elems),
        for k, v in params.items():
            results_dict[k].append(v)

    # Define the default data variables as those with normalized Ja closest to 1
    idx_default_Ja = np.argmin(np.abs(np.array(results_dict["Ja_normalized"]) - 1))
    print(
        f"J_A/n closest to 1: {results_dict['Ja_normalized'][idx_default_Ja]}"
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

    # Create an xarray dataset to store inversion results
    dataset = xr.Dataset()
    for k, v in results_dict.items():
        v = np.array(v)
        dims = ["ensemble"] + [f"nvar{i}" for i in range(1, v.ndim)]
        dataset[k] = (dims, v)

    # save index number of ens member with J_A/n
    # closes to 1 as the default member
    dataset.attrs = {"default_member_index": idx_default_Ja}

    # ensemble dimension to end
    dataset = dataset.transpose(..., "ensemble")

    # Specify attributes
    dataset.xhat.attrs["long_name"] = "Posterior scaling factors"
    dataset.xhat.attrs["units"] = "1"
    dataset.lnxn.attrs["long_name"] = "Posterior log scaling factors"
    dataset.lnxn.attrs["units"] = "1"
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
    dataset.prior_err_buffer.attrs["long_name"] = "Prior error for buffer elements"
    dataset.prior_err_buffer.attrs["units"] = "1"

    # Calculate the mean of the ensemble as the main result
    dataset_mean = dataset.mean(dim="ensemble")

    # Specify attributes
    dataset_mean.xhat.attrs["long_name"] = "Posterior scaling factors"
    dataset_mean.xhat.attrs["units"] = "1"
    dataset_mean.lnxn.attrs["long_name"] = "Posterior log scaling factors"
    dataset_mean.lnxn.attrs["units"] = "1"
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
    dataset_mean.prior_err_buffer.attrs["long_name"] = "Prior error for buffer elements"
    dataset_mean.prior_err_buffer.attrs["units"] = "1"

    dataset.to_netcdf(
        results_save_path.replace(".nc", "_ensemble.nc"),
        encoding={v: {"zlib": True, "complevel": 1} for v in dataset.data_vars},
    )

    dataset_mean.to_netcdf(
        results_save_path,
        encoding={v: {"zlib": True, "complevel": 1} for v in dataset_mean.data_vars},
    )

    # make gridded posterior
    make_gridded_posterior(
        results_save_path.replace(".nc", "_ensemble.nc"),
        state_vector_filepath,
        "gridded_posterior_ln.nc",
    )


if __name__ == "__main__":
    config_path = sys.argv[1]
    state_vector_filepath = sys.argv[2]
    jacobian_sf = None if sys.argv[3] == "None" else sys.argv[3]

    config = load_config(config_path)
    lognormal_invert(config, state_vector_filepath, jacobian_sf)
