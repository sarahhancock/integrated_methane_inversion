"""Shared, solver-independent observation-error normal-equation operator.

This module builds the observation-error normal-equation products

    KTinvSoK = K^T So^-1 K
    KTinvSoy = K^T So^-1 (y - F(xA))
    ytinvSoy = (y - F(xA))^T So^-1 (y - F(xA))

for a single, solver-independent observation-error covariance So (used identically by the
analytical, softplus, and lognormal solvers). So is either a plain per-observation diagonal
(obs_error) or that diagonal combined with a same-day two-exponential off-diagonal spatial
correlation plus an adjacent-day (lag-1) temporal correlation, applied EXACTLY by a day-blocked
block-Thomas solve.

It lives in its own module (only numpy + scipy) so it can be imported and unit-tested without the
satellite/plotting import chain (cartopy, GOOPy) that invert.py pulls in; invert.py re-exports these
functions so there is a SINGLE source of truth for the So weighting.
"""

import numpy as np


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
