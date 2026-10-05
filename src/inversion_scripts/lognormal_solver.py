"""Reusable lognormal (Chen et al. 2022) Levenberg-Marquardt solver on FIXED normal-equation products.

`run_lognormal` is the single source of truth for the lognormal emissions solve, shared by the IMI driver
(lognormal_invert.py) and the South America research pipeline. It takes the observation-error
normal-equation products assembled ONCE by the shared So operator (compute_so_normal_equations),

    M    = K^T So^-1 K                     (physical Jacobian K; So applied ONCE)
    v_bg = K^T So^-1 (y - y_background)     (innovation vs the emission-free BACKGROUND simulation)

and iterates the Chen et al. (2022, eqn 2) update in log space using the algebraic identity

    K'^T So^-1 K'  =  M (x) outer(sc, sc)          (sc scales K's columns each iteration)
    K'^T So^-1 r   =  sc * (v_bg - M @ xfield)

so So^-1 is applied ONCE rather than re-applied every iteration. That is what makes the lognormal solver
practical with the off-diagonal (correlated) So, whose day-blocked block-Thomas solve is expensive.

Bookkeeping (mean-preserving prior; Chen et al. 2022 + Hancock et al. 2025):
  The prior EMISSION MEAN is the inventory (scale factor 1). Because the emissions are lognormal, the prior
  MEDIAN scale factor is prior_scale_i = exp(-0.5 sigma_ln,ii^2) < 1, and lnxa = log(prior_scale) is the
  median-centered log prior. The solver optimizes the MEDIAN (log state lnxn) and reports the MEAN field
  median * exp(0.5 diag S_post) (Hancock et al. 2025 Eq. 6) via the lagged offset c. prior_scale is applied
  EXACTLY ONCE (through the misfit factor sc = prior_scale * exp(lnxn + c)); at the prior the mean field is
  prior_scale * exp(lnxn + c) = prior_scale * exp(lnxa + 0.5 sigma^2)... i.e. exactly 1 (the inventory).

This is verified against compute_table_2yr.py's (independently written, kmedian-checked) lognormal_solve to
machine precision, and it fixes the previous lognormal_invert.py driver, which applied prior_scale twice and
so biased the posterior mean low by a factor of prior_scale. numpy only (imports without cartopy/GOOPy).
"""

import numpy as np


def run_lognormal(
    M, v_bg, invlnsa, invlnsa_constraint, lnxa, prior_scale, n_roi, num_normal_elems,
    gamma=1.0, kappa=10.0, convergence_threshold=5e-3, max_iter=300,
    mean_fit=True, mean_relaxation=0.5, mean_every=1, medjac=False, clip_lnxn=(-40.0, 10.0),
    sa_phys=None,
):
    """Lognormal L-M solve on fixed normal-equation products (Chen et al. 2022, eqn 2).

    Arguments
      M                   (ntot, ntot) : K^T So^-1 K (physical Jacobian; So applied once).
      v_bg                (ntot,)      : K^T So^-1 (y - y_background) (innovation vs the emission-free run).
      invlnsa             (ntot, ntot) : inverse log-space prior covariance (unweighted; posterior + Ja).
      invlnsa_constraint  (ntot, ntot) : inverse log-space CONSTRAINT prior covariance (OH-weighted); used in
                                         the (1+kappa) Levenberg-Marquardt Hessian and the prior term. Equal to
                                         invlnsa when there is no OH weighting.
      lnxa                (ntot, 1)    : median-centered log prior, log(prior_scale) on the ROI block (=-0.5
                                         sigma_ln^2 when xA=1) and 0 on the normal (buffer/BC/OH) block.
      prior_scale         (n_roi,)     : per-ROI-element mean->median shift exp(-0.5 diag lnSa_ROI) = exp(lnxa_ROI).
      n_roi               int          : number of ROI (lognormal) state-vector elements.
      num_normal_elems    int          : number of normal (buffer/BC/OH) elements = ntot - n_roi.
      gamma, kappa        floats       : regularization strength and L-M damping (Chen et al. 2022).
      convergence_threshold float      : relative change of the fitted ROI field to stop (default 5e-3).
      max_iter            int          : hard backstop on iterations.
      mean_fit            bool         : fit the posterior MEAN (median*exp(c)) inside the L-M misfit (True), or
                                         fit the MEDIAN and leave the caller to inflate post hoc (False -> c==0).
      mean_relaxation     float in (0,1] : under-relaxation of the lagged offset c=0.5 diag(S_post) (mean_fit only).
                                         <1 damps the fixed-point iteration for stability; the converged result is
                                         unchanged. Ignored when mean_fit is False.
      mean_every          int >= 1     : refresh the lagged offset c (the O(n^3) diag(inv(gamma Mp + Sa^-1)),
                                         benchmarked as the single largest per-iteration cost) only every
                                         mean_every iterations during the bulk, then every iteration once the
                                         field is within convergence_threshold (so the final convergence uses a
                                         fresh c each step, as at mean_every=1). 1 (default) is the exact original
                                         schedule (c refreshed every iteration); >1 trades a small perturbation of
                                         loosely-constrained cells (within the convergence band) for ~2x fewer
                                         diag(inv) solves. Borrowed from run_softplus's mean_every.
      medjac              bool         : linearize the Jacobian at the MEDIAN (diagnostic; residual still on the
                                         mean). Default False (mean Jacobian).
      clip_lnxn           (lo, hi) or None : numerical guard clipping the ROI log state each iteration (no-op for
                                         well-posed inversions).
      sa_phys             (ntot, ntot) or None : PHYSICAL (untransformed) prior covariance for the linear (normal)
                                         averaging kernel / DOFS. When None the AK is skipped (A is None) and the
                                         caller computes its own.

    Returns dict:
      xhat   (ntot, 1) : posterior MEAN scale factors (the summable field; == median when mean_fit is False).
      x_med  (ntot, 1) : posterior MEDIAN scale factors (exp(lnxn_ROI); lnxn for normal elems).
      lnxn   (ntot, 1) : posterior median log scale factors (inventory-relative).
      c      (ntot, 1) : converged log-space median->mean offset (0 on normal elems; 0 when mean_fit is False).
      S_post (ntot, ntot) : posterior error covariance (log space, unweighted prior).
      A      (ntot, ntot) or None : linear (normal) averaging kernel from sa_phys.
      Ja     float : prior cost on the ROI block (log space).
      n_iter int   : iterations run.
    """
    nn = num_normal_elems
    ne = n_roi
    lnxa = np.asarray(lnxa, dtype=float).reshape(-1)
    v_bg = np.asarray(v_bg, dtype=float).reshape(-1)
    prior_scale = np.asarray(prior_scale, dtype=float).reshape(-1)

    s = prior_scale                                   # = exp(lnxa[:ne]); the prior median scale
    sf = np.concatenate((s, np.ones(nn)))             # emission-SF factor for the MEAN field
    # Iterate in median-prior-relative log space. The ROI block is centered at 0 (prior_scale is folded into
    # sc), so its prior center is 0; the NORMAL (buffer/BC/OH) block is linear and centered at its own prior
    # lnxa[ne:] (0 for a BC-only tail, but e.g. buffer/OH SF priors of 1 in the IMI driver).
    lnxa_center = lnxa.copy(); lnxa_center[:ne] = 0.0
    lnxn = lnxa_center.copy()
    c = np.zeros(ne + nn)                             # log-space median->mean offset (0 for normal elems)
    Mp = None
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        xn = lnxn.copy(); xn[:ne] = np.exp(lnxn[:ne] + c[:ne])          # MEAN field entering the misfit
        sc = np.concatenate((s * xn[:ne], np.ones(nn)))                # sc = prior_scale * mean_ROI
        scj = sc if not medjac else np.concatenate((s * np.exp(lnxn[:ne]), np.ones(nn)))
        Mp = M * np.outer(scj, scj)                                    # = K'^T So^-1 K'
        g3 = scj * (v_bg - M @ (sf * xn))                              # = K'^T So^-1 (y_bg - K@meanfield)
        gMp = gamma * Mp
        lnxn_new = lnxn + np.linalg.solve(
            gMp + (1.0 + kappa) * invlnsa_constraint, gamma * g3 - invlnsa_constraint @ (lnxn - lnxa_center)
        )
        if clip_lnxn is not None:
            lnxn_new[:ne] = np.clip(lnxn_new[:ne], clip_lnxn[0], clip_lnxn[1])
        step = np.max(
            np.abs(np.exp(lnxn_new[:ne] + c[:ne]) - np.exp(lnxn[:ne] + c[:ne]))
            / np.maximum(np.exp(lnxn[:ne] + c[:ne]), 1e-9)
        )
        lnxn = lnxn_new
        # Lagged median->mean offset c = 0.5 diag(S_post). The diag(inv(gMp + invlnsa)) is a FULL O(n^3)
        # inverse and (benchmarked at n~1300, SA) the single largest per-iteration cost (~50% of the loop).
        # With mean_every>1 it is refreshed only every mean_every iterations during the bulk, and then every
        # iteration once the field is within convergence_threshold, so the FINAL convergence uses a fresh c
        # each step -- the same schedule as mean_every=1 near the fixed point. mean_every=1 (default) is the
        # exact original behavior (c refreshed every iteration; the else-branch below is never taken).
        c_change = 0.0
        if mean_fit and (it % mean_every == 0 or (mean_every > 1 and step < convergence_threshold)):
            c_target = 0.5 * np.clip(np.diag(np.linalg.inv(gMp + invlnsa))[:ne], 0.0, None)
            c_prev = c[:ne].copy()
            c[:ne] = (1.0 - mean_relaxation) * c[:ne] + mean_relaxation * c_target
            c_change = float(np.max(np.abs(c[:ne] - c_prev))) if ne else 0.0
        # At mean_every==1 (or mean_fit=False) require only the field step, exactly as before; for mean_every>1
        # also require the freshly-refreshed c to be stationary so we never stop on a stale mean correction.
        if it > 0 and step < convergence_threshold and (
            mean_every == 1 or not mean_fit or c_change < convergence_threshold
        ):
            break

    # median-prior-relative -> inventory-relative physical log SF (median)
    lnxn[:ne] = lnxa[:ne] + lnxn[:ne]
    x_med = lnxn.copy(); x_med[:ne] = np.exp(lnxn[:ne])
    x_mean = lnxn.copy(); x_mean[:ne] = np.exp(lnxn[:ne] + c[:ne])

    # posterior error covariance at the converged fitted field
    sc = (np.concatenate((x_med[:ne], np.ones(nn))) if (medjac or not mean_fit)
          else np.concatenate((x_mean[:ne], np.ones(nn))))
    Mp = M * np.outer(sc, sc)
    S_post = np.linalg.inv(gamma * Mp + invlnsa)

    # Linear (normal) averaging kernel for the DOFS diagnostic: physical Jacobian, physical prior, NO log
    # transform (matches the analytical and softplus solvers). Caller supplies the physical prior cov.
    A = None
    if sa_phys is not None:
        Md_phys = gamma * M
        sa_phys = np.asarray(sa_phys, dtype=float)
        w_sp, V_sp = np.linalg.eigh(0.5 * (sa_phys + sa_phys.T))
        sa_phys = (V_sp * np.clip(w_sp, 1.0e-10, None)) @ V_sp.T
        A = np.linalg.solve(Md_phys + np.linalg.inv(sa_phys), Md_phys)

    # Ja on the ROI block (log space, unweighted prior; penalty on lnxn - median-centered prior)
    d = lnxn[:ne] - lnxa[:ne]
    Ja = float(d @ (invlnsa[:ne, :ne] @ d))

    return {
        "xhat": x_mean.reshape(-1, 1),
        "x_med": x_med.reshape(-1, 1),
        "lnxn": lnxn.reshape(-1, 1),
        "c": c.reshape(-1, 1),
        "S_post": S_post,
        "A": A,
        "Ja": Ja,
        "n_iter": n_iter,
    }
