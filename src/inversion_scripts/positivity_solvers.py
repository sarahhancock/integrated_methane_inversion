"""Softplus positivity solver for the IMI analytical inversion.

The softplus transform maps an unconstrained state z to strictly positive emission
scale factors, so the posterior emissions stay non-negative without a log-barrier
or a bounded quadratic program:

    x = s * log(1 + e^{z/s})     (smooth, and linear for large z)

(The lognormal transform x = exp(z), Chen et al. 2022, is provided separately by
lognormal_invert.py via LognormalErrors=true.)

It is solved with a Levenberg-Marquardt (Gauss-Newton with kappa damping)
iteration that operates purely on the pre-assembled normal-equation products

    KTinvSoK  = K^T So^-1 K              (n_elements x n_elements)
    KTinvSoy  = K^T So^-1 (y - F(x_A))   (n_elements,)
    ytinvSoy  = (y - F(x_A))^T So^-1 (y - F(x_A))   (scalar)

so it inherits whatever observation error covariance So (diagonal or with the
off-diagonal correction) invert.py has already assembled.  Only the
region-of-interest emission elements use the transform; the boundary-condition
and OH elements stay in normal (linear) space.

The posterior mean of a transformed Gaussian is not the transform of the mean, so
with posterior_mean=True the reported scale factors are the true posterior mean
(Gauss-Hermite quadrature), following the convention that the reported emissions
are the posterior mean.

Diagnostics (DOFS, J_A/DOFS, J_O/(m-DOFS)) are reported from the *linear*
averaging kernel A = (gamma K^T So^-1 K + Sa^-1)^-1 (gamma K^T So^-1 K), i.e. the
true data resolution, and are NOT distorted by the transform Jacobian.
"""

import numpy as np

# Gauss-Hermite nodes/weights for the softplus posterior-mean quadrature.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(24)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def softplus(z, scale):
    """x = scale * log(1 + exp(z/scale)); numerically stable via logaddexp."""
    return scale * np.logaddexp(0.0, z / scale)


def softplus_derivative(z, scale):
    """dx/dz = 1 / (1 + exp(-z/scale)) = logistic(z/scale)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z / scale, -500.0, 500.0)))


def softplus_inverse(x, scale):
    """z such that softplus(z, scale) = x, for x > 0."""
    return scale * np.log(np.expm1(np.clip(x / scale, 1e-12, 700.0)))


def softplus_posterior_mean(z, variance, scale):
    """E[softplus(Z)] for Z ~ N(z, variance), by Gauss-Hermite quadrature."""
    sigma = np.sqrt(np.clip(variance, 0.0, None))
    acc = np.zeros_like(np.asarray(z, dtype=float))
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS):
        acc += weight * softplus(z + np.sqrt(2.0) * sigma * node, scale)
    return acc / np.sqrt(np.pi)


def softplus_posterior_mean_derivative(z, variance, scale):
    """d/dz E[softplus(Z)] for Z ~ N(z, variance)."""
    sigma = np.sqrt(np.clip(variance, 0.0, None))
    acc = np.zeros_like(np.asarray(z, dtype=float))
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS):
        acc += weight * softplus_derivative(z + np.sqrt(2.0) * sigma * node, scale)
    return acc / np.sqrt(np.pi)


# --------------------------------------------------------------------------- #
# Diagnostics (linear data resolution; identical convention for every solver)
# --------------------------------------------------------------------------- #
def inversion_diagnostics(delta, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa, gamma, n_obs, n_roi):
    """Return DOFS, J_A/DOFS, J_O/(m-DOFS) using the LINEAR averaging kernel.

    delta = xhat - x_A in scale-factor space (length n_elements).
    DOFS = sum over the region of interest of clip(diag(A), 0, 1),
      A = (gamma KTinvSoK + Sa^-1)^-1 (gamma KTinvSoK)   (linear resolution).
    J_A = delta_roi^T Sa^-1_roi delta_roi                (prior cost).
    J_O = ytinvSoy - 2 delta.KTinvSoy + delta.KTinvSoK.delta   (obs cost, exact).
    """
    Md = gamma * np.asarray(KTinvSoK, dtype=float)
    A_linear = np.linalg.solve(Md + inv_Sa, Md)
    dofs = float(np.clip(np.diag(A_linear)[:n_roi], 0.0, 1.0).sum())

    d = np.asarray(delta, dtype=float)
    J_A = float(d[:n_roi] @ (inv_Sa[:n_roi, :n_roi] @ d[:n_roi]))
    J_O = float(ytinvSoy - 2.0 * (d @ KTinvSoy) + d @ (KTinvSoK @ d))
    return {
        "DOFS": dofs,
        "J_A": J_A,
        "J_A_over_DOFS": J_A / max(dofs, 1e-9),
        "J_O": J_O,
        "J_O_over_m_minus_DOFS": J_O / max(n_obs - dofs, 1e-9),
    }


# --------------------------------------------------------------------------- #
# Softplus solver
# --------------------------------------------------------------------------- #
def run_softplus(
    KTinvSoK,
    KTinvSoy,
    ytinvSoy,
    inv_Sa,
    inv_Sa_constraint,
    n_obs,
    n_roi,
    x_prior,
    scale=0.1,
    gamma=1.0,
    kappa=10.0,
    posterior_mean=True,
    mean_relaxation=0.5,
    mean_every=20,
    max_iter=500,
    tol=5e-3,
):
    """Softplus-positivity inversion in normal-equation (KTinvSoK, KTinvSoy) space.

    Arguments
      KTinvSoK, KTinvSoy, ytinvSoy : pre-assembled normal-equation products.
      inv_Sa            : linear prior error covariance inverse (n_elements^2), for
                          the posterior covariance and diagnostics.
      inv_Sa_constraint : the (optionally OH-weighted) prior inverse used in the
                          Levenberg-Marquardt step (matches invert.py's analytical path).
      n_roi             : number of region-of-interest (softplus) elements; the
                          remaining elements (BC, OH) stay linear.
      x_prior           : full prior state vector (scale factors, length n_elements):
                          1 over the ROI and OH, 0 for concentration-space BC elements.
      scale             : softplus smoothing scale s (smaller -> closer to ReLU).
      gamma, kappa      : regularization factor and LM damping (Chen et al. kappa=10).
      posterior_mean    : report the posterior mean (Gauss-Hermite) rather than the mode.

    Returns xhat, delta, S_post, A, diagnostics, n_iter.
    """
    n = KTinvSoK.shape[0]
    Md = gamma * np.asarray(KTinvSoK, dtype=float)
    vd = gamma * np.asarray(KTinvSoy, dtype=float)

    # Prior in transform space: invert softplus over the ROI, identity elsewhere.
    x_prior = np.asarray(x_prior, dtype=float)
    z_prior = x_prior.copy()
    z_prior[:n_roi] = softplus_inverse(x_prior[:n_roi], scale)

    z = z_prior.copy()
    variance = np.zeros(n)

    def field(state, var):
        scale_factors = state.copy()
        deriv = np.ones(n)
        if posterior_mean:
            scale_factors[:n_roi] = softplus_posterior_mean(state[:n_roi], var[:n_roi], scale)
            deriv[:n_roi] = softplus_posterior_mean_derivative(state[:n_roi], var[:n_roi], scale)
        else:
            scale_factors[:n_roi] = softplus(state[:n_roi], scale)
            deriv[:n_roi] = softplus_derivative(state[:n_roi], scale)
        return scale_factors, deriv

    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        sf, deriv = field(z, variance)
        data_hessian = Md * np.outer(deriv, deriv)
        gradient = deriv * (vd - Md @ (sf - x_prior))
        z_new = z + np.linalg.solve(
            data_hessian + (1.0 + kappa) * inv_Sa_constraint,
            gradient - inv_Sa_constraint @ (z - z_prior),
        )
        sf_new, _ = field(z_new, variance)
        step = np.max(np.abs(sf_new[:n_roi] - sf[:n_roi]) / np.maximum(sf[:n_roi], 1e-9))
        z = z_new
        # Update the posterior-mean variance only periodically (mirrors the lognormal solver's
        # mean-correction c, refreshed every MF_EVERY iterations): holding the variance fixed
        # between updates lets the mode iteration settle.  Updating it every iteration -- the
        # previous behaviour -- chased a moving mean and did not converge on tight priors, which
        # drove the region-of-interest cells to the softplus floor (the cities zeroed out).
        var_change = 0.0
        if posterior_mean and (it % mean_every == 0 or step < tol):
            var_target = np.clip(np.diag(np.linalg.inv(data_hessian + inv_Sa))[:n_roi], 0.0, None)
            var_prev = variance[:n_roi].copy()
            variance[:n_roi] = (1.0 - mean_relaxation) * variance[:n_roi] + mean_relaxation * var_target
            var_change = float(np.max(np.abs(variance[:n_roi] - var_prev)))
        if it > 0 and step < tol and var_change < tol:
            break

    sf, deriv = field(z, variance)
    data_hessian = Md * np.outer(deriv, deriv)
    S_post = np.linalg.inv(data_hessian + inv_Sa)
    A = S_post @ data_hessian

    xhat = sf.copy()
    delta = xhat - x_prior
    diagnostics = inversion_diagnostics(delta, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa, gamma, n_obs, n_roi)
    return xhat, delta, S_post, A, diagnostics, n_iter

