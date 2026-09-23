"""Softplus positivity solver (run_softplus).

The softplus solver optimizes in transform space z (x = softplus(z)) so every region-of-interest scale
factor stays strictly positive. In the near-linear regime (comfortably positive posterior) it must
reproduce the analytical normal-equation solution; where the analytical solver would go negative, the
softplus solver must keep every scale factor positive.
"""
import numpy as np

from invert import compute_so_normal_equations
from softplus_invert import (
    run_softplus,
    softplus,
    softplus_inverse,
    softplus_posterior_mean,
)


def test_softplus_roundtrips_on_positive_scale_factors():
    # x = softplus(softplus_inverse(x)) for positive scale factors (the meaningful range; the reverse
    # z -> softplus -> inverse is numerically lossy once softplus(z) underflows to ~0).
    x = np.linspace(0.01, 5.0, 60)
    assert np.allclose(softplus(softplus_inverse(x, 0.1), 0.1), x, atol=1e-7)


def test_softplus_and_mean_are_positive():
    z = np.linspace(-5.0, 5.0, 50)
    assert np.all(softplus(z, 0.1) > 0)
    assert np.all(softplus_posterior_mean(z, np.full_like(z, 0.04), 0.1) > 0)


def _linear_problem(n, n_obs, seed, delta_true, prior_err=0.5, so_scale=1.0):
    rng = np.random.RandomState(seed)
    K = rng.randn(n_obs, n)
    so = np.full(n_obs, so_scale)
    xa = np.ones(n)                              # prior scale factors = 1 over the ROI
    dy = K @ np.asarray(delta_true, dtype=float)  # innovation y - F(xA) for a linear forward
    KTinvSoK, KTinvSoy, ytinvSoy = compute_so_normal_equations(K, dy, so, None, None, None, None)
    inv_Sa = np.diag(np.ones(n) / prior_err ** 2)
    return K, xa, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa


def test_matches_analytical_when_comfortably_positive():
    n, n_obs = 5, 400
    delta_true = np.array([0.2, -0.25, 0.1, 0.3, -0.15])   # xhat ~ 1 +/- 0.3 -> positive, non-binding
    K, xa, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa = _linear_problem(n, n_obs, 1, delta_true)
    gamma = 1.0
    delta_an = np.linalg.solve(gamma * KTinvSoK + inv_Sa, gamma * KTinvSoy)
    xhat_an = xa + delta_an
    assert np.all(xhat_an > 0.3)                 # positivity is not binding here

    xhat_sp, _, _, _, diag, n_iter = run_softplus(
        KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa, inv_Sa, n_obs, n, xa, scale=0.1, gamma=gamma
    )
    assert n_iter < 500                          # converged
    assert np.all(xhat_sp > 0)                   # positivity guaranteed
    assert np.allclose(xhat_sp, xhat_an, atol=0.05)   # near-linear regime -> matches the linear solve


def test_returns_normal_averaging_kernel():
    # The reported averaging kernel must be the NORMAL (linear) one -- data resolution
    # A = (gamma K^T So^-1 K + Sa^-1)^-1 (gamma K^T So^-1 K) -- NOT the transform-space kernel, and its
    # trace over the ROI must equal the reported DOFS.
    n, n_obs = 5, 400
    delta_true = np.array([0.2, -0.25, 0.1, 0.3, -0.15])
    K, xa, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa = _linear_problem(n, n_obs, 1, delta_true)
    gamma = 1.0
    _, _, _, A, diag, _ = run_softplus(
        KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa, inv_Sa, n_obs, n, xa, scale=0.1, gamma=gamma
    )
    Md = gamma * KTinvSoK
    A_normal = np.linalg.solve(Md + inv_Sa, Md)      # the linear/analytical averaging kernel
    assert np.allclose(A, A_normal, atol=1e-10)      # transform-independent, matches the analytical solver
    assert np.isclose(np.trace(A[:n, :n]), diag["DOFS"], atol=1e-6)   # DOFS = trace of the ROI AK


def test_max_iter_backstop_warns_loudly(capsys):
    # Default max_iter is None (run to convergence, no cap). A set max_iter is only a safety backstop:
    # reaching it without convergence must print a loud warning and return, not stop silently.
    n, n_obs = 5, 400
    delta_true = np.array([0.2, -0.25, 0.1, 0.3, -0.15])
    K, xa, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa = _linear_problem(n, n_obs, 1, delta_true)
    # max_iter=1 cannot converge (the convergence test requires it > 0), so it always hits the backstop
    _, _, _, _, _, n_iter = run_softplus(
        KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa, inv_Sa, n_obs, n, xa, scale=0.1, max_iter=1
    )
    assert n_iter == 1
    assert "did not converge" in capsys.readouterr().out


def test_enforces_positivity_where_linear_goes_negative():
    n, n_obs = 4, 400
    delta_true = np.array([-2.5, 0.2, 0.1, 0.0])  # element 0 drives the linear scale factor negative
    K, xa, KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa = _linear_problem(n, n_obs, 7, delta_true, prior_err=1.0)
    delta_an = np.linalg.solve(KTinvSoK + inv_Sa, KTinvSoy)
    xhat_an = xa + delta_an
    assert np.any(xhat_an < 0)                   # the analytical solver goes negative

    xhat_sp, _, _, _, diag, n_iter = run_softplus(
        KTinvSoK, KTinvSoy, ytinvSoy, inv_Sa, inv_Sa, n_obs, n, xa, scale=0.1
    )
    assert n_iter < 500
    assert np.all(xhat_sp > 0)                   # every scale factor kept strictly positive
