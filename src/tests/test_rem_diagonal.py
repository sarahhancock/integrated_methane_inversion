"""Residual-error-method (REM) diagonal So.

So_i = Sk_i * g(counts_i)/g(1), where Sk is the P=1 (individual-observation) error variance and
g(P) is the Chen et al. 2023 super-observation count-variance model. A single retrieval leaves the
variance unchanged; averaging more retrievals reduces it monotonically. This guards the
individual-observation REM (the fix that must not silently regress to super-ob variance)."""
import numpy as np
from build_obs_error_covariance import build_diagonal_so, residual_count_variance

# default REM fit parameters (build_obs_error_covariance CLI defaults)
R, SIG_R, SIG_T = 0.23, 13.30, 4.13


def test_single_observation_unchanged():
    sk = np.array([100.0, 50.0])
    counts = np.array([1.0, 1.0])
    so = build_diagonal_so(sk, counts, R, SIG_R, SIG_T, floor_variance=0.0)
    assert np.allclose(so, sk)  # g(1)/g(1) = 1


def test_more_counts_reduce_variance():
    sk = np.full(3, 100.0)
    counts = np.array([1.0, 4.0, 16.0])
    so = build_diagonal_so(sk, counts, R, SIG_R, SIG_T, floor_variance=0.0)
    assert so[0] > so[1] > so[2]        # averaging reduces the error variance
    assert np.isclose(so[0], sk[0])     # P=1 unchanged (individual-obs Sk)


def test_count_variance_is_monotone_decreasing():
    v1 = residual_count_variance(1.0, R, SIG_R, SIG_T)
    v10 = residual_count_variance(10.0, R, SIG_R, SIG_T)
    vinf = residual_count_variance(1e6, R, SIG_R, SIG_T)
    assert v1 > v10 > vinf
    # the P->inf floor is the transport term (retrieval noise averages out, correlated part remains)
    assert np.isclose(vinf, SIG_T ** 2 + SIG_R ** 2 * R, rtol=1e-3)
