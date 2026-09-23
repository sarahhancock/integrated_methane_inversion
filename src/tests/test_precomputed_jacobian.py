"""Precomputed-Jacobian scaling of the merged Jacobian.

When PrecomputedJacobian=true, merge_partial_k stores the RAW reference Jacobian, so invert.py applies
the per-element scale factors to the merged K a single time before the merged-path solve (which then
supports the off-diagonal So and softplus/lognormal solvers). Only the leading emission/buffer columns
are scaled; the trailing boundary-condition and OH columns are left unscaled.
"""
import numpy as np
import pytest

from invert import apply_precomputed_sf_to_merged_k, compute_so_normal_equations
from utils import align_obs_rows_with_reference


def test_scales_leading_emission_columns_only():
    K = np.ones((5, 6))
    sf = np.array([2.0, 3.0, 0.5])          # 3 emission elements; cols 3,4,5 are BC/OH
    Ks = apply_precomputed_sf_to_merged_k(K, sf)
    assert np.allclose(Ks[:, 0], 2.0)
    assert np.allclose(Ks[:, 1], 3.0)
    assert np.allclose(Ks[:, 2], 0.5)
    assert np.allclose(Ks[:, 3:], 1.0)      # BC/OH columns unscaled


def test_does_not_mutate_input():
    K = np.ones((3, 4))
    K0 = K.copy()
    apply_precomputed_sf_to_merged_k(K, np.array([2.0, 2.0]))
    assert np.allclose(K, K0)               # returns a new array; input untouched


def test_full_length_scale_factors():
    K = np.arange(12, dtype=float).reshape(3, 4)
    sf = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(apply_precomputed_sf_to_merged_k(K, sf), K * sf[None, :])


def test_single_scaling_not_double():
    # applying once must equal a plain single column scale (guards against double-counting)
    rng = np.random.RandomState(0)
    K = rng.randn(8, 5)
    sf = np.array([1.5, 0.5, 2.0])          # 3 emission cols, 2 trailing BC/OH
    expected = K.copy()
    expected[:, :3] *= sf
    assert np.allclose(apply_precomputed_sf_to_merged_k(K, sf), expected)


def test_too_many_scale_factors_raises():
    with pytest.raises(ValueError):
        apply_precomputed_sf_to_merged_k(np.ones((2, 3)), np.ones(4))


def test_scaled_k_flows_through_so_operator():
    # the scaled merged K must feed the same solver-independent So operator (diagonal here) cleanly
    rng = np.random.RandomState(3)
    K = rng.randn(20, 5)
    dy = rng.randn(20)
    so = np.abs(rng.rand(20)) + 1.0
    sf = np.array([2.0, 0.5, 1.0])          # scale 3 emission cols; 2 trailing unscaled
    Ks = apply_precomputed_sf_to_merged_k(K, sf)
    KTinvSoK, _, _ = compute_so_normal_equations(Ks, dy, so, None, None, None, None)
    assert np.allclose(KTinvSoK, Ks.T @ (Ks * (1.0 / so)[:, None]))


# ---- reference-observation alignment (merge_partial_k + day-level precomputed path) ----

def _obs(lon, lat, count):
    # obs_GC columns: [obs_xch4, gc_xch4, lon, lat, count, ...]; alignment keys on cols 2:.
    return np.array([9.0, 9.0, lon, lat, count])


def test_alignment_is_order_independent():
    # The reference run can store the same scenes in a different order; each current obs must map to
    # its matching reference row (indexing reference K by current positions would misalign).
    ref = np.array([_obs(-102, 31, 3), _obs(-101, 32, 5), _obs(-100, 33, 2)])
    cur = np.array([_obs(-100, 33, 2), _obs(-102, 31, 3)])   # reversed subset
    obs_ind, ref_ind = align_obs_rows_with_reference(cur, ref)
    assert list(obs_ind) == [0, 1]
    assert list(ref_ind) == [2, 0]


def test_alignment_drops_unmatched_current_obs():
    ref = np.array([_obs(-102, 31, 3), _obs(-101, 32, 5)])
    cur = np.array([_obs(-101, 32, 5), _obs(-55, 10, 1)])    # 2nd obs not in reference
    obs_ind, ref_ind = align_obs_rows_with_reference(cur, ref)
    assert list(obs_ind) == [0]
    assert list(ref_ind) == [1]


def test_alignment_no_overlap_returns_empty():
    ref = np.array([_obs(-102, 31, 3)])
    cur = np.array([_obs(10, 10, 1)])
    obs_ind, ref_ind = align_obs_rows_with_reference(cur, ref)
    assert obs_ind.size == 0 and ref_ind.size == 0


def test_alignment_duplicate_scenes_match_one_to_one():
    # two identical-metadata current rows consume two reference rows (deque), not the same one twice
    ref = np.array([_obs(-100, 33, 2), _obs(-100, 33, 2)])
    cur = np.array([_obs(-100, 33, 2), _obs(-100, 33, 2)])
    obs_ind, ref_ind = align_obs_rows_with_reference(cur, ref)
    assert sorted(ref_ind.tolist()) == [0, 1]
