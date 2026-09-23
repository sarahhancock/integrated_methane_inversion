"""BTR -> lognormal prior covariance conversion (build_lognormal_prior_cov).

A relative uncertainty u in arithmetic scale-factor space maps to log space via
Sigma_ln = ln(1 + Sa_rel), so a lognormal scale factor SF = e^z (z ~ N(mu, Sigma_ln)) reproduces the
same arithmetic relative covariance exactly, with the per-element mean->median shift exp(-sigma_ln^2/2).
"""
import numpy as np
import pytest

lognormal_invert = pytest.importorskip("lognormal_invert")
build_lognormal_prior_cov = lognormal_invert.build_lognormal_prior_cov


def _write_prior_npz(path, C, sigma_scale):
    np.savez(
        str(path),
        covariance=C,
        sigma_scale=sigma_scale,
        state_vector_ids=np.arange(1, C.shape[0] + 1),
    )


def test_fallback_without_npz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # No OffDiagonalPriorCov / no prebuilt npz -> geometric-factor fallback lnSa = (ln sa)^2 I.
    lnSa, prior_scale = build_lognormal_prior_cov({}, 2.0, 4)
    assert np.allclose(lnSa, (np.log(2.0) ** 2) * np.eye(4))
    assert np.allclose(prior_scale, np.exp(-0.5 * np.diag(lnSa)))


def test_diagonal_btr_cv_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 5
    _write_prior_npz(tmp_path / "prior_norm_error_covariance.npz", np.eye(n), np.ones(n))
    u = 0.5
    lnSa, prior_scale = build_lognormal_prior_cov({"OffDiagonalPriorCov": True}, u, n)
    # diagonal sigma_ln^2 = ln(1 + u^2); off-diagonal zero (C = I)
    assert np.allclose(np.diag(lnSa), np.log(1 + u ** 2))
    assert np.allclose(lnSa - np.diag(np.diag(lnSa)), 0.0)
    # per-element mean->median shift
    assert np.allclose(prior_scale, np.exp(-0.5 * np.diag(lnSa)))
    # exact CV round-trip: Var[e^z] = e^{s2} - 1 = u^2 with mu = -s2/2 (so E[SF] = 1)
    s2 = float(lnSa[0, 0])
    assert np.isclose(np.exp(s2) - 1.0, u ** 2, atol=1e-12)


def test_offdiagonal_conversion_matches_log1p_with_psd_repair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 3
    C = np.array([[1.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.0]])
    sigma_scale = np.array([1.0, 2.0, 0.5])
    _write_prior_npz(tmp_path / "prior_norm_error_covariance.npz", C, sigma_scale)
    sa = 0.3
    lnSa, _ = build_lognormal_prior_cov({"OffDiagonalPriorCov": True}, sa, n)
    # Reproduce the exact documented pipeline: Sa_rel = diag(sig) C diag(sig), lnSa = ln(1+Sa_rel),
    # symmetrise, nearest-PSD repair (eigenvalue clip).
    sig = sa * sigma_scale
    Sa_rel = sig[:, None] * C * sig[None, :]
    expected = np.log1p(Sa_rel)
    expected = 0.5 * (expected + expected.T)
    w, V = np.linalg.eigh(expected)
    expected = (V * np.clip(w, 1.0e-10, None)) @ V.T
    assert np.allclose(lnSa, expected, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(lnSa)) > -1e-9   # PSD


def test_roi_block_truncation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    C = np.eye(6)
    sigma_scale = np.arange(1, 7, dtype=float)
    _write_prior_npz(tmp_path / "prior_norm_error_covariance.npz", C, sigma_scale)
    n = 4  # smaller than the prebuilt covariance -> take the ROI (non-buffer) block
    lnSa, _ = build_lognormal_prior_cov({"OffDiagonalPriorCov": True}, 0.2, n)
    assert lnSa.shape == (n, n)
    assert np.allclose(np.diag(lnSa), np.log1p((0.2 * sigma_scale[:n]) ** 2))


def test_prebuilt_smaller_than_n_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_prior_npz(tmp_path / "prior_norm_error_covariance.npz", np.eye(3), np.ones(3))
    with pytest.raises(ValueError):
        build_lognormal_prior_cov({"OffDiagonalPriorCov": True}, 0.3, 5)
