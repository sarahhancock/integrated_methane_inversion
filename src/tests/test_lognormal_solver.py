"""Reusable lognormal solver (run_lognormal) on fixed normal-equation products.

run_lognormal assembles the observation-error normal equations M = K^T So^-1 K and v = K^T So^-1 (y-y_bg)
ONCE and iterates the Chen et al. (2022) Levenberg-Marquardt update algebraically, using the identity
K'^T So^-1 K' = M (x) outer(sc, sc). These tests check:

  1. it reproduces, to machine precision, an independent reference implementation of the SAME mean-preserving
     lognormal method (the compute_table_2yr.py / kmedian-checked bookkeeping), for both mean_fit modes;
  2. the mean-preserving property: with no data, the posterior MEAN scale factor is exactly the inventory
     (1.0) -- NOT prior_scale (the bug in the previous lognormal_invert.py driver, which applied prior_scale
     twice and biased the mean low);
  3. the fixed-M/v algebra equals a re-applied dense So^-1.
"""
import numpy as np
import pytest

from src.inversion_scripts.lognormal_solver import run_lognormal


def _reference_solver(M, v_bg, invlnSa, lnxa, ne, nb, kappa=10.0, maxit=300, thr=5e-3,
                      mean_fit=True, mf_omega=0.5, medjac=False):
    """Independent reference: the mean-preserving lognormal L-M bookkeeping (prior mean = inventory),
    written from compute_table_2yr.py's lognormal_solve. v_bg is background-relative; internally the prior
    is centered at the median so prior_scale is applied exactly once."""
    s = np.exp(lnxa[:ne]); sf = np.r_[s, np.ones(nb)]
    lnxn = np.zeros(ne + nb); c = np.zeros(ne + nb)
    it = 0
    for it in range(maxit):
        xn = lnxn.copy(); xn[:ne] = np.exp(lnxn[:ne] + c[:ne]); sc = np.r_[s * xn[:ne], np.ones(nb)]
        scj = np.r_[s * np.exp(lnxn[:ne]), np.ones(nb)] if medjac else sc
        Mp = M * np.outer(scj, scj); g3 = scj * (v_bg - M @ (sf * xn))
        lnxn_new = lnxn + np.linalg.solve(Mp + (1.0 + kappa) * invlnSa, g3 - invlnSa @ lnxn)
        lnxn_new[:ne] = np.clip(lnxn_new[:ne], -40.0, 10.0)
        step = np.max(np.abs(np.exp(lnxn_new[:ne] + c[:ne]) - np.exp(lnxn[:ne] + c[:ne])) /
                      np.maximum(np.exp(lnxn[:ne] + c[:ne]), 1e-9))
        lnxn = lnxn_new
        if mean_fit:
            _ct = 0.5 * np.clip(np.diag(np.linalg.inv(Mp + invlnSa))[:ne], 0.0, None)
            c[:ne] = (1.0 - mf_omega) * c[:ne] + mf_omega * _ct
        if it > 0 and step < thr:
            break
    lnxn[:ne] = lnxa[:ne] + lnxn[:ne]
    x_mean = lnxn.copy(); x_mean[:ne] = np.exp(lnxn[:ne] + c[:ne])
    x_med = lnxn.copy(); x_med[:ne] = np.exp(lnxn[:ne])
    return lnxn, x_med, x_mean, it + 1


def _build_case(seed, ne=25, nb=4, m=300, gamma=1.0):
    rng = np.random.default_rng(seed)
    ntot = ne + nb
    K = rng.normal(size=(m, ntot)) * 0.8
    x_true = np.concatenate([np.exp(rng.normal(0.0, 1.3, ne)), rng.normal(0, 1.5, nb)])
    y_bg = K @ x_true + rng.normal(0, 0.15, m)                          # sat - background
    A = rng.normal(size=(m, m)); Soi = np.linalg.inv(A @ A.T + 0.3 * m * np.eye(m))
    M = K.T @ Soi @ K
    v_bg = K.T @ Soi @ y_bg
    B = rng.normal(size=(ne, ne)); lnSa_ROI = 0.4 * (B @ B.T) / ne + 0.5 * np.eye(ne)
    lnSad = np.clip(np.diag(lnSa_ROI), 0.0, None)
    invlnSa = np.zeros((ntot, ntot)); invlnSa[:ne, :ne] = np.linalg.inv(lnSa_ROI)
    for b in range(nb):
        invlnSa[ne + b, ne + b] = 1.0 / (10.0 ** 2)
    lnxa = np.zeros(ntot); lnxa[:ne] = -0.5 * lnSad
    prior_scale = np.exp(lnxa[:ne])
    return dict(M=M, v_bg=v_bg, invlnSa=invlnSa, lnxa=lnxa, prior_scale=prior_scale,
                ne=ne, nb=nb, gamma=gamma, lnSa_ROI=lnSa_ROI)


@pytest.mark.parametrize("mean_fit", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_run_lognormal_matches_reference(seed, mean_fit):
    c = _build_case(seed)
    maxit = 300 if mean_fit else 80
    ln_ref, xmed_ref, xmean_ref, nit_ref = _reference_solver(
        c["M"], c["v_bg"], c["invlnSa"], c["lnxa"], c["ne"], c["nb"],
        maxit=maxit, mean_fit=mean_fit, mf_omega=0.5)
    assert nit_ref > 15, "case should iterate many times"
    out = run_lognormal(
        c["M"], c["v_bg"], c["invlnSa"], c["invlnSa"], c["lnxa"].reshape(-1, 1), c["prior_scale"],
        c["ne"], c["nb"], gamma=1.0, kappa=10.0, convergence_threshold=5e-3, max_iter=maxit,
        mean_fit=mean_fit, mean_relaxation=0.5, medjac=False, clip_lnxn=(-40.0, 10.0))
    assert np.allclose(out["lnxn"].reshape(-1), ln_ref, rtol=1e-9, atol=1e-11)
    assert np.allclose(out["x_med"].reshape(-1), xmed_ref, rtol=1e-9, atol=1e-11)
    assert np.allclose(out["xhat"].reshape(-1), xmean_ref, rtol=1e-9, atol=1e-11)
    assert out["n_iter"] == nit_ref


def test_prior_mean_is_inventory():
    """With no data (M=0, v=0) the posterior MEAN scale factor must be the inventory (1.0), NOT prior_scale.

    This is the mean-preserving property of the lognormal prior and the regression guard against the
    previous driver's double-prior_scale bug (which gave prior mean == prior_scale ~ 0.6)."""
    rng = np.random.default_rng(0)
    ne, nb = 20, 3; ntot = ne + nb
    B = rng.normal(size=(ne, ne)); lnSa_ROI = 0.4 * (B @ B.T) / ne + 0.6 * np.eye(ne)
    lnSad = np.clip(np.diag(lnSa_ROI), 0.0, None)
    invlnSa = np.zeros((ntot, ntot)); invlnSa[:ne, :ne] = np.linalg.inv(lnSa_ROI)
    for b in range(nb):
        invlnSa[ne + b, ne + b] = 1.0 / (10.0 ** 2)
    lnxa = np.zeros(ntot); lnxa[:ne] = -0.5 * lnSad
    prior_scale = np.exp(lnxa[:ne])
    assert prior_scale.min() < 0.8   # non-trivial mean/median shift
    out = run_lognormal(
        np.zeros((ntot, ntot)), np.zeros(ntot), invlnSa, invlnSa, lnxa.reshape(-1, 1), prior_scale,
        ne, nb, gamma=1.0, mean_fit=True, mean_relaxation=1.0, max_iter=50)
    assert np.allclose(out["xhat"].reshape(-1)[:ne], 1.0, atol=1e-9), \
        "posterior MEAN at the prior must equal the inventory (1.0)"
    assert np.allclose(out["x_med"].reshape(-1)[:ne], prior_scale, atol=1e-9), \
        "posterior MEDIAN at the prior must equal prior_scale"


def test_prior_mean_with_nonzero_normal_block():
    """The IMI driver's normal (buffer/BC/OH) block has NON-zero linear priors (buffer SF=1, OH SF=1, BC=0).
    With no data the ROI mean must be the inventory (1.0) AND the normal block must sit at its own prior."""
    rng = np.random.default_rng(2)
    ne, n_buf, n_bc, n_oh = 15, 3, 4, 1
    nn = n_buf + n_bc + n_oh; ntot = ne + nn
    B = rng.normal(size=(ne, ne)); lnSa_ROI = 0.4 * (B @ B.T) / ne + 0.6 * np.eye(ne)
    lnSad = np.clip(np.diag(lnSa_ROI), 0.0, None)
    sa_normal = np.r_[np.full(n_buf, 0.5 ** 2), np.full(n_bc, 10.0 ** 2), np.full(n_oh, 0.1 ** 2)]
    lnsa = np.zeros((ntot, ntot)); lnsa[:ne, :ne] = lnSa_ROI
    lnsa[np.arange(ne, ntot), np.arange(ne, ntot)] = sa_normal
    invlnsa = np.linalg.inv(lnsa)
    prior_scale = np.exp(-0.5 * lnSad)
    # driver bookkeeping: lnxa = concat(log(prior_scale), [buffer=1, BC=0, OH=1])
    lnxa = np.concatenate([np.log(prior_scale), np.ones(n_buf), np.zeros(n_bc), np.ones(n_oh)]).reshape(-1, 1)
    out = run_lognormal(
        np.zeros((ntot, ntot)), np.zeros(ntot), invlnsa, invlnsa, lnxa, prior_scale, ne, nn,
        gamma=1.0, mean_fit=True, mean_relaxation=1.0, max_iter=50)
    assert np.allclose(out["xhat"].reshape(-1)[:ne], 1.0, atol=1e-9)                 # ROI mean = inventory
    normal_post = out["xhat"].reshape(-1)[ne:]
    assert np.allclose(normal_post[:n_buf], 1.0, atol=1e-9)                          # buffer sits at SF=1
    assert np.allclose(normal_post[n_buf:n_buf + n_bc], 0.0, atol=1e-9)              # BC sits at 0
    assert np.allclose(normal_post[n_buf + n_bc:], 1.0, atol=1e-9)                   # OH sits at SF=1


@pytest.mark.parametrize("mean_every", [2, 5])
def test_mean_every_converges_to_same_answer(mean_every):
    """mean_every>1 refreshes the O(n^3) mean-fit offset c only periodically (then every iter once the field
    is within tolerance). It converges to the SAME fixed point as mean_every=1 (c every iter): the reported
    aggregate posterior mean must be unchanged within the convergence band, and the default (mean_every=1)
    stays bit-identical (guarded by test_run_lognormal_matches_reference)."""
    c = _build_case(3)
    common = dict(gamma=1.0, kappa=10.0, convergence_threshold=5e-3, max_iter=300,
                  mean_fit=True, mean_relaxation=0.5, medjac=False, clip_lnxn=(-40.0, 10.0))
    base = run_lognormal(c["M"], c["v_bg"], c["invlnSa"], c["invlnSa"], c["lnxa"].reshape(-1, 1),
                         c["prior_scale"], c["ne"], c["nb"], mean_every=1, **common)
    per = run_lognormal(c["M"], c["v_bg"], c["invlnSa"], c["invlnSa"], c["lnxa"].reshape(-1, 1),
                        c["prior_scale"], c["ne"], c["nb"], mean_every=mean_every, **common)
    b = base["xhat"].reshape(-1)[:c["ne"]]
    p = per["xhat"].reshape(-1)[:c["ne"]]
    assert abs(p.sum() - b.sum()) / abs(b.sum()) < 1e-3          # aggregate ROI mean unchanged
    assert np.max(np.abs(p - b) / np.maximum(np.abs(b), 1e-9)) < 5e-2   # per-element within the conv band


def test_crux_identity_fixed_mv_equals_reapplied_so():
    """M (x) outer(sc,sc) == K'^T So^-1 K' and sc*(v - M@field) == K'^T So^-1 (y - K@field)."""
    rng = np.random.default_rng(0)
    m, n, nn = 200, 20, 4
    ntot = n + nn
    K = rng.normal(size=(m, ntot)); y = rng.normal(size=m)
    A = rng.normal(size=(m, m)); Soi = np.linalg.inv(A @ A.T + m * np.eye(m))
    field = np.concatenate([np.abs(rng.normal(size=n)) + 0.1, rng.normal(size=nn)]).reshape(-1, 1)
    sc = np.concatenate([field[:n, 0], np.ones(nn)])
    Kp = K * sc[None, :]
    M = K.T @ Soi @ K; v = K.T @ Soi @ y
    assert np.allclose(M * np.outer(sc, sc), Kp.T @ Soi @ Kp, rtol=1e-10, atol=1e-12)
    resid = y - (K @ field).flatten()
    assert np.allclose(sc * (v - (M @ field).flatten()), Kp.T @ Soi @ resid, rtol=1e-10, atol=1e-10)
