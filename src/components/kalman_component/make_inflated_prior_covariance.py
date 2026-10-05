#!/usr/bin/env python3
"""Sequential (Kalman) covariance propagation with RTPS inflation.

Companion to make_nudged_scale_factors.py: that script relaxes the posterior MEAN toward
the prior between periods; this one relaxes the posterior COVARIANCE (spread) toward the
static Sa0, so the next period's prior error covariance is the previous posterior S_post
inflated back toward climatology (Relaxation-to-Prior-Spread, Whitaker & Hamill 2012; cf.
Pendergrass et al. 2025 Eq. 8, applied here in the analytical / full-covariance form):

    Sa^(t+1) = R S_post^(t) R,   R_ii = (alpha*sigma0_i + (1-alpha)*sigma_a_i) / sigma_a_i,

with sigma_a = sqrt(diag S_post) (posterior std), sigma0 = sqrt(diag Sa0) (static prior std),
alpha in [0,1] (0.7 recommended). The output keeps S_post's data-informed correlation
structure but resets each element's std to alpha*sigma0 + (1-alpha)*sigma_a.

The result is written back in the format invert.py's prebuilt-covariance path expects:
a normalized correlation matrix `covariance` (unit diagonal), matching `state_vector_ids`,
and a per-element `sigma_scale` such that invert.py rebuilds
    Sa = (PriorError * sigma_scale) ⊗ correlation ⊗ (PriorError * sigma_scale) = Sa^(t+1).
For the first period (no previous S_post) it just copies the static baseline covariance.
"""
import argparse
import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-cov", required=True,
                   help="Static Sa0 prior_norm_error_covariance.npz (from build_national_inventory_prior_covariance.py)")
    p.add_argument("--previous-inversion-result",
                   help="Previous period's inversion_result*.nc containing S_post; omit for the first period")
    p.add_argument("--prior-err", type=float, required=True, help="PriorError scalar (emission SF prior sigma)")
    p.add_argument("--alpha", type=float, default=0.7, help="RTPS weight on the static prior spread (default 0.7)")
    p.add_argument("--output", required=True, help="Output prior_norm_error_covariance.npz for the next period")
    return p.parse_args()


def load_S_post(path, n_emis):
    ds = xr.load_dataset(path)
    for name in ("S_post", "Spost", "posterior_covariance", "S_posterior"):
        if name in ds:
            S = np.asarray(ds[name].values, dtype=np.float64)
            if S.ndim != 2 or S.shape[0] != S.shape[1]:
                raise ValueError(f"{name} in {path} is not square: {S.shape}")
            return S[:n_emis, :n_emis]                    # emission block (drop BC tail)
    raise KeyError(f"No S_post-like variable found in {path}; has {list(ds.data_vars)}")


def main():
    a = parse_args()
    base = np.load(a.baseline_cov)
    C0 = np.asarray(base["covariance"], dtype=np.float64)          # normalized correlation
    ids = np.asarray(base["state_vector_ids"])
    n = C0.shape[0]
    scale0 = (np.asarray(base["sigma_scale"], dtype=np.float64)
              if "sigma_scale" in base.files else np.ones(n))
    sigma0 = a.prior_err * scale0                                  # static prior per-element std (Sa0)

    if not a.previous_inversion_result:
        # first period: emit the static baseline unchanged
        np.savez(a.output, covariance=C0.astype(np.float32),
                 state_vector_ids=ids.astype(np.int32), sigma_scale=scale0.astype(np.float32))
        print(f"[first period] wrote static baseline covariance to {a.output}")
        return

    S_post = load_S_post(a.previous_inversion_result, n)
    sigma_a = np.sqrt(np.clip(np.diag(S_post), 1e-12, None))       # posterior per-element std
    R = (a.alpha * sigma0 + (1.0 - a.alpha) * sigma_a) / sigma_a   # RTPS inflation factors
    Sa_next = (R[:, None] * S_post) * R[None, :]                   # = R S_post R

    sigma_next = np.sqrt(np.clip(np.diag(Sa_next), 1e-12, None))   # = alpha*sigma0 + (1-alpha)*sigma_a
    C_next = Sa_next / sigma_next[:, None] / sigma_next[None, :]   # normalized correlation
    np.fill_diagonal(C_next, 1.0)
    C_next = np.clip(0.5 * (C_next + C_next.T), -1.0, 1.0)
    scale_next = sigma_next / a.prior_err                          # so invert.py rebuilds Sa_next

    np.savez(a.output, covariance=C_next.astype(np.float32),
             state_vector_ids=ids.astype(np.int32), sigma_scale=scale_next.astype(np.float32))
    print(f"[RTPS alpha={a.alpha}] inflated S_post -> {a.output}; "
          f"per-element sigma range [{sigma_next.min():.2f}, {sigma_next.max():.2f}] "
          f"(posterior [{sigma_a.min():.2f}, {sigma_a.max():.2f}], static [{sigma0.min():.2f}, {sigma0.max():.2f}])")


if __name__ == "__main__":
    main()
