# IMI branch — updated-error-method integration audit

**Branch:** `south_america_trends_imi_dev` (geoschem/IMI fork, `dev`).
**Question:** does this IMI branch implement the current *production* error method (as in `south_america_trends/compute_table_2yr.py`, `generate_residual_so.py`, `offdiag_so_recipe.py`, and the Nigeria port `nigeria_test/nigeria_production.py`)?
**Verdict:** **Not yet.** Only the observation-error **diagonal** is correct. The off-diagonal Sₒ is the *superseded* taper+cutoff method, the prior Sₐ builders implement a different (non-production) construction, the production positivity solvers (lognormal-MEANFIT, softplus) are absent, the DOFS / Jo/(m−DOFS) diagnostics are missing, and — importantly — the two solver methods the user has explicitly forbidden are present and config-wired.

Audit done read-only by three parallel agents + a direct read of `lognormal_invert.py`. File:line citations below are into `src/inversion_scripts/`.

---

## 1. Observation error Sₒ — `build_residual_obs_covariance.py`

**Diagonal (count-variance): CORRECT.** ✅
- Count-variance model σ²(p)=σ_trans²+σ_ret²[(1−r)/p+r] (L278-281) matches `generate_residual_so.py:29-32`.
- `sk_est` = per-cell within-cell variance of the demeaned residual (L322-325).
- **Key construction** `so = sk_est × (gp/np.nanmax(gp))` then floored (L370-372) — matches `generate_residual_so.main()` line-for-line. This is the exact thing my Nigeria hand-port had gotten wrong; the IMI branch has it right.
- ⚠️ Minor deviation: floor uses the raw `--floor-variance` default 50 (L374) instead of `max(σ_trans²+r·σ_ret², 50)` (ref `generate_residual_so.py:321-326`, ≈58 with fallback params). Low impact; fix for exactness.

**Off-diagonal: WRONG — superseded taper+cutoff.** ❌
- `fit_correlation_model` fits only SINGLE-exponential / gaussian (L471-473); there is no two-exponential `A1e^{−d/L1}+A2e^{−d/L2}` anywhere.
- Hard **375 km cutoff** (`EMPIRICAL_CUTOFF_KM=375.0`, L762) — shorter than one L2 (398 km), so it discards the broad transport tail that is the whole point of the two-component fit.
- The consumer `invert.py` (L290-292) applies the **spherical Wendland taper** `1−1.5u+0.5u³` for the `"empirical"` form — exactly the taper the current method drops.
- **Required:** two-exponential (SA-locked A1=0.385/L1=26, A2=0.459/L2=398), **no taper, no cutoff** (STCUT≈20000). See `offdiag_so_recipe.py` + memory `sa-offdiag-so-twoexp`.

## 2. Prior error Sₐ — `build_full_prior_covariance.py` (+ `build_national_inventory_prior_covariance.py`)

**Implements 0 of 3 production components.** ❌
- `build_full_prior_covariance.py` (2026-05-13) is a generic **Balasus normalized correlation** matrix: `Sa_norm = exp(−d/L) ⊙ cosine_similarity(sector_fractions)`, unit diagonal, scaled by a scalar `PriorError` (L126-133). No BTR, no σ_nat, no rank-1, no ensemble. Keyword scan for `btr|ensemble|wetland|tceq4|rank1|n_eff|r01|wetcap|d2|nugget` → zero matches.
- Newer sibling `build_national_inventory_prior_covariance.py` (2026-07-14, the `PriorCovarianceMethod: national_inventory` path) is closer — it solves one scalar correlation ρ per country×sector to match the BTR aggregate — but is still **one-component** (not two-component TCEQ4), uses a **uniform scalar magnitude** (not per-element σ_nat²E²), has **no data-driven anthro ensemble**, and gives wetlands a **flat per-sector nugget** (Saunois 0.28 + optional amplitude), not the 3-product ensemble field.

**Required (port of `compute_table_2yr.py build_Sa0`, ref L643-724):**
- Two-component TCEQ4: within-country national rank-1 `σ_nat²E_iE_j` (σ_nat so national aggregate = u_BTR, floored **BTRFLOOR=0.30**) + diagonal local `ratio_i²σ_nat²E_i²`, `ratio_i=√(1+(R01²−1)/n_eff)`, **R01=2.5**.
- Data-driven anthro ensemble (ANTHCOV): `diag(σ_sE)·exp(−d/L_s)·diag(σ_sE)`, σ_s = inter-inventory range/d2 (per-sector d2 in `_D2N`), + retained BTR rank-1.
- Data-driven wetland ensemble (WETCOV): 3-product range/d2/Ēbar, clip **[0.15, WETCAP=2.0]**, `diag(σE)exp(−d/L)diag(σE)`, **d2=1.693 (N=3)**. (Nigeria version already built: `nigeria_test/nigeria_wetland_cov.npz`, rel-median 1.21, L=50 km.)

## 3. Solvers — `lognormal_invert.py` + `invert.py`

**Production positivity solvers absent; forbidden solvers present.** ❌ / 🚨
- `lognormal_invert.py`: IMI **stock lognormal** (Chen 2022, γ-sweep, κ=10). Does compute a lognormal posterior mean (L295-302) ≈ MEANFIT in spirit, and uses no QP/barrier. But it is the γ-regularized IMI formulation, not the production direct-calibrated-covariance lognormal, and has **no softplus** and **no Jo/(m−DOFS)**.
- `invert.py`: 🚨 contains **`run_newton_barrier`** (log-barrier positivity, L501/506) and **`run_bounded_qp`** (box-constrained QP via L-BFGS-B bounds, L665/700/734), both config-wired (`InversionMethod`, L1777-1782). **These are the two methods the user has explicitly forbidden ("NEVER bounded-QP / log-barrier").** They should be removed/disabled. `invert.py` has no lognormal/softplus.
- **Diagnostics:** `invert.py` builds the averaging kernel A (L975) and Ja (L977) but **never computes DOFS** (no `trace`/clipped-diag), its "Ja_normalized" is **Ja/n_elements** (L978) not Ja/DOFS, and **Jo/(m−DOFS) is entirely missing**. `lognormal_invert.py` likewise reports only Ja/n. The reference computes DOFS = Σclip(diag(A),0,1), Ja/DOFS, and Jo/(m−DOFS) with the correct denominator.

**Required:** port `lognormal_solve` (MEANFIT) + `softplus_solve` from `nigeria_production.py:62-122`; remove the barrier/QP methods; add DOFS, Ja/DOFS, Jo/(m−DOFS) diagnostics.

---

## Proposed integration order (each with a validation gate)

1. **Off-diagonal Sₒ** → replace single-exp+375 km+taper with the two-exponential no-taper/no-cutoff builder; and add the physical floor to the diagonal. Gate: reproduces `generate_residual_so`/`offdiag_so_recipe` on SA obs.
2. **Prior Sₐ** → port `build_Sa0` (two-component TCEQ4 + ANTHCOV + WETCOV) into the `national_inventory` builder. Gate: per-cell σ + national aggregates match `compute_table_2yr` on SA.
3. **Solvers** → add lognormal-MEANFIT + softplus; remove barrier/QP; add DOFS/Ja-DOFS/Jo-(m−DOFS). Gate: **reproduce SA production MASTERT: prior 116 → post 95.5 Tg/yr, DOFS 287, Ja/DOFS 1.14, Jo/m 1.08, nNeg 0.**
4. Wire config flags in `run_inversion.sh` and document.

**Guardrail:** these are substantial edits to a shared IMI dev branch (incl. removing config-exposed solvers). I have NOT made them — they're staged here for review. The Nigeria test (below) validates the method components independently first.

---

## INTEGRATION STATUS (2026-09-13)

Scope narrowed by the user to **(1) softplus, (2) off-diag Sₐ, (3) off-diag Sₒ**. Hard constraints: IMI-faithful,
no bespoke terminology, **do not break anything else in the IMI**, and **keep the day-by-day streaming obs path**
(for state vectors too large to assemble the full system at once).

### (1) Softplus — DONE + validated ✅
- New module **`src/inversion_scripts/positivity_solvers.py`**: `run_softplus` (Levenberg-Marquardt softplus, x = s·log(1+e^{z/s}),
  posterior MEAN via Gauss-Hermite) + `inversion_diagnostics` (linear DOFS, J_A/DOFS, J_O/(m−DOFS)). Works purely
  in (KᵀSₒ⁻¹K, KᵀSₒ⁻¹dy) space, so it inherits the off-diagonal Sₒ and prebuilt Sₐ that `invert.py` assembles.
- Wired into `invert.py` `solve_inversion_from_k` as `InversionMethod: softplus` (config `SoftplusScale`, default 0.1;
  κ=10). Only the ROI is transformed; BC/OH stay linear.
- **barrier_normal / bounded_qp fully removed** (user: not needed): `run_newton_barrier`, `run_bounded_qp`,
  `eval_barrier_normal_cost` and all their constants/params/config reads are deleted; the dispatch is analytical|softplus
  only and rejects the old method names with a clear error. (Zero `barrier`/`bounded_qp` references remain.)
- Day-by-day streaming path preserved and **guarded to analytical-only** (softplus/off-diag Sₒ require the merged path).
- `positivity_solvers.py` added to the `inversion.sh` copy list.
- Validated end-to-end (synthetic): analytical path unchanged, softplus positive + matches analytical where the
  solution is >0, forbidden methods rejected, `py_compile` clean.

### (3) Off-diagonal Sₒ — DONE + validated ✅
- `build_residual_obs_covariance.py`: added a **two-exponential fit** A1·e^{−d/L1}+A2·e^{−d/L2} (`fit_correlation_model`)
  and made it the **primary** saved model (`functional_form="two_exponential"`), **no taper**, cutoff = **3·L2**
  (captures the transport tail; single-exp + empirical lookup kept as fallbacks).
- `invert.py`: `load_so_corr_params` recognizes `two_exponential`; `build_sparse_so_correction` applies
  A1·e^{−d/L1}+A2·e^{−d/L2} with **no taper** (hard cutoff via the neighbor search). Status prints updated.
- Validated: consumer reproduces the two-exp values with no taper (P=0.283 vs taper-would-give 0.215 at 193 km).
- NOTE: IMI applies off-diag Sₒ as a **first-order Woodbury** correction (So⁻¹≈D⁻¹(I−P)D⁻¹), accurate for weak
  correlation; the two-exp is strong (nearest-neighbor ρ~0.56), so this is approximate vs the production exact
  per-day block solve. The neighbor-K / row-sum-cap / PSD diagnostics already guard stability. Exact block Sₒ⁻¹
  would be a larger, separate change to the assembly.

### (2) Off-diagonal Sₐ — DONE + validated (EXACT) ✅
`build_national_inventory_prior_covariance.py` gains an opt-in **`NationalPriorTwoComponent`** (config, default off =
existing one-component, so existing runs are untouched). `build_two_component_covariance` builds the full
two-component prior: per (country, sector), national rank-1 `snat²·EᵢEⱼ` + local diagonal `snat²·(ratio_i²−1)·Eᵢ²`,
`ratio_i = √(1+(R01²−1)/n_eff_i)` (n_eff = effective native-cell count, accumulated in
`emission_weighted_element_table`), `snat = u_BTR/√(1+Q)` so the national aggregate = u_BTR; summed over sectors;
divided by the total per-element emission → scale-factor covariance.
**The earlier "structural blocker" was wrong.** ANY covariance decomposes exactly as `Sₐ = σ·C·σ` with `σ_i=√(Sₐ_ii)`
and unit-diagonal `C_ij=Sₐ_ij/(σ_iσ_j)` — which is precisely IMI's `(sigma_scale, correlation)` contract. So the full
two-component Sₐ is written with NO approximation and NO change to `invert.py`. Config: `NationalPriorGridNationalRatio`
(R01, default 2.5), `NationalPriorMinUncertainty` (BTR floor, default 0.30). Validated: per (country, sector) national
aggregate = u_BTR exactly (ratio 1.0000), PSD, unit-diagonal correlation, local grid excess present.

---

## Real-data IMI test (2026-09-13)

A full Permian GEOS-Chem run was not feasible in-session, so the modified inversion was exercised on **real cached IMI
Jacobian products** (SA run, 2019-07, ~130k super-obs) through `invert.py solve_inversion_from_k` — the identical code
path a real run uses (`scratchpad/test_imi_realdata.py`):
- **analytical** baseline: unchanged (min emission SF −1.9, i.e. it does go negative — the motivation for softplus).
- **softplus**: min emission SF **+0.07** (positive), converges in 24 iterations, DOFS 95.4 from the **linear** kernel
  (confirms softplus reports the normal DOFS), Jₒ/(m−DOFS) ≈ 1.8.
- **two-exponential off-diag Sₒ (EXACT)**: now applied via `build_offdiag_so_normal_equations` — the production
  day-blocked block-Thomas solve of So=D(I+P)D (per-day dense LU + optional lag-1 temporal), giving exact
  KᵀSo⁻¹K/KᵀSo⁻¹dy, NOT the first-order Woodbury. Real data: "applied EXACTLY ... temporal_rho=0.19", no
  scaling/skipping; softplus stays positive. Validated separately that the block-Thomas equals a brute-force dense
  So⁻¹ to 2.5e-16 (same-day and lag-1 temporal). The first-order Woodbury path remains only as a fallback for the
  weaker empirical/single-exp forms.

### (3) Off-diagonal Sₒ — DONE + validated (EXACT) ✅
The two-exponential model (builder + consumer) is applied with the **exact** production method (day-blocked
block-Thomas), selected automatically when `form=two_exponential` and per-observation lat/lon/**dates** are present
(dates now loaded by `load_merged_jacobian_products` and threaded through). Config: `OffDiagonalObsCovExact`
(default true), `OffDiagonalObsCovTemporalRho` (default 0.0; SA production uses 0.19). No first-order approximation
for the two-exp.

## Real PERMIAN out-of-the-box test (2026-09-13)

Ran the modified inversion on a completed Permian 1-week run's real GEOS-Chem Jacobian data
(`imi_test_runs/Permian_sectoral_ak_vis_test_run13`, 2019-01-01->08, 759 emission + 4 BC elements, 11 TROPOMI
overpasses; K scaled by 1e9 as the loaders do) via `solve_inversion_from_k` (`scratchpad/test_permian.py`):
- **analytical** reproduces run13's existing `inversion_result.nc` (pre-change code) with **corr = 0.9984, rms = 0.008**
  -> the changes do NOT regress the standard IMI analytical inversion.
- **softplus** positive (min emission SF 0.63), converges in 23 iterations.
- **softplus + EXACT two-exponential off-diag So** (day-blocked block-Thomas): applied exactly, positive (min 0.75),
  Jo/(m-DOFS) drops 0.76 -> 0.43 (correlated obs errors correctly down-weighted).
(1 week is the size cached; a fresh 1-month would require re-running the GC Jacobian simulations.)
