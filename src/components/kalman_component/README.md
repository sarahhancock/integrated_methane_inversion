# Resumable Monthly Kalman Workflow

This extends the existing Kalman component with a resumable monthly workflow
that can be tested against the IMI dev branch without maintaining a separate
project-specific wrapper system.

The workflow is intentionally split into two phases:

1. Prepare prior-run TROPOMI products for the residual-error method.
2. After the residual-error `So` files have been inspected and saved, run
   monthly Jacobians, merge the monthly inversion products, and run sequential
   inversions with nudging toward the prior.

## Scripts

- `run_prepare_obs_products_for_so.sh CONFIG [END_LIMIT] [START_OVERRIDE]`
  runs the monthly `_0000` prior simulation when needed, keeps the prior
  restarts and HEMCO prior emissions, and writes merged observation products
  under `inversion_data/`.

- `run_monthly_jacobians_kalman.sh CONFIG [END_LIMIT] [NUDGE_FACTOR] [START_OVERRIDE]`
  requires existing monthly residual-error `So` files, then runs the monthly
  Jacobian stage, merges `K` while preserving those `So` files, and runs the
  sequential inversions. The default nudging factor is `0.2`.

- `prepare_month.sh CONFIG [RUN_ROOT] [jacobian|inversion]` is the lower-level
  setup step used by the two drivers. It updates dates, keeps `_0000` restarts,
  preserves archived HEMCO prior outputs, refreshes inversion scripts from the
  active IMI checkout, and clears only transient staging files.

- `run_monthly_stage.sh CONFIG [full|jacobian|inversion|obs_products]` creates a
  temporary stage-specific config and calls `run_imi.sh`.

## Resume Behavior

The monthly drivers run one month at a time and wait for each Slurm job to
finish before continuing. Re-running the same command skips existing `_0000`
prior runs when the month-end restart already exists, reuses archived HEMCO
prior output when present, and reuses merged inversion products where the
inversion scripts support it.

## Typical Order

```bash
sbatch src/components/kalman_component/run_prepare_obs_products_for_so.sh config.yml 20200101
```

Inspect the residual-error fit and write monthly `So` files to
`inversion_data/so/so_START_END.npz`, then run:

```bash
sbatch src/components/kalman_component/run_monthly_jacobians_kalman.sh config.yml 20200101 0.2
```

The same drivers can also be dispatched from `run_imi.sh` with:

```yaml
KalmanMode: true
ResumableMonthlyKalman: true
MonthlyKalmanStage: "obs_products"  # or "inversion"
MonthlyKalmanEndDate: 20200101
NudgeFactor: 0.2
MonthlyKalmanDryRun: true
UseSatDiagnOverpass: true
```

When `ResumableMonthlyKalman` is true, `run_imi.sh` calls this component's
monthly driver instead of the original period-based Kalman loop. The nested
one-month IMI stages force `KalmanMode: false` so they run only the requested
Jacobian or inversion work and do not recurse into the Kalman dispatcher.

`UseSatDiagnOverpass: true` switches the Jacobian HISTORY setup from hourly
`SpeciesConc`/`StateMetLevEdge` output to the existing GEOS-Chem
`SatDiagn`/`SatDiagnEdge` overpass collections. This reduces GEOS-Chem output
from 24 hourly full-domain files per day to the local overpass window, while
still writing the full domain at that overpass time. It is not yet the true
sparse lat/lon/time target-file diagnostic; that would require a GEOS-Chem
source-level History extension.

No code is committed by these scripts. They only prepare and run the workflow in
the working tree and configured run directories.
