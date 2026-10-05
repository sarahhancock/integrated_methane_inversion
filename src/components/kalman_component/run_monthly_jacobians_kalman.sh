#!/bin/bash
#SBATCH --job-name=imi_monthly_kf
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --output=imi_monthly_kf_%j.out
#SBATCH --error=imi_monthly_kf_%j.err

set -euo pipefail

# sbatch copies the batch script to the scheduler spool dir, so BASH_SOURCE resolves to /var/slurmd
# there -> derive from it only as a fallback, prefer an inherited/explicit IMI_SOURCE_DIR, and export it
# so the child stage jobs (also sbatch'd) inherit the correct path.
IMI_SOURCE_DIR="${IMI_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export IMI_SOURCE_DIR
MONTHLY_DIR="${IMI_SOURCE_DIR}/src/components/kalman_component"
MONTHLY_CONFIG="${MONTHLY_DIR}/monthly_config.py"
PREPARE_MONTH="${MONTHLY_DIR}/prepare_month.sh"
RUN_STAGE="${MONTHLY_DIR}/run_monthly_stage.sh"

CONFIG_FILE="${1:?config path required}"
CONFIG_FILE="$(readlink -f "$CONFIG_FILE")"
END_LIMIT="${2:-}"
NUDGE_FACTOR="${3:-0.2}"
START_OVERRIDE="${4:-}"

cfgget() {
  if [[ $# -gt 1 ]]; then
    python "$MONTHLY_CONFIG" get "$CONFIG_FILE" "$1" --default "$2"
  else
    python "$MONTHLY_CONFIG" get "$CONFIG_FILE" "$1"
  fi
}

# Resources for the per-month STAGE jobs. The prior GEOS-Chem sim runs INLINE inside run_imi.sh
# (run_prior_simulation.sh: ./RunName_0000.run), so the stage job itself must carry the GC resources.
# Without these flags the stage was submitted with SLURM defaults (~1 core) and the prior sim crawled
# and was killed partway (exit 159). Derived from the config; single node for OpenMP.
STAGE_CPUS="$(cfgget RequestedCPUs 32)"
STAGE_MEM="$(cfgget RequestedMemory 32000)"
STAGE_TIME="$(cfgget RequestedTime 0-12:00)"
STAGE_PART="$(cfgget SchedulerPartition sapphire,shared,huce_cascade)"
STAGE_RES=(-N 1 -c "$STAGE_CPUS" --mem="${STAGE_MEM}" -t "$STAGE_TIME" -p "$STAGE_PART")

make_month_config() {
  local start="$1"
  local end="$2"
  local out="$3"
  python "$MONTHLY_CONFIG" write-temp "$CONFIG_FILE" "$out" \
    --set StartDate="$start" \
    --set EndDate="$end" \
    --set NudgeFactor="$NUDGE_FACTOR" >/dev/null
}

submit_and_wait() {
  local job_id="$1"
  local job_state=""
  printf "[%s] Waiting for job %s\n" "$(date '+%F %T')" "$job_id"
  while true; do
    local queue_state
    queue_state=$(squeue -h -j "$job_id" -o "%T" || true)
    if [[ -n "$queue_state" ]]; then
      printf "[%s] Job %s state: %s\n" "$(date '+%F %T')" "$job_id" "$queue_state"
      sleep 30
      continue
    fi
    job_state=$(sacct -j "$job_id" -X -n -o State 2>/dev/null | awk 'NF{print $1; exit}')
    if [[ -z "$job_state" ]]; then
      job_state=$(scontrol show job "$job_id" 2>/dev/null | awk -F= '/JobState=/{print $2; exit}' | awk '{print $1}')
    fi
    case "${job_state%%[*+]*}" in
      COMPLETED)
        printf "[%s] Job %s completed\n" "$(date '+%F %T')" "$job_id"
        return 0
        ;;
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|BOOT_FAIL|OUT_OF_MEMORY)
        printf "Job %s finished with state: %s\n" "$job_id" "$job_state" >&2
        return 1
        ;;
    esac
    sleep 30
  done
}

RUN_NAME="$(cfgget RunName)"
OUTPUT_PATH="$(cfgget OutputPath)"
RUN_ROOT="${OUTPUT_PATH}/${RUN_NAME}"
START="${START_OVERRIDE:-$(cfgget StartDate)}"
INITIAL_START="$START"
if [[ -z "$END_LIMIT" ]]; then
  END_LIMIT="$(cfgget FinalDate "$(cfgget EndDate "")")"
fi

CONDA_ENV="$(cfgget CondaEnv "")"
CONDA_FILE="$(cfgget CondaFile "")"
if [[ -n "$CONDA_FILE" ]]; then
  set +eu   # bashrc/conda init emits non-fatal failures under batch (e.g. `ulimit -c` cannot modify) that
  source "$CONDA_FILE"   # would otherwise trip `set -e` and abort the driver before it starts
  [[ -n "$CONDA_ENV" ]] && conda activate "$CONDA_ENV"
  set -eu
fi
export PYTHONPATH="${IMI_SOURCE_DIR}:${RUN_ROOT}/inversion:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-imi}-${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR"

STATEVECTOR_FILE="${RUN_ROOT}/StateVector.nc"
INV_DIR="${RUN_ROOT}/inversion"
PRIOR_CACHE="${RUN_ROOT}/prior_hemco_runs_output"
KALMAN_OUTPUT_DIR="${RUN_ROOT}/inversion_data/kalman_output"
NUDGED_SF_DIR="${RUN_ROOT}/inversion_data/nudged_scale_factors"
mkdir -p "$KALMAN_OUTPUT_DIR" "$NUDGED_SF_DIR"

LON_MIN_INV="$(cfgget LonMinInvDomain)"
LON_MAX_INV="$(cfgget LonMaxInvDomain)"
LAT_MIN_INV="$(cfgget LatMinInvDomain)"
LAT_MAX_INV="$(cfgget LatMaxInvDomain)"
RES="$(cfgget Res)"
REQUIRE_RESIDUAL_SO="$(cfgget MonthlyKalmanRequireResidualSo true)"
PRESERVE_RESIDUAL_SO="$(cfgget MonthlyKalmanPreserveResidualSo true)"
ADJUST_PRIOR_WITH_SF="$(cfgget MonthlyKalmanAdjustPriorWithScaleFactors true)"
CLIP_NEGATIVE_K="$(cfgget MonthlyKalmanClipNegativeK true)"
MONTHLY_DRY_RUN="$(cfgget MonthlyKalmanDryRun false)"

MONTH_CONFIG="$(mktemp "${RUN_ROOT}/imi_monthly_kf.XXXXXX.yml")"       # shared storage, NOT node-local /tmp:
INVERT_CONFIG="$(mktemp "${RUN_ROOT}/imi_monthly_invert.XXXXXX.yml")"  # these are read by sbatch'd stage jobs on other nodes
trap 'rm -f "$MONTH_CONFIG" "$INVERT_CONFIG"' EXIT

printf "[%s] Running monthly Jacobians and Kalman inversions from %s through %s with nudge_factor=%s\n" \
  "$(date '+%F %T')" "$START" "$END_LIMIT" "$NUDGE_FACTOR"

printf "\n[%s] Phase 1/2: build monthly Jacobians and merged K products\n" "$(date '+%F %T')"
while [[ "$START" < "$END_LIMIT" ]]; do
  END="$(python "$MONTHLY_CONFIG" month-after "$START")"
  SO_FILE="${RUN_ROOT}/inversion_data/so/so_${START}_${END}.npz"
  OBS_FILE="${RUN_ROOT}/inversion_data/observations/observations_${START}_${END}.npz"
  if [[ "${MONTHLY_DRY_RUN,,}" == "true" ]]; then
    printf "\n[%s] [dry-run] Month %s -> %s\n" "$(date '+%F %T')" "$START" "$END"
    printf "[dry-run] require_residual_so=%s preserve_residual_so=%s adjust_prior_with_sf=%s clip_negative_k=%s\n" \
      "$REQUIRE_RESIDUAL_SO" "$PRESERVE_RESIDUAL_SO" "$ADJUST_PRIOR_WITH_SF" "$CLIP_NEGATIVE_K"
    printf "[dry-run] Would prepare Jacobian setup, submit Jacobian stage, prepare inversion setup, and submit K-merge stage\n"
    START="$END"
    continue
  fi
  if [[ "${REQUIRE_RESIDUAL_SO,,}" == "true" && ! -f "$SO_FILE" ]]; then
    printf "Missing residual-error So file: %s\n" "$SO_FILE" >&2
    exit 1
  fi
  if [[ ! -f "$OBS_FILE" ]]; then
    printf "Missing processed observation file: %s\n" "$OBS_FILE" >&2
    exit 1
  fi

  printf "\n[%s] Building Jacobian for %s -> %s\n" "$(date '+%F %T')" "$START" "$END"
  make_month_config "$START" "$END" "$MONTH_CONFIG"
  "$PREPARE_MONTH" "$MONTH_CONFIG" "$RUN_ROOT" jacobian
  JAC_JOB_ID=$(sbatch --parsable --export=ALL "${STAGE_RES[@]}" "$RUN_STAGE" "$MONTH_CONFIG" jacobian)
  printf "[%s] Submitted Jacobian-stage job %s\n" "$(date '+%F %T')" "$JAC_JOB_ID"
  submit_and_wait "$JAC_JOB_ID"

  "$PREPARE_MONTH" "$MONTH_CONFIG" "$RUN_ROOT" inversion
  PREP_JOB_ID=$(sbatch --parsable "${STAGE_RES[@]}" --export=ALL,IMI_PREPARE_INVERSION_ONLY=true,IMI_PRESERVE_EXISTING_SO="$PRESERVE_RESIDUAL_SO" "$RUN_STAGE" "$MONTH_CONFIG" inversion)
  printf "[%s] Submitted merged-product job %s\n" "$(date '+%F %T')" "$PREP_JOB_ID"
  submit_and_wait "$PREP_JOB_ID"

  START="$END"
done

printf "\n[%s] Phase 2/2: run sequential Kalman inversions\n" "$(date '+%F %T')"
START="$INITIAL_START"
while [[ "$START" < "$END_LIMIT" ]]; do
  END="$(python "$MONTHLY_CONFIG" month-after "$START")"
  printf "\n[%s] Inverting month %s -> %s\n" "$(date '+%F %T')" "$START" "$END"
  make_month_config "$START" "$END" "$MONTH_CONFIG"

  if [[ "${MONTHLY_DRY_RUN,,}" == "true" ]]; then
    printf "[dry-run] Would create nudged scale factors with nudge_factor=%s\n" "$NUDGE_FACTOR"
    printf "[dry-run] Would run invert.py with AdjustPriorWithScaleFactors=%s and ClipNegativeK=%s\n" \
      "$ADJUST_PRIOR_WITH_SF" "$CLIP_NEGATIVE_K"
    printf "[dry-run] Would save cumulative posterior scale factors\n"
    START="$END"
    continue
  fi

  python "${MONTHLY_DIR}/make_nudged_scale_factors.py" \
    --start "$START" \
    --end "$END" \
    --initial-start "$INITIAL_START" \
    --prior-cache "$PRIOR_CACHE" \
    --state-vector "$STATEVECTOR_FILE" \
    --inversion-utils-dir "$INV_DIR" \
    --previous-scale-dir "$KALMAN_OUTPUT_DIR" \
    --outdir "$NUDGED_SF_DIR" \
    --nudge-factor "$NUDGE_FACTOR"
  NUDGED_SF="${NUDGED_SF_DIR}/ScaleFactors_${START}.nc"

  N_ELEMENTS="$(python "$MONTHLY_CONFIG" count-elements "$MONTH_CONFIG" "$STATEVECTOR_FILE")"
  python "$MONTHLY_CONFIG" write-temp "${RUN_ROOT}/config_${RUN_NAME}.yml" "$INVERT_CONFIG" \
    --set StartDate="$START" \
    --set EndDate="$END" \
    --set NudgedScaleFactorPath="$NUDGED_SF" \
    --set AdjustPriorWithScaleFactors="$ADJUST_PRIOR_WITH_SF" \
    --set ClipNegativeK="$CLIP_NEGATIVE_K" >/dev/null

  (
    cd "$INV_DIR"
    python invert.py "$INVERT_CONFIG" "$N_ELEMENTS" data_converted "./inversion_result_kalman_${START}.nc" \
      "$LON_MIN_INV" "$LON_MAX_INV" "$LAT_MIN_INV" "$LAT_MAX_INV" "$RES" None "$STATEVECTOR_FILE"
    python make_gridded_posterior.py "./inversion_result_kalman_${START}.nc" "$STATEVECTOR_FILE" "./gridded_posterior_kalman_${START}.nc"
  )

  python "${MONTHLY_DIR}/save_cumulative_scale_factors.py" \
    --gridded-posterior "${INV_DIR}/gridded_posterior_kalman_${START}.nc" \
    --nudged-scale-factors "$NUDGED_SF" \
    --out "${KALMAN_OUTPUT_DIR}/ScaleFactors_${START}.nc"

  START="$END"
done

printf "[%s] Finished monthly Jacobian and Kalman workflow through %s\n" "$(date '+%F %T')" "$END_LIMIT"
