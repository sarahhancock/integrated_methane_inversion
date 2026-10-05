#!/bin/bash
#SBATCH --job-name=imi_monthly_obs
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --output=imi_monthly_obs_%j.out
#SBATCH --error=imi_monthly_obs_%j.err

set -euo pipefail

# sbatch copies the batch script to the scheduler spool dir, so BASH_SOURCE resolves to /var/slurmd there
# -> prefer an inherited/explicit IMI_SOURCE_DIR, fall back to BASH_SOURCE, and export for child jobs.
IMI_SOURCE_DIR="${IMI_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export IMI_SOURCE_DIR
MONTHLY_DIR="${IMI_SOURCE_DIR}/src/components/kalman_component"
MONTHLY_CONFIG="${MONTHLY_DIR}/monthly_config.py"
PREPARE_MONTH="${MONTHLY_DIR}/prepare_month.sh"
RUN_STAGE="${MONTHLY_DIR}/run_monthly_stage.sh"

CONFIG_FILE="${1:?config path required}"
CONFIG_FILE="$(readlink -f "$CONFIG_FILE")"
END_LIMIT="${2:-}"
START_OVERRIDE="${3:-}"

cfgget() {
  if [[ $# -gt 1 ]]; then
    python "$MONTHLY_CONFIG" get "$CONFIG_FILE" "$1" --default "$2"
  else
    python "$MONTHLY_CONFIG" get "$CONFIG_FILE" "$1"
  fi
}

make_month_config() {
  local start="$1"
  local end="$2"
  local out="$3"
  python "$MONTHLY_CONFIG" write-temp "$CONFIG_FILE" "$out" \
    --set StartDate="$start" \
    --set EndDate="$end" >/dev/null
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
REQUESTED_CPUS="$(cfgget RequestedCPUs 1)"
REQUESTED_MEMORY="$(cfgget RequestedMemory 8000)"
REQUESTED_TIME="$(cfgget RequestedTime "0-12:00")"
SCHEDULER_PARTITION="$(cfgget SchedulerPartition "")"
START="${START_OVERRIDE:-$(cfgget StartDate)}"
MONTHLY_DRY_RUN="$(cfgget MonthlyKalmanDryRun false)"
if [[ -z "$END_LIMIT" ]]; then
  END_LIMIT="$(cfgget FinalDate "$(cfgget EndDate "")")"
fi

MONTH_CONFIG="$(mktemp "${RUN_ROOT}/imi_obs_products.XXXXXX.yml")"   # shared storage, NOT node-local /tmp: this config is read by an sbatch'd stage on another node
trap 'rm -f "$MONTH_CONFIG"' EXIT

printf "[%s] Preparing monthly observation products from %s through %s\n" \
  "$(date '+%F %T')" "$START" "$END_LIMIT"

while [[ "$START" < "$END_LIMIT" ]]; do
  END="$(python "$MONTHLY_CONFIG" month-after "$START")"
  printf "\n[%s] Month %s -> %s\n" "$(date '+%F %T')" "$START" "$END"
  make_month_config "$START" "$END" "$MONTH_CONFIG"

  if [[ "${MONTHLY_DRY_RUN,,}" == "true" ]]; then
    printf "[dry-run] Would prepare monthly Jacobian/prior setup for %s -> %s\n" "$START" "$END"
    printf "[dry-run] Would run or reuse prior 0000 simulation for %s -> %s\n" "$START" "$END"
    printf "[dry-run] Would prepare inversion setup and submit obs_products stage\n"
    START="$END"
    continue
  fi

  "$PREPARE_MONTH" "$MONTH_CONFIG" "$RUN_ROOT" jacobian

  PRIOR_RUN_DIR="${RUN_ROOT}/jacobian_runs/${RUN_NAME}_0000"
  PRIOR_RUN_SCRIPT="${PRIOR_RUN_DIR}/${RUN_NAME}_0000.run"
  PRIOR_END_RESTART="${PRIOR_RUN_DIR}/Restarts/GEOSChem.Restart.${END}_0000z.nc4"
  if [[ -e "$PRIOR_END_RESTART" ]]; then
    printf "[%s] Reusing completed prior 0000 run; found %s\n" "$(date '+%F %T')" "$PRIOR_END_RESTART"
  else
    sbatch_args=(--parsable -c "$REQUESTED_CPUS" --mem "$REQUESTED_MEMORY" -t "$REQUESTED_TIME")
    if [[ -n "$SCHEDULER_PARTITION" ]]; then
      sbatch_args+=(-p "$SCHEDULER_PARTITION")
    fi
    PRIOR_JOB_ID=$(
      cd "$PRIOR_RUN_DIR"
      sbatch "${sbatch_args[@]}" "$PRIOR_RUN_SCRIPT"
    )
    printf "[%s] Submitted prior 0000 job %s\n" "$(date '+%F %T')" "$PRIOR_JOB_ID"
    submit_and_wait "$PRIOR_JOB_ID"
  fi

  "$PREPARE_MONTH" "$MONTH_CONFIG" "$RUN_ROOT" inversion
  OBS_JOB_ID=$(sbatch --parsable --export=ALL -c "$REQUESTED_CPUS" --mem "$REQUESTED_MEMORY" -t "$REQUESTED_TIME" ${SCHEDULER_PARTITION:+-p "$SCHEDULER_PARTITION"} "$RUN_STAGE" "$MONTH_CONFIG" obs_products)
  printf "[%s] Submitted observation-products job %s\n" "$(date '+%F %T')" "$OBS_JOB_ID"
  submit_and_wait "$OBS_JOB_ID"

  START="$END"
done

printf "\n[%s] Finished monthly observation products. Use the residual-error workflow to inspect and save So before running Jacobian inversions.\n" "$(date '+%F %T')"
