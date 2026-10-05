#!/bin/bash

set -euo pipefail

# sbatch runs this from the scheduler spool dir, so BASH_SOURCE resolves to /var/slurmd -> prefer the
# IMI_SOURCE_DIR exported by the parent driver (run_monthly_jacobians_kalman.sh), fall back to BASH_SOURCE.
IMI_SOURCE_DIR="${IMI_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONFIG_FILE="${1:?config path required}"
CONFIG_FILE="$(readlink -f "$CONFIG_FILE")"
STAGE="${2:-full}"
MONTHLY_CONFIG="${IMI_SOURCE_DIR}/src/components/kalman_component/monthly_config.py"

TMP_CONFIG="$(mktemp /tmp/imi_monthly_config.XXXXXX.yml)"
trap 'rm -f "$TMP_CONFIG"' EXIT

common_flags=(
  --set KalmanMode=false
  --set ResumableMonthlyKalman=false
  --set RunSetup=false
  --set SetupTemplateRundir=false
  --set SetupSpinupRun=false
  --set SetupJacobianRuns=false
  --set SetupInversion=false
  --set SetupPosteriorRun=false
  --set DoHemcoPriorEmis=false
  --set DoSpinup=false
  --set ReDoJacobian=false
  --set DoPosterior=false
)

case "$STAGE" in
  full)
    cp "$CONFIG_FILE" "$TMP_CONFIG"
    ;;
  jacobian)
    python "$MONTHLY_CONFIG" write-temp "$CONFIG_FILE" "$TMP_CONFIG" \
      "${common_flags[@]}" \
      --set DoJacobian=true \
      --set DoInversion=false
    ;;
  inversion)
    python "$MONTHLY_CONFIG" write-temp "$CONFIG_FILE" "$TMP_CONFIG" \
      "${common_flags[@]}" \
      --set DoJacobian=false \
      --set DoInversion=true
    ;;
  obs_products)
    python "$MONTHLY_CONFIG" write-temp "$CONFIG_FILE" "$TMP_CONFIG" \
      "${common_flags[@]}" \
      --set DoJacobian=false \
      --set DoInversion=true
    export IMI_OBS_PRODUCTS_ONLY=true
    ;;
  *)
    printf "Unknown monthly stage: %s\n" "$STAGE" >&2
    exit 1
    ;;
esac

cd "$IMI_SOURCE_DIR"
printf "Running monthly IMI stage: %s\n" "$STAGE"
bash run_imi.sh "$TMP_CONFIG"
