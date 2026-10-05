#!/bin/bash

set -euo pipefail

IMI_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KALMAN_DIR="${IMI_SOURCE_DIR}/src/components/kalman_component"
BASE_CONFIG="${1:-${IMI_SOURCE_DIR}/config.yml}"
OUTDIR="${2:-${KALMAN_DIR}/test_setups/generated}"

cd "$IMI_SOURCE_DIR"

printf "Generating Kalman test configs from %s\n" "$BASE_CONFIG"
python "${KALMAN_DIR}/make_resumable_kalman_test_configs.py" "$BASE_CONFIG" --outdir "$OUTDIR"

printf "\nChecking shell syntax\n"
bash -n run_imi.sh "${KALMAN_DIR}"/*.sh

printf "\nChecking Python syntax\n"
python -m py_compile "${KALMAN_DIR}"/*.py src/utilities/sanitize_input_yaml.py

printf "\nValidating generated configs\n"
for config in "${OUTDIR}"/*.yml; do
  [[ "$(basename "$config")" == "manifest.yml" ]] && continue
  printf "  %s\n" "$config"
  python src/utilities/sanitize_input_yaml.py "$config"
  python src/utilities/parse_yaml.py "$config" >/dev/null
  python "${KALMAN_DIR}/monthly_config.py" get "$config" RunName >/dev/null
  python "${KALMAN_DIR}/monthly_config.py" count-elements "$config" "$(python "${KALMAN_DIR}/monthly_config.py" get "$config" StateVectorFile)" >/dev/null 2>&1 || true
done

printf "\nKalman test setup validation complete.\n"
