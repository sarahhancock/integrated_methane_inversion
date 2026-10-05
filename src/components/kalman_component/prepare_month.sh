#!/bin/bash

set -euo pipefail

IMI_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MONTHLY_DIR="${IMI_SOURCE_DIR}/src/components/kalman_component"
MONTHLY_CONFIG="${MONTHLY_DIR}/monthly_config.py"
CONFIG_FILE="${1:?config path required}"
CONFIG_FILE="$(readlink -f "$CONFIG_FILE")"
RUN_ROOT_OVERRIDE="${2:-}"
SETUP_STAGE="${3:-jacobian}"

cfgget() {
  if [[ $# -gt 1 ]]; then
    python "$MONTHLY_CONFIG" get "$CONFIG_FILE" "$1" --default "$2"
  else
    python "$MONTHLY_CONFIG" get "$CONFIG_FILE" "$1"
  fi
}

RUN_NAME="$(cfgget RunName)"
OUTPUT_PATH="$(cfgget OutputPath)"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${OUTPUT_PATH}/${RUN_NAME}}"
CONDA_ENV="$(cfgget CondaEnv "")"
CONDA_FILE="$(cfgget CondaFile "")"
GEOSCHEM_ENV_REL="$(cfgget GEOSChemEnv "")"
START_DATE="$(cfgget StartDate)"
END_DATE="$(cfgget EndDate "")"
REQUESTED_CPUS="$(cfgget RequestedCPUs 1)"
REQUESTED_MEMORY="$(cfgget RequestedMemory 8000)"
REQUESTED_TIME="$(cfgget RequestedTime "0-12:00")"
SCHEDULER_PARTITION="$(cfgget SchedulerPartition "")"
OPTIMIZE_SOIL="$(cfgget OptimizeSoil false)"
RESTART_FILE_PREFIX="$(cfgget RestartFilePrefix "")"
BC_PATH="$(cfgget BCpath "")"
BC_VERSION="$(cfgget BCversion "")"
SPECIES="$(cfgget Species CH4)"

if [[ -z "$END_DATE" ]]; then
  END_DATE="$(python "$MONTHLY_CONFIG" month-after "$START_DATE")"
fi

PREP_JACOBIANS=true
PREP_PRIOR_HEMCO=true
if [[ "$SETUP_STAGE" == "inversion" ]]; then
  PREP_JACOBIANS=false
  PREP_PRIOR_HEMCO=false
  if [[ "${OPTIMIZE_SOIL,,}" == "true" ]]; then
    PREP_PRIOR_HEMCO=true
  fi
elif [[ "$SETUP_STAGE" != "jacobian" ]]; then
  printf "Unknown monthly setup stage: %s\n" "$SETUP_STAGE" >&2
  exit 1
fi

JAC_DIR="${RUN_ROOT}/jacobian_runs"
ZERO_RUN="${JAC_DIR}/${RUN_NAME}_0000"
HEMCO_DIR="${RUN_ROOT}/hemco_prior_emis"
HEMCO_OUTPUT_DIR="${HEMCO_DIR}/OutputDir"
PRIOR_HEMCO_DIR="${RUN_ROOT}/prior_hemco_runs_output"
RUN_INV_SCRIPT="${RUN_ROOT}/inversion/run_inversion.sh"
INV_DIR="${RUN_ROOT}/inversion"
INV_OPERATORS_DIR="${INV_DIR}/operators"
TARGET_HEMCO_FILE="${HEMCO_OUTPUT_DIR}/HEMCO_sa_diagnostics.${START_DATE}0000.nc"
ARCHIVED_HEMCO_FILE="${PRIOR_HEMCO_DIR}/HEMCO_sa_diagnostics.${START_DATE}0000.nc"
OUTPUT_CONFIG="${RUN_ROOT}/config_${RUN_NAME}.yml"
HEMCO_RUN_SCRIPT="${HEMCO_DIR}/${RUN_NAME}_HEMCO_Prior_Emis.run"
GEOSCHEM_ENV_ABS="${IMI_SOURCE_DIR}/${GEOSCHEM_ENV_REL}"

printf "[%s] monthly prepare for %s -> %s (%s)\n" "$(date '+%F %T')" "$START_DATE" "$END_DATE" "$SETUP_STAGE"
python "$MONTHLY_CONFIG" write-temp "$CONFIG_FILE" "$OUTPUT_CONFIG" \
  --set StartDate="$START_DATE" \
  --set EndDate="$END_DATE" >/dev/null

if [[ -n "$CONDA_FILE" ]]; then
  set +u
  source "$CONDA_FILE"
  [[ -n "$CONDA_ENV" ]] && conda activate "$CONDA_ENV"
  set -u
fi
export PYTHONPATH="${IMI_SOURCE_DIR}:${PYTHONPATH:-}"

if "$PREP_JACOBIANS"; then
  python "${MONTHLY_DIR}/optimize_history_rc.py" "${RUN_ROOT}/template_run" "$CONFIG_FILE"
  printf "[%s] Clearing perturbation outputs while preserving %s\n" "$(date '+%F %T')" "$ZERO_RUN"
  find "$JAC_DIR" \
    -path "*/OutputDir/*" \
    ! -path "${ZERO_RUN}/OutputDir/*" \
    -type f -delete
fi

sed -i "s|^invPath=.*|invPath=${IMI_SOURCE_DIR}|" "$RUN_INV_SCRIPT"

printf "[%s] Syncing inversion scripts from IMI source\n" "$(date '+%F %T')"
mkdir -p "$INV_OPERATORS_DIR"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/setup_gc_cache.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/setup_jacobian_obs_cache.py" "$INV_DIR/" 2>/dev/null || true
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/jacobian.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/merge_partial_k.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/invert.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/softplus_invert.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/lognormal_invert.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/build_length_scale_prior_covariance.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/build_national_inventory_prior_covariance.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/build_sector_ensemble_prior_covariance.py" "$INV_DIR/"
cp "${IMI_SOURCE_DIR}/src/inversion_scripts/operators/"*.py "$INV_OPERATORS_DIR/"
cp "${IMI_SOURCE_DIR}/src/utilities/config_utils.py" "$INV_DIR/"

inject_env() {
  local script="$1"
  [[ -f "$script" && -n "$GEOSCHEM_ENV_REL" ]] || return 0
  if ! grep -q "$GEOSCHEM_ENV_ABS" "$script"; then
    sed -i "/^#SBATCH --mail-type=END/a set +u" "$script"
    sed -i "/^##SBATCH --mail-type=END/a set +u" "$script"
    sed -i "/set +u/a source ${GEOSCHEM_ENV_ABS}" "$script"
    sed -i "/source ${GEOSCHEM_ENV_ABS//\//\\/}/a set -u" "$script"
  fi
}

process_prior_excluding_soil_sink() {
  local file="$1"
  python - "$file" <<'PY'
import sys
import xarray as xr

path = sys.argv[1]
with xr.open_dataset(path) as opened:
    ds = opened.load()
if "EmisCH4_Total" in ds and "EmisCH4_SoilAbsorb" in ds:
    ds["EmisCH4_Total_ExclSoilAbs"] = ds["EmisCH4_Total"] - ds["EmisCH4_SoilAbsorb"]
    ds["EmisCH4_Total_ExclSoilAbs"].attrs = ds["EmisCH4_Total"].attrs
    ds.to_netcdf(path)
PY
}

if "$PREP_JACOBIANS"; then
  FORMATTED_START="[${START_DATE}, 000000]"
  FORMATTED_END="[${END_DATE}, 000000]"
  for dir in "$JAC_DIR"/"${RUN_NAME}"_*; do
    CONFIG_PATH="${dir}/geoschem_config.yml"
    [[ -f "$CONFIG_PATH" ]] || continue
    inject_env "${dir}/$(basename "$dir").run"
    sed -i "s/^  start_date: .*/  start_date: ${FORMATTED_START}/" "$CONFIG_PATH"
    sed -i "s/^  end_date: .*/  end_date: ${FORMATTED_END}/" "$CONFIG_PATH"
    python "${MONTHLY_DIR}/optimize_history_rc.py" "$dir" "$CONFIG_FILE"
  done
fi

if "$PREP_PRIOR_HEMCO"; then
  sed -i \
    -e "s|^START: .*|START: ${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2} 00:00:00|" \
    -e "s|^END: .*|END:   ${END_DATE:0:4}-${END_DATE:4:2}-${END_DATE:6:2} 00:00:00|" \
    "${HEMCO_DIR}/HEMCO_sa_Time.rc"

  inject_env "$HEMCO_RUN_SCRIPT"
  mkdir -p "$HEMCO_OUTPUT_DIR" "$PRIOR_HEMCO_DIR"
  if [[ ! -f "$TARGET_HEMCO_FILE" && -f "$ARCHIVED_HEMCO_FILE" ]]; then
    printf "[%s] Restoring archived HEMCO prior %s\n" "$(date '+%F %T')" "$ARCHIVED_HEMCO_FILE"
    cp -p "$ARCHIVED_HEMCO_FILE" "$TARGET_HEMCO_FILE"
  fi
  if [[ ! -f "$TARGET_HEMCO_FILE" ]]; then
    printf "[%s] Submitting HEMCO prior emissions job\n" "$(date '+%F %T')"
    sbatch_args=(-W -c "$REQUESTED_CPUS" --mem "$REQUESTED_MEMORY" -t "$REQUESTED_TIME")
    if [[ -n "$SCHEDULER_PARTITION" ]]; then
      sbatch_args+=(-p "$SCHEDULER_PARTITION")
    fi
    (
      cd "$HEMCO_DIR"
      sbatch "${sbatch_args[@]}" "${RUN_NAME}_HEMCO_Prior_Emis.run"
    )
    [[ -f "$TARGET_HEMCO_FILE" ]] || { printf "Expected HEMCO prior was not created: %s\n" "$TARGET_HEMCO_FILE" >&2; exit 1; }
    if [[ "${OPTIMIZE_SOIL,,}" == "true" ]]; then
      process_prior_excluding_soil_sink "$TARGET_HEMCO_FILE"
    fi
  else
    printf "[%s] Reusing HEMCO prior %s\n" "$(date '+%F %T')" "$TARGET_HEMCO_FILE"
  fi
  cp -p "$TARGET_HEMCO_FILE" "$ARCHIVED_HEMCO_FILE"
fi

if "$PREP_JACOBIANS"; then
  if [[ -z "$RESTART_FILE_PREFIX" && -n "$BC_PATH" && -n "$BC_VERSION" ]]; then
    RESTART_FILE_PREFIX="${BC_PATH}/${BC_VERSION}/GEOSChem.BoundaryConditions."
  fi
  BC_TEMPLATE_PATH="${RESTART_FILE_PREFIX}\$YYYY\$MM\$DD_0000z.nc4"
  BC_FOR_START="${RESTART_FILE_PREFIX}${START_DATE}_0000z.nc4"
  RESTART_NAME="GEOSChem.Restart.${START_DATE}_0000z.nc4"
  SPINUP_RESTART="${RUN_ROOT}/spinup_run/Restarts/${RESTART_NAME}"
  ABS_1PPB_RESTART="${RUN_ROOT}/jacobian_1ppb_ics_bcs/Restarts/GEOSChem.Restart.1ppb.${START_DATE}_0000z.nc4"
  ABS_1PPB_BC="${RUN_ROOT}/jacobian_1ppb_ics_bcs/BCs/GEOSChem.BoundaryConditions.1ppb.${START_DATE}_0000z.nc4"

  if [[ -n "$RESTART_FILE_PREFIX" && -f "$BC_FOR_START" ]]; then
    python "${IMI_SOURCE_DIR}/src/components/jacobian_component/make_jacobian_icbc.py" \
      "$CONFIG_FILE" "$BC_FOR_START" "${RUN_ROOT}/jacobian_1ppb_ics_bcs/BCs" "$START_DATE" "$SPECIES"
  fi

  for dir in "$JAC_DIR"/"${RUN_NAME}"_*; do
    RESTART_DIR="${dir}/Restarts"
    HEMCO_FILE="${dir}/HEMCO_Config.rc"
    [[ -d "$RESTART_DIR" ]] || continue
    if [[ "$dir" == "$ZERO_RUN" ]]; then
      find "$RESTART_DIR" -maxdepth 1 -name 'GEOSChem.Restart.*_0000z.nc4' -xtype l -delete
      ZERO_RESTART="${RESTART_DIR}/${RESTART_NAME}"
      if [[ -e "$ZERO_RESTART" ]]; then
        printf "[%s] Preserving 0000 restart %s\n" "$(date '+%F %T')" "$ZERO_RESTART"
      elif [[ -e "$SPINUP_RESTART" ]]; then
        ln -sfn "$SPINUP_RESTART" "$ZERO_RESTART"
      else
        printf "Missing 0000 restart %s and spinup fallback %s\n" "$ZERO_RESTART" "$SPINUP_RESTART" >&2
        exit 1
      fi
      sed -i "/(((GC_RESTART/,/)))GC_RESTART/ s|SpeciesBC_CH4|SpeciesRst_CH4|" "$HEMCO_FILE"
      [[ -n "$BC_TEMPLATE_PATH" ]] && sed -i "/(((GC_BCs/,/)))GC_BCs/ s|.*SpeciesBC_CH4.*|* BC_CH4 $BC_TEMPLATE_PATH SpeciesBC_CH4 1900-2100/1-12/1-31/* EFY xyz 1 CH4 - 1 1|" "$HEMCO_FILE"
    else
      find "$RESTART_DIR" -maxdepth 1 -name 'GEOSChem.Restart.*_0000z.nc4' -type l -delete
      [[ -e "$ABS_1PPB_RESTART" ]] && ln -sfn "$ABS_1PPB_RESTART" "${RESTART_DIR}/${RESTART_NAME}"
      sed -i "/(((GC_RESTART/,/)))GC_RESTART/ s|SpeciesRst_CH4|SpeciesBC_CH4|" "$HEMCO_FILE"
      [[ -e "$ABS_1PPB_BC" ]] && sed -i "/(((GC_BCs/,/)))GC_BCs/ s|.*SpeciesBC_CH4.*|* BC_CH4 $ABS_1PPB_BC SpeciesBC_CH4 1980-2021/1-12/1-31/* C xyz 1 CH4 - 1 1|" "$HEMCO_FILE"
    fi
  done
fi

for subdir in data_converted data_geoschem data_sensitivities data_visualization; do
  find "${RUN_ROOT}/inversion/${subdir}" -type f -delete 2>/dev/null || true
done

printf "[%s] Finished monthly prepare for %s -> %s\n" "$(date '+%F %T')" "$START_DATE" "$END_DATE"
