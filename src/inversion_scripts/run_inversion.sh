#!/bin/bash

#SBATCH -o run_inversion_%j.out

##=======================================================================
## Parse config.yml file
##=======================================================================

send_error() {
    file=`basename "$0"`
    printf "\nInversion Error: on line number ${1} of ${file}: IMI exiting."
    echo "Error Status: 1" > .error_status_file.txt
    exit 1
}

# remove error status file if present
rm -f .error_status_file.txt

# trap and exit on errors
trap 'send_error $LINENO' ERR

printf "\n=== PARSING CONFIG FILE ===\n"

invPath={INVERSION_PATH}
configFile={CONFIG_FILE}
if [[ "$configFile" = /* ]]; then
    configPath="$configFile"
else
    configPath="${invPath}/${configFile}"
fi

# Get configuration
#  This defines $StartDate, $EndDate, $nBufferClusters, $RunName
#  It also define $PriorError, $ObsError, $Gamma, $PrecomputedJacobian
#  Parsing the config file here facilitates generation of inversion ensembles
#  All that needs to be done is to edit the config file for $PriorError,
#   $ObsError, and $Gamma
#  Make sure $PrecomputedJacobian is true, and then re-run this script
#   (or run_imi.sh with only the $DoInversion module switched on in config.yml).
PythonEnv=$({ grep '^PythonEnv:' ${configPath} || true; } |
    sed 's/PythonEnv://' |
    sed 's/#.*//' |
    sed 's/^[[:space:]]*//' |
    tr -d '"')
CondaFile=$(eval echo $({ grep '^CondaFile:' ${configPath} || true; } |
    sed 's/CondaFile://' |
    sed 's/#.*//' |
    sed 's/^[[:space:]]*//' |
    tr -d '"'))
CondaEnv=$({ grep '^CondaEnv:' ${configPath} || true; } |
    sed 's/CondaEnv://' |
    sed 's/#.*//' |
    sed 's/^[[:space:]]*//' |
    tr -d '"')
set +u
if [[ -n "$PythonEnv" ]]; then
    source ${invPath}/${PythonEnv}
else
    source "$CondaFile"
    conda activate "$CondaEnv"
fi
parsed_config=$(python ${invPath}/src/utilities/parse_yaml.py ${configPath}) || exit 1
eval "$parsed_config"

#=======================================================================
# Configuration (these settings generated on initial setup)
#=======================================================================
LonMinInvDomain={LON_MIN}
LonMaxInvDomain={LON_MAX}
LatMinInvDomain={LAT_MIN}
LatMaxInvDomain={LAT_MAX}
nElements={STATE_VECTOR_ELEMENTS}
nTracers={NUM_JACOBIAN_TRACERS}
OutputPath={OUTPUT_PATH}
Res={RES}
period_i={PERIOD}
StateVectorFile={STATE_VECTOR_PATH}

EmissionElements=$(python - "$StateVectorFile" <<'PY'
import sys
import numpy as np
import xarray as xr
ds = xr.load_dataset(sys.argv[1])
vals = np.unique(ds["StateVector"].values)
vals = vals[vals > 0]
print(len(vals))
PY
)
nElements=$EmissionElements
if "$OptimizeBCs"; then
    nElements=$((nElements + 4))
fi
if "$OptimizeOH"; then
    if "$isRegional"; then
        nElements=$((nElements + 1))
    else
        nElements=$((nElements + 2))
    fi
fi

KalmanInversionSubdir="${KalmanInversionSubdir:-kf_inversions}"   # separate periods folder (e.g. improved-error run)
if "$KalmanMode"; then
    InvDir="${OutputPath}/${RunName}/${KalmanInversionSubdir}/period${period_i}"
else
    InvDir="${OutputPath}/${RunName}/inversion"
fi

JacobianRunsDir="${OutputPath}/${RunName}/jacobian_runs"
PriorRunDir="${JacobianRunsDir}/${RunName}_0000"
PriorEmisDir="${OutputPath}/${RunName}/hemco_prior_emis/OutputDir"
BackgroundRunDir="${JacobianRunsDir}/${RunName}_background"
PosteriorRunDir="${OutputPath}/${RunName}/posterior_run"
GCDir="${InvDir}/data_geoschem"
GCVizDir="${InvDir}/data_geoschem_prior"
JacobianDir="${InvDir}/data_converted"
satelliteCache="${OutputPath}/${RunName}/satellite_data"

# For Kalman filter: assume first inversion period (( period_i = 1 )) by default
# Switch is flipped to false automatically if (( period_i > 1 ))
FirstSimSwitch=$1

printf "\n=== EXECUTING RUN_INVERSION.SH ===\n"
    
#=======================================================================
# Error checks
#=======================================================================

# Make sure specified paths exist
if [[ ! -d ${JacobianRunsDir} ]]; then
    printf "${JacobianRunsDir} does not exist. Please fix JacobianRunsDir in run_inversion.sh.\n"
    exit 1
fi
if [[ ! -f ${StateVectorFile} ]]; then
    printf "${StateVectorFile} does not exist. Please fix StateVectorFile in run_inversion.sh.\n"
    exit 1
fi

#=======================================================================
# Setup GC data directory in workdir
#=======================================================================

printf "Calling setup_gc_cache.py\n"
ObsOnlyCache="${ObservationOnlyJacobianCache:-false}"
export PYTHONPATH="${PYTHONPATH:-}:${OutputPath}:${invPath}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-imi}-${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR"
if "$LognormalErrors"; then
    # for lognormal errors we use the clean background run
    GCsourcepth="${BackgroundRunDir}/OutputDir"
    PriorOutputDir="${PriorRunDir}/OutputDir"
    # also need the prior cache so that we can visualize the prior simulation
    python ${InvDir}/setup_gc_cache.py $StartDate $EndDate $PriorOutputDir $GCVizDir $satelliteCache $SatelliteProduct $UseWaterObs $LonMinInvDomain $LonMaxInvDomain $LatMinInvDomain $LatMaxInvDomain $Species $ObsOnlyCache $UseGCHP; wait
else
    # for normal errors we use the prior run
    GCsourcepth="${PriorRunDir}/OutputDir"
fi
python ${InvDir}/setup_gc_cache.py $StartDate $EndDate $GCsourcepth $GCDir $satelliteCache $SatelliteProduct $UseWaterObs $LonMinInvDomain $LonMaxInvDomain $LatMinInvDomain $LatMaxInvDomain $Species $ObsOnlyCache $UseGCHP; wait
printf "DONE -- setup_gc_cache.py\n\n"

#=======================================================================
# setup geoschem cache for pseudo observations if doing OSSE
#=======================================================================
if "$EnableOSSE"; then
    RunDirOSSE="${OutputPath}/${RunName}/osse_observations_run"
    GCDirOSSE="./data_geoschem_osse"
    mkdir -p  $GCDirOSSE
    # If simulating observations, we need to postprocess the observation data
    python postproc_diags.py $RunName $RunDirOSSE $PrevDir $StartDate $Res; wait
    python setup_gc_cache.py $StartDate $EndDate "${RunDirOSSE}/OutputDir" $GCDirOSSE $satelliteCache $SatelliteProduct $UseWaterObs $LonMinInvDomain $LonMaxInvDomain $LatMinInvDomain $LatMaxInvDomain $Species $ObsOnlyCache $UseGCHP; wait
fi

#=======================================================================
# Optionally build prior error covariance matrix with off-diagonal 
# elements based on specified length scale
#=======================================================================
if "$OffDiagonalPriorCov"; then
    PriorCovarianceMethod="${PriorCovarianceMethod:-length_scale}"
    if [[ "$PriorCovarianceMethod" == "national_inventory" ]]; then
        python build_national_inventory_prior_covariance.py $StateVectorFile $PriorEmisDir $configPath $StartDate $EndDate $nBufferClusters; wait
    elif [[ "$PriorCovarianceMethod" == "sector_ensemble" ]]; then
        python build_sector_ensemble_prior_covariance.py $StateVectorFile $PriorEmisDir $configPath $StartDate $EndDate $nBufferClusters; wait
    else
        python build_full_prior_covariance.py $StateVectorFile $PriorEmisDir $LengthScalePriorCov $StartDate $EndDate $nBufferClusters; wait
    fi
    # Sequential-KF RTPS inflation: for period>1, relax the previous period's posterior covariance
    # back toward this static Sa0 (Whitaker & Hamill 2012; Pendergrass et al. 2025 Eq. 8), replacing
    # the static prior_norm_error_covariance.npz. Off unless KalmanCovarianceInflation: true.
    if [ "${KalmanCovarianceInflation:-false}" = "true" ] && "$KalmanMode" && [ "$period_i" -gt 1 ]; then
        prevResult="${OutputPath}/${RunName}/${KalmanInversionSubdir}/period$((period_i-1))/inversion_result.nc"
        if [ -f "$prevResult" ]; then
            cp prior_norm_error_covariance.npz prior_norm_error_covariance_static.npz
            python ${invPath}/src/components/kalman_component/make_inflated_prior_covariance.py \
                --baseline-cov prior_norm_error_covariance_static.npz \
                --previous-inversion-result "$prevResult" \
                --prior-err "$(echo $PriorError | tr -d '[] ')" \
                --alpha "${KalmanRTPSAlpha:-0.7}" \
                --output prior_norm_error_covariance.npz; wait
        else
            echo "WARNING: KalmanCovarianceInflation on but previous result missing: $prevResult; using static Sa0"
        fi
    fi
fi

#=======================================================================
# Generate Jacobian matrix files 
#=======================================================================

isPost="False"
if ! "$PrecomputedJacobian"; then
    buildJacobian="True"
    jacobian_sf="None"
else
    buildJacobian="False"
    jacobian_sf=${InvDir}/jacobian_scale_factors.npy
fi
ObsProductsOnly="${IMI_OBS_PRODUCTS_ONLY:-false}"
AllowMissingK="False"
if [[ "$ObsProductsOnly" == "true" ]]; then
    buildJacobian="False"
    AllowMissingK="True"
fi
MergedDir="${OutputPath}/${RunName}/inversion_data"
MergedK="${MergedDir}/K/K_${StartDate}_${EndDate}.npz"
MergedY="${MergedDir}/y/y_${StartDate}_${EndDate}.npz"
MergedPrior="${MergedDir}/xch4_0/xch4_0_${StartDate}_${EndDate}.npz"
MergedSo="${MergedDir}/so/so_${StartDate}_${EndDate}.npz"
MergedObs="${MergedDir}/observations/observations_${StartDate}_${EndDate}.npz"
ReuseMerged=false
if ! "$LognormalErrors" && [[ -f "$MergedK" ]] && [[ -f "$MergedY" ]] && [[ -f "$MergedPrior" ]] && [[ -f "$MergedSo" ]]; then
    ReuseMerged=true
fi
if [[ "$ObsProductsOnly" == "true" ]] && [[ -f "$MergedObs" ]] && [[ -f "$MergedY" ]] && [[ -f "$MergedPrior" ]] && [[ -f "$MergedSo" ]]; then
    ReuseMerged=true
fi

if "$ReuseMerged"; then
    printf "Reusing existing merged inversion products for %s -> %s\n\n" "$StartDate" "$EndDate"
else
    printf "Calling jacobian.py\n"
    if [[ "$ObsProductsOnly" == "true" ]]; then
        find "${InvDir}/data_converted" -type f -name '*.pkl' -delete 2>/dev/null || true
        find "${InvDir}/data_visualization" -type f -name '*.pkl' -delete 2>/dev/null || true
    fi
    python -u ${InvDir}/jacobian.py ${InvDir} ${configPath} $StartDate $EndDate $LonMinInvDomain $LonMaxInvDomain $LatMinInvDomain $LatMaxInvDomain $nElements $Species $satelliteCache $SatelliteProduct $UseWaterObs $isPost $period_i $buildJacobian False; wait
    if "$LognormalErrors"; then
        # for lognormal error visualization of the prior we sample the prior run
        # without constructing the jacobian matrix
        python ${InvDir}/jacobian.py ${InvDir} ${configPath} $StartDate $EndDate $LonMinInvDomain $LonMaxInvDomain $LatMinInvDomain $LatMaxInvDomain $nElements $Species $satelliteCache $SatelliteProduct $UseWaterObs $isPost $period_i False True; wait
    fi
    printf " DONE -- jacobian.py\n\n"

    printf "Calling merge_partial_k.py\n"
    python ${InvDir}/merge_partial_k.py $JacobianDir $StateVectorFile ${OutputPath}/${RunName}/config_${RunName}.yml $PrecomputedJacobian $AllowMissingK
    printf "DONE -- merge_partial_k.py\n\n"

    if ! "$PrecomputedJacobian"; then
        printf "Cleaning transient Jacobian observation-space files\n"
        find "${InvDir}/data_converted" -type f -name '*.pkl' -delete 2>/dev/null || true
        find "${InvDir}/data_visualization" -type f -name '*.pkl' -delete 2>/dev/null || true
        printf "DONE -- cleanup transient Jacobian files\n\n"
    fi
fi

#=======================================================================
# Optionally build residual-error observational error covariance (So)
# with off-diagonal spatial correlations derived from residual anomalies
#=======================================================================
OffDiagonalObsCov="${OffDiagonalObsCov:-false}"
if "$OffDiagonalObsCov"; then
    printf "Calling build_residual_obs_covariance.py\n"
    python ${InvDir}/build_residual_obs_covariance.py \
        --start $StartDate \
        --end $EndDate \
        --run-root "${OutputPath}/${RunName}" \
        --state-vector $StateVectorFile \
        --obs-error-name "${ObsError:-15.0}"; wait
    printf "DONE -- build_residual_obs_covariance.py\n\n"
fi

if [[ "$ObsProductsOnly" == "true" ]]; then
    printf "IMI_OBS_PRODUCTS_ONLY=true; stopping after merged observation products.\n"
    exit 0
fi

if [[ "${IMI_PREPARE_INVERSION_ONLY:-false}" == "true" ]]; then
    printf "IMI_PREPARE_INVERSION_ONLY=true; stopping after merged inversion products.\n"
    exit 0
fi

#=======================================================================
# Do inversion
#=======================================================================
if "$LognormalErrors"; then
    # then we run the inversion
    printf "Calling lognormal_invert.py\n"
    python ${InvDir}/lognormal_invert.py ${configPath} $StateVectorFile $jacobian_sf
    printf "DONE -- lognormal_invert.py\n\n"
else
    posteriorSF="./inversion_result.nc"
    python_args=(${InvDir}/invert.py ${OutputPath}/${RunName}/config_${RunName}.yml $nElements $JacobianDir $posteriorSF $LonMinInvDomain $LonMaxInvDomain $LatMinInvDomain $LatMaxInvDomain $Res $jacobian_sf $StateVectorFile)

    printf "Calling invert.py\n"
    python "${python_args[@]}"; wait
    printf "DONE -- invert.py\n\n"
    #=======================================================================
    # Create gridded posterior scaling factor netcdf file
    #=======================================================================
    GriddedPosterior="./gridded_posterior.nc"

    printf "Calling make_gridded_posterior.py\n"
    python ${InvDir}/make_gridded_posterior.py $posteriorSF $StateVectorFile $GriddedPosterior; wait
    printf "DONE -- make_gridded_posterior.py\n\n"
fi

printf "Exiting run_inversion.sh"

exit 0
