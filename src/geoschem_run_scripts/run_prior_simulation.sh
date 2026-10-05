#!/bin/bash
#SBATCH -J {RunName}

### Run directory
RUNDIR=$(pwd -P)

### Get current task ID
xstr="0000"

# This checks for the presence of the error status file. If present, this indicates 
# a previous prior sim exited with an error, so this prior will not run
FILE=.error_status_file.txt
if test -f "$FILE"; then
    echo "$FILE exists. Exiting."
    echo "prior simulation: ${xstr} exited without running."
    exit 1
fi

### Run GEOS-Chem in the directory corresponding to the cluster Id
cd  ${RUNDIR}/{RunName}_${xstr}
# DEBUG/ITERATION reuse: if ${RUNDIR}/.reuse_prior_sim exists and this period's prior-sim
# output is already present, skip the GEOS-Chem run and reuse it. This lets downstream
# (inversion/posterior) fixes be tested without re-simulating an unchanged prior. Absent the
# flag file (the real KF), behaviour is unchanged.
if [[ -f "${RUNDIR}/.reuse_prior_sim" ]] && ls OutputDir/GEOSChem.SpeciesConc.*.nc4 >/dev/null 2>&1; then
    echo "Reusing existing prior-sim output (${RUNDIR}/.reuse_prior_sim set); skipping GEOS-Chem."
    cd "${RUNDIR}"; echo "finished prior simulation: ${xstr} (reused existing output)"; exit 0
fi
if {UseGCHP}; then
    ./cleanRunDir.sh
    echo "{StartDate} 000000" > cap_restart
    sed -i -e "s/Run_Duration=\"[0-9]\{8\} 000000\"/Run_Duration=\"{RunDuration} 000000\"/" \
        setCommonRunSettings.sh
fi
./{RunName}_${xstr}.run

# save the exit code of the prior simulation cmd
retVal=$?

# Check whether the prior sim finished successfully. If not, write to a hidden file. 
# The presence of the .error_status_file.txt indicates whether an error ocurred. 
# This is needed because scripts that set off sbatch jobs have no knowledge of 
# whether the job finished successfully.
if [ $retVal -ne 0 ]; then
    rm -f .error_status_file.txt
    echo "Error Status: $retVal" > ../.error_status_file.txt
    echo "prior simulation: ${xstr} exited with error code: $retVal"
    echo "Check the log file in the ${RUNDIR}/{RunName}_${xstr} directory for more details."
    exit $retVal
fi

echo "finished prior simulation: ${xstr}"

exit 0
