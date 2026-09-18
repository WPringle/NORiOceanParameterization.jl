#!/bin/bash
#####
##### Submit wrapper for run_LES_lake_east_GLEN_UVstress.pbs
#####
# Directional wind-stress variant (momentum flux split into Qᵁ/Qᵛ via wind_direction).
#
# PBS -N / -o / -e directives are parsed before the job runs, so they cannot see
# a -v CLOSURE=... variable.  This wrapper computes the job name and the job-level
# log paths from FORCING_SOURCE / CLOSURE and passes them to qsub on the command
# line (command-line -N/-o/-e override the directives in the .pbs file).
#
# Usage:
#   ./submit_LES_lake_east_GLEN_UVstress.sh [FORCING_SOURCE] [CLOSURE] [Cd]
#
#   FORCING_SOURCE : direct (default) | coare_wind
#   CLOSURE        : (empty, default = resolved LES) | SmagorinskyLilly | DynamicSmagorinsky | WENO5
#   Cd             : (empty, default = free-slip bottom) | quadratic bottom drag coefficient, e.g. 2e-3
#
# Examples:
#   ./submit_LES_lake_east_GLEN_UVstress.sh                                 # direct, no closure, free-slip
#   ./submit_LES_lake_east_GLEN_UVstress.sh coare_wind                      # coare, no closure
#   ./submit_LES_lake_east_GLEN_UVstress.sh direct DynamicSmagorinsky       # direct + dynamic Smagorinsky
#   ./submit_LES_lake_east_GLEN_UVstress.sh direct "" 2e-3                  # direct, no closure, Cd=2e-3
#   ./submit_LES_lake_east_GLEN_UVstress.sh coare_wind SmagorinskyLilly 2e-3 # coare + Smag-Lilly + Cd=2e-3

FORCING_SOURCE=${1:-direct}
CLOSURE=${2:-}
CD=${3:-}

# Tag used for the job name and job-level log files
# (e.g. "direct_UVstress", "coare_wind_DynamicSmagorinsky_Cd2e-3_UVstress").
TAG="${FORCING_SOURCE}${CLOSURE:+_${CLOSURE}}${CD:+_Cd${CD}}_UVstress"
JOBNAME="LES_GLEN_${TAG}"

# Submit from the project root so logs/ matches the existing runs' location
# (the PBS does `cd $PBS_O_WORKDIR` and writes logs/ relative to here; the
#  simulation OUTPUT_DIR inside the PBS is absolute, so it is unaffected).
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

qsub -N "${JOBNAME}" \
     -o "logs/${JOBNAME}.out" \
     -e "logs/${JOBNAME}.err" \
     -v "FORCING_SOURCE=${FORCING_SOURCE},CLOSURE=${CLOSURE},CD=${CD}" \
     training/run_LES_lake_east_GLEN_UVstress.pbs

echo "Submitted ${JOBNAME}  (FORCING_SOURCE=${FORCING_SOURCE}, CLOSURE=${CLOSURE:-none}, Cd=${CD:-none})"
echo "  job-level logs : logs/${JOBNAME}.out / .err"
echo "  per-year logs  : logs/LES_lake_east_GLEN_${TAG}_<year>.log"
