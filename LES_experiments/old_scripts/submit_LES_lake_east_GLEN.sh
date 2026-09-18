#!/bin/bash
#####
##### Submit wrapper for run_LES_lake_east_GLEN.pbs
#####
# PBS -N / -o / -e directives are parsed before the job runs, so they cannot see
# a -v CLOSURE=... variable.  This wrapper computes the job name and the job-level
# log paths from FORCING_SOURCE / CLOSURE and passes them to qsub on the command
# line (command-line -N/-o/-e override the directives in the .pbs file).
#
# Usage:
#   ./submit_LES_lake_east_GLEN.sh [FORCING_SOURCE] [CLOSURE]
#
#   FORCING_SOURCE : direct (default) | coare_wind
#   CLOSURE        : (empty, default = resolved LES) | SmagorinskyLilly | DynamicSmagorinsky
#
# Examples:
#   ./submit_LES_lake_east_GLEN.sh                              # direct, no closure
#   ./submit_LES_lake_east_GLEN.sh coare_wind                   # coare, no closure
#   ./submit_LES_lake_east_GLEN.sh direct DynamicSmagorinsky    # direct + dynamic Smagorinsky
#   ./submit_LES_lake_east_GLEN.sh coare_wind SmagorinskyLilly  # coare + Smagorinsky-Lilly
#   ./submit_LES_lake_east_GLEN.sh coare_wind DynamicSmagorinsky  # coare + dynamic Smag

FORCING_SOURCE=${1:-direct}
CLOSURE=${2:-}

# Tag used for the job name and job-level log files (e.g. "direct", "coare_wind_DynamicSmagorinsky").
TAG="${FORCING_SOURCE}${CLOSURE:+_${CLOSURE}}"
JOBNAME="LES_GLEN_${TAG}"

# Submit from the project root so logs/ matches the existing runs' location
# (the PBS does `cd $PBS_O_WORKDIR` and writes logs/ relative to here; the
#  simulation OUTPUT_DIR inside the PBS is absolute, so it is unaffected).
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

qsub -N "${JOBNAME}" \
     -o "logs/${JOBNAME}.out" \
     -e "logs/${JOBNAME}.err" \
     -v "FORCING_SOURCE=${FORCING_SOURCE},CLOSURE=${CLOSURE}" \
     training/run_LES_lake_east_GLEN.pbs

echo "Submitted ${JOBNAME}  (FORCING_SOURCE=${FORCING_SOURCE}, CLOSURE=${CLOSURE:-none})"
echo "  job-level logs : logs/${JOBNAME}.out / .err"
echo "  per-year logs  : logs/LES_lake_east_GLEN_${TAG}_<year>.log"
