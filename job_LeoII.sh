#!/bin/bash
#PBS -N gs2_leoII
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /gpfs/home/jd925/GravSphere2/logs/


set -o pipefail   # without this the exit status is tee's, so a crashed run looks successful

# hx1 paths. On CX3 this tree lives at
# /rds/general/user/jd925/home/PhD_first_year/GravSphere2 instead, and conda is
# the /sw-eb module rather than the one in $HOME -- change both below together.
WORKDIR=/gpfs/home/jd925/GravSphere2

cd $WORKDIR
mkdir -p $WORKDIR/logs

# No keep/drop toggle here as in job_Seg1.sh: Leo II is a single 175-star sample
# from Spencer+17 with no ambiguous star to switch in and out, so there is
# nothing to choose at submission time. Everything else is set in
# initialise_LeoII.py.
#
# Nothing to uncomment in gravsphere2.py either: run_LeoII.py pins
# GS2_OBJECT=LeoII itself before importing it, and still aborts on a mismatch
# rather than writing another galaxy's chains into Output/LeoII/.

# PBS runs this as a non-interactive shell, which does not source ~/.bashrc, so
# the conda shell function does not exist and a bare 'conda activate' fails.
source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate gravsphere

# Keep BLAS single-threaded: the parallelism is over dynesty's pool, not linalg.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 'resume' picks up the checkpoint if one exists and starts fresh if it does not,
# so the same script is safe to submit first time and to resubmit after a
# walltime kill. Without it a resubmit silently restarts from zero and
# overwrites the checkpoint.
#
# Expect to use that here. The walltime matches job_Seg1.sh, but Leo II is a
# good deal more expensive: 175 tracers against Segue 1's 68, and the
# individual-star photometric term is a sum over all of them, plus ndims 19
# rather than 18 from the free stellar mass. Budget on the order of 2-3x the
# Segue 1 wall clock and plan to resubmit rather than raising walltime blindly.
#
# -u is needed because piping to tee makes Python block-buffer stdout, so the
# "live" log would otherwise only appear in ~8 kB bursts.
python -u run_LeoII.py 32 resume 2>&1 | tee $WORKDIR/logs/live_gs2_leoII.log
