#!/bin/bash
#PBS -N gs2_segue1
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /gpfs/home/jd925/GravSphere2/logs/


set -o pipefail   # without this the exit status is tee's, so a crashed run looks successful

# hx1 paths. On CX3 this tree lives at
# /rds/general/user/jd925/home/PhD_first_year/GravSphere2 instead, and conda is
# the /sw-eb module rather than the one in $HOME -- change both below together.
WORKDIR=/gpfs/home/jd925/GravSphere2

cd $WORKDIR
mkdir -p $WORKDIR/logs

# ---------------------------------------------------------------------------
# Which Segue 1 sample to run. Edit this one line, then submit.
#
#   keep=true    68 stars, the ambiguous J100704.35+160459.4 kept
#   keep=false   67 stars, that star removed
#
# The two runs ARE the systematic (see the notes at the top of
# initialise_Seg1.py), so expect to run both. Each writes to its own
# Output/Segue1_{keep,drop}/ tree, so they never overwrite each other.
#
# Bash, so it must be keep=true with no spaces around the '=' and no quotes.
# qsub takes a copy of this script at submission time, so editing the line
# below and submitting again does not disturb a job that is already queued or
# running.
keep=false
# ---------------------------------------------------------------------------

case "${keep,,}" in
    true|1|yes)  export GS2_VARIANT=keep ;;
    false|0|no)  export GS2_VARIANT=drop ;;
    *) echo "[job] keep must be true or false, got '${keep}'" >&2; exit 1 ;;
esac
echo "[job] keep=${keep} -> GS2_VARIANT=${GS2_VARIANT}"

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
# -u is needed because piping to tee makes Python block-buffer stdout, so the
# "live" log would otherwise only appear in ~8 kB bursts.
python -u run_Seg1.py 32 resume 2>&1 | tee $WORKDIR/logs/live_gs2_segue1_${GS2_VARIANT}.log
