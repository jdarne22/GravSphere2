#!/bin/bash
#PBS -N gs2_segII
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /gpfs/home/jd925/GravSphere2/logs/


set -o pipefail   # without this the exit status is tee's, so a crashed run looks successful


WORKDIR=/gpfs/home/jd925/GravSphere2

cd $WORKDIR
mkdir -p $WORKDIR/logs


source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate gravsphere

# Keep BLAS single-threaded: the parallelism is over dynesty's pool, not linalg.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


python -u run_Seg2.py 32 resume 2>&1 | tee $WORKDIR/logs/live_gs2_seg2.log
