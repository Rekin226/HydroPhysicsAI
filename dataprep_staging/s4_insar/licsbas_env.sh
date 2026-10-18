#!/usr/bin/env bash
# WP S4a: run a command with the separate LiCSBAS conda env and the LiCSBAS checkout on PATH /
# PYTHONPATH (what LiCSBAS's own bashrc_LiCSBAS.sh sets), single-threaded BLAS, no GPU.
# Never activates or modifies the `hydro` env. Wrap with tools/cpu_slot.sh for long jobs:
#   dataprep_staging/tools/cpu_slot.sh dataprep_staging/s4_insar/licsbas_env.sh LiCSBAS02_ml_prep.py ...
# Paths resolve relative to this file (repo) and $HOME (conda env); override with
# LICSBAS_ENV / LICSBAS_PATH.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LICSBAS_ENV="${LICSBAS_ENV:-$HOME/miniforge3/envs/licsbas}"
export LICSBAS_PATH="${LICSBAS_PATH:-$REPO/results/dataprep/insar_tools/LiCSBAS}"
export PATH="$LICSBAS_PATH/bin:$LICSBAS_ENV/bin:$PATH"
export PYTHONPATH="$LICSBAS_PATH/LiCSBAS_lib"
export GDAL_DATA="$LICSBAS_ENV/share/gdal" PROJ_DATA="$LICSBAS_ENV/share/proj"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MPLBACKEND=Agg
exec "$@"
