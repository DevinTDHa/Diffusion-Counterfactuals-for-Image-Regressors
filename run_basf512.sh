#!/bin/sh
#SBATCH --job-name=dae_basf512
#SBATCH --partition=gpu-7d
#SBATCH --constraint=80gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/diffae

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

apptainer run \
    -B /home/tha/datasets/squashed/basf_resize512.sqfs:/data/basf:image-src=/ \
    --nv /home/tha/apptainers/thesis.sif \
    python run_basf512.py --max_time="6:23:55:00"
