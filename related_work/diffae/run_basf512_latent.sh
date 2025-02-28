#!/bin/sh
#SBATCH --job-name=dae_basf512_latent
#SBATCH --partition=gpu-7d
#SBATCH --constraint=80gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/diffae

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

source /home/tha/hydra.env

apptainer run \
    -B /home/tha/datasets/squashed/basf_resize512.sqfs:/data/basf:image-src=/ \
    --nv /home/tha/apptainers/thesis.sif \
    python run_basf512_infer.py

apptainer run \
    -B /home/tha/datasets/squashed/basf_resize512.sqfs:/data/basf:image-src=/ \
    --nv /home/tha/apptainers/thesis.sif \
    python run_basf512_latent.py
