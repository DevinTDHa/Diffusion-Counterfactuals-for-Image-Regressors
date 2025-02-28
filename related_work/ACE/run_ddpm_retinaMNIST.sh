#!/bin/bash
#SBATCH --job-name=retinaMNIST_ddpm
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%j.out
#SBATCH --chdir=/home/tha/ACE
#SBATCH --signal=SIGUSR1@600
set -x

apptainer run --nv /home/tha/apptainers/thesis.sif \
    python retinaMNIST-train-diffusion.py
