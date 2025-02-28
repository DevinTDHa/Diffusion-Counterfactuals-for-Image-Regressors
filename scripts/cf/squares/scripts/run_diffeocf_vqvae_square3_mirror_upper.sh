#!/bin/bash
#SBATCH --job-name=dcf_square
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/squares
set -x

source /home/tha/hydra.env

# Define variables for the arguments
GMODEL_CONFIG="/home/tha/PyTorch-VAE/runs/VQVAE_square3/version_0/square_vq_vae.yaml"
GMODEL_PATH="/home/tha/PyTorch-VAE/runs/VQVAE_square3/version_0/checkpoints/epoch=94-step=9500.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/square3/version_0/checkpoints/last.ckpt"
ATTACK_STYLE="z"
NUM_STEPS=100
LR=0.005
CONFIDENCE_THRESHOLD=0.05
TARGET=mirror
IMG_FOLDER="/home/tha/datasets/square3_mirrored/squares_upper/"
SIZE=64
RESULT_DIR="/home/tha/thesis_runs/diffeocf_results/square3_mirror_upper_vqvae"

apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    -B /home/tha/datasets/squashed/square3.sqfs:/data/square3:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_vqvae_square3.py \
    --gmodel_path $GMODEL_PATH \
    --gmodel_config $GMODEL_CONFIG \
    --rmodel_path $RMODEL_PATH \
    --attack_style $ATTACK_STYLE \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMG_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR
