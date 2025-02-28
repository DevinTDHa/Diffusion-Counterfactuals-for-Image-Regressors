#!/bin/bash
#SBATCH --job-name=dcf_ffhq
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/cf/basf

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/basf512_ddim/last.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/basf-512/version_1/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/basf_oracle-512/version_1/checkpoints/last.ckpt"
NUM_STEPS=100
LR=0.01
CONFIDENCE_THRESHOLD=0.05
IMAGE_FOLDER="/data/basf_resize512"
SIZE=256

# DAE params
FORWARD_T=250
# BACKWARD_T=10 # 52 GB
BACKWARD_T=15 # 76.5 GB

TARGET=0.6
STOP_AT=0.6
RESULT_DIR="/home/tha/thesis_runs/diffeocf_dae_results/basf_dae_t=$TARGET"

# Run the Python script with the arguments
apptainer run \
    -B /home/tha/datasets/squashed/basf_resize512.sqfs:/data/basf_resize512:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_basf.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --roracle_path $RORACLE_PATH \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --stop_at $STOP_AT \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    --forward_t $FORWARD_T \
    --backward_t $BACKWARD_T \
    --limit_samples 10

TARGET=0.0
STOP_AT=0.0
RESULT_DIR="/home/tha/thesis_runs/diffeocf_dae_results/basf_dae_t=$TARGET"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_imdb.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --roracle_path $RORACLE_PATH \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --stop_at $STOP_AT \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    --forward_t $FORWARD_T \
    --backward_t $BACKWARD_T \
    --limit_samples 10
