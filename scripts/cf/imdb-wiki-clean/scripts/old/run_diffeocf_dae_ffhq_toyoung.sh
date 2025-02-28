#!/bin/bash
#SBATCH --job-name=dcf_ffhq
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/cf/imdb-wiki-clean

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/ffhq256_autoenc/last.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean_oracle-256/version_0/checkpoints/last.ckpt"
NUM_STEPS=100
LR=0.01
CONFIDENCE_THRESHOLD=0.05
IMAGE_FOLDER="/home/tha/datasets/ffhq_samples"
SIZE=256

# DAE params
FORWARD_T=250
BACKWARD_T=20

STOP_AT=0.1
TARGET=0.1
RESULT_DIR="diffeocf_dae_results/ffhq_dae_t=$TARGET"

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
    --backward_t $BACKWARD_T

TARGET="-inf"
RESULT_DIR="diffeocf_dae_results/ffhq_dae_t=$TARGET"

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
    --target=$TARGET \
    --stop_at $STOP_AT \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    --forward_t $FORWARD_T \
    --backward_t $BACKWARD_T