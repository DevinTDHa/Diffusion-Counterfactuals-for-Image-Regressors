#!/bin/bash
#SBATCH --job-name=dcf_age
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/imdb-wiki-clean

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/ffhq256_autoenc/last.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean-256/version_0/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/master-thesis-xai/diff_cf_ir/scripts/train/runs/imdb_clean_oracle-256/version_0/checkpoints/last.ckpt"

# Algorithm params
NUM_STEPS=150
CONFIDENCE_THRESHOLD=0.05

# DAE params
FORWARD_T=250

# Positional arguments
IMAGE_FOLDER=$1
SIZE=$2
TARGET=$3
STOP_AT=$4
LR=${5:-0.0039}
BACKWARD_T=${6:-20}
DIST=${7:-none}
OPTIMIZER=${8:-adam}

# Check if the positional arguments are provided
if [ -z "$IMAGE_FOLDER" ] || [ -z "$TARGET" ] || [ -z "$STOP_AT" ]; then
    echo "Usage: $0 <image_folder> <size> <target> <stop_at> [lr] [backward_t] [dist] [optimizer]"
    exit 1
fi

echo "Running with the following parameters:"
echo "IMAGE_FOLDER=$IMAGE_FOLDER, SIZE=$SIZE, TARGET=$TARGET, STOP_AT=$STOP_AT, LR=$LR, BACKWARD_T=$BACKWARD_T, DIST=$DIST, OPTIMIZER=$OPTIMIZER"

IMAGE_FOLDER_BASENAME=$(basename $IMAGE_FOLDER)
TODAY=$(date '+%Y-%m-%d')
RESULT_DIR="/home/tha/thesis_runs/dcf_dae_res=$TODAY/dae_imgs=${IMAGE_FOLDER_BASENAME}_t=${TARGET}_dist=${DIST}_lr=${LR}_bt=${BACKWARD_T}_opt=${OPTIMIZER}"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_imdb.py \
    --gmodel_path=$GMODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --roracle_path=$RORACLE_PATH \
    --num_steps=$NUM_STEPS \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --target="$TARGET" \
    --stop_at="$STOP_AT" \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --dist="$DIST" \
    --image_folder="$IMAGE_FOLDER" \
    --size="$SIZE" \
    --result_dir="$RESULT_DIR" \
    --forward_t=$FORWARD_T \
    --backward_t=$BACKWARD_T
