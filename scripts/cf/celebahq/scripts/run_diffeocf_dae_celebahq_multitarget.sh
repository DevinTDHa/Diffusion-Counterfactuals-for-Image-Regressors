#!/bin/bash
#SBATCH --job-name=dcf_age_multitarget
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/celebahq
#SBATCH --signal=SIGUSR1@600

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/ffhq256_autoenc/last.ckpt"

# Algorithm params
NUM_STEPS=200
CONFIDENCE_THRESHOLD=0.05

# DAE params
FORWARD_T=250

# Positional arguments
LR=${1:-0.002}
BACKWARD_T=${2:-10}
DIST=${3:-"l1=1e-5"}
DIST_TYPE=${4:-latent}
OPTIMIZER=${5:-adam}
RMODEL_PATH=${6:-"/home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt"}
RORACLE_PATH=${7:-"/home/tha/thesis_runs/regressor/imdb_wiki_densenet_fullft-256/version_0/checkpoints/last.ckpt"}
OUTPUT_PATH=${8:-"/home/tha/thesis_runs/dae/celebahq"}
if [[ "$RMODEL_PATH" == *"linear_only"* ]]; then
    LINEAR_ONLY=1
else
    LINEAR_ONLY=0
fi

# Check if the positional arguments are provided
if [ "$1" == "-h" ] || [ $# -gt 8 ]; then
    echo "Usage: $0 [lr=${LR}] [backward_t=${BACKWARD_T}] [dist=${DIST}] [dist_type=${DIST_TYPE}] [optimizer=${OPTIMIZER}] [rmodel_path=${RMODEL_PATH}] [roracle_path=${RORACLE_PATH}] [output_path=${OUTPUT_PATH}]"
    exit 0
fi

echo "Running with the following parameters:"
echo "LR=$LR, BACKWARD_T=$BACKWARD_T, DIST=$DIST, DIST_TYPE=$DIST_TYPE, OPTIMIZER=$OPTIMIZER, LINEAR_ONLY=$LINEAR_ONLY"

IMAGE_FOLDER="/data/CelebAMask-HQ/"
NAME="CelebaHQ_multi-lr=$LR-bt=$BACKWARD_T-dist=$DIST-dist_type=$DIST_TYPE-opt=$OPTIMIZER-linear_only=$LINEAR_ONLY"
RESULT_DIR="$OUTPUT_PATH/$NAME"

# Run the Python script with the arguments. It will terminate early if SIGUSR1 is received.
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_celebahq_multitarget.py \
    --gmodel_path=$GMODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --roracle_path=$RORACLE_PATH \
    --num_steps=$NUM_STEPS \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --dist="$DIST" \
    --dist_type="$DIST_TYPE" \
    --image_folder="$IMAGE_FOLDER" \
    --result_dir="$RESULT_DIR" \
    --forward_t=$FORWARD_T \
    --backward_t=$BACKWARD_T \
    --limit_samples=100
