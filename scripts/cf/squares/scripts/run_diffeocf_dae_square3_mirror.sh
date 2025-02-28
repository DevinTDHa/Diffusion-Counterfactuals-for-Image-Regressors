#!/bin/bash
#SBATCH --job-name=dcf_square
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --constraint="80gb"
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/cf/squares
SQFS_FILE="/home/tha/datasets/squashed/square3_mirror.sqfs"
if [ ! -f /tmp/data.sqfs ]; then
    cp $SQFS_FILE /tmp/data.sqfs
fi

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffae/checkpoints/square64_ddim/last.ckpt"
RMODEL_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/square3/version_0/checkpoints/last.ckpt"
NUM_STEPS=200
BATCH_SIZE=38
LR=0.002
CONFIDENCE_THRESHOLD=0.05

BACKWARD_T=10

IMG_FOLDER="/data/square3/squares_lower/"
RESULT_DIR="/home/tha/thesis_runs/dae/square3_mirror_lower"
apptainer run \
    -B /tmp/data.sqfs:/data/square3:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_square3.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --backward_t $BACKWARD_T \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMG_FOLDER \
    --result_dir $RESULT_DIR \
    >logs/square3_mirror_lower_dae.log 2>&1 &

IMG_FOLDER="/data/square3/squares_upper/"
RESULT_DIR="/home/tha/thesis_runs/dae/square3_mirror_upper"
apptainer run \
    -B /tmp/data.sqfs:/data/square3:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_square3.py \
    --gmodel_path $GMODEL_PATH \
    --rmodel_path $RMODEL_PATH \
    --backward_t $BACKWARD_T \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMG_FOLDER \
    --result_dir $RESULT_DIR \
    >logs/square3_mirror_upper_dae.log 2>&1 &

wait
