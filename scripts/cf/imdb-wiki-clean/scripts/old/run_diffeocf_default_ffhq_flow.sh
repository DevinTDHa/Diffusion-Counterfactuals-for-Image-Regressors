#!/bin/bash
#SBATCH --job-name=dcf_imdb
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/cf/imdb-wiki-clean

# Define variables for the arguments
GMODEL_PATH="/home/tha/diffeo-cf/models/2022_Counterfactuals_pretrained_models/checkpoints/generative_models/CelebA_Glow.pth"
GMODEL_TYPE="Flow"
RMODEL_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean-64/version_0/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean_oracle-64/version_0/checkpoints/last.ckpt"
DATASET="CelebA"
ATTACK_STYLE="z"
NUM_STEPS=100
LR=0.005
CONFIDENCE_THRESHOLD=0.05
IMAGE_FOLDER="/home/tha/datasets/ffhq_samples"
SIZE=64

export THESIS_DEBUG=true

TARGET=0.8
NAME="diffeocf_default_ffhq_flow_t=$TARGET"
RESULT_DIR="diffeocf_results/$NAME"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_default_imdb.py \
    --gmodel_path $GMODEL_PATH \
    --gmodel_type $GMODEL_TYPE \
    --rmodel_path $RMODEL_PATH \
    --roracle_path $RORACLE_PATH \
    --dataset $DATASET \
    --attack_style $ATTACK_STYLE \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    >logs/$NAME.log 2>&1 &

TARGET=0.1
NAME="diffeocf_default_ffhq_flow_t=$TARGET"
RESULT_DIR="diffeocf_results/$NAME"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_default_imdb.py \
    --gmodel_path $GMODEL_PATH \
    --gmodel_type $GMODEL_TYPE \
    --rmodel_path $RMODEL_PATH \
    --roracle_path $RORACLE_PATH \
    --dataset $DATASET \
    --attack_style $ATTACK_STYLE \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --target $TARGET \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --image_folder $IMAGE_FOLDER \
    --size $SIZE \
    --result_dir $RESULT_DIR \
    >logs/$NAME.log 2>&1 &

wait
