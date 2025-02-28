#!/bin/bash
#SBATCH --job-name=dcf_retinaMNIST
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/cf/retinaMNIST

# Define variables for the arguments
GMODEL_PATH="/home/tha/python_repos/matanat_dae_counterfactual/pretrained/retina128_epoch=9259-step=1250000.ckpt"
RMODEL_PATH="/home/tha/thesis_runs/regressor/retinaMNIST_reg-128/version_0/checkpoints/last.ckpt"
RORACLE_PATH="/home/tha/thesis_runs/regressor/retinaMNIST_oracle-128/version_0/checkpoints/last.ckpt"

# Algorithm params
NUM_STEPS=150
CONFIDENCE_THRESHOLD=0.05
BATCH_SIZE=10

# DAE params
FORWARD_T=250

# Positional arguments
LR=${1:-0.02}
BACKWARD_T=${2:-15}
DIST=${3:-"l1=1e-5"}
OPTIMIZER=${4:-adam}

# Check if the positional arguments are provided
if [ $# -gt 4 ]; then
    echo "Usage: $0 [lr] [backward_t] [dist] [optimizer]"
    exit 1
fi

echo "Running with the following parameters:"
echo "LR=$LR, BACKWARD_T=$BACKWARD_T, DIST=$DIST, OPTIMIZER=$OPTIMIZER"

RESULT_DIR="/home/tha/thesis_runs/dae/retinaMNIST/retinaMNIST_lr=${LR}_bt=${BACKWARD_T}_opt=${OPTIMIZER}"

# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    --nv \
    ~/apptainers/thesis.sif \
    python run_diffeocf_dae_retinaMNIST.py \
    --gmodel_path=$GMODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --roracle_path=$RORACLE_PATH \
    --num_steps=$NUM_STEPS \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --batch_size=$BATCH_SIZE \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --dist="$DIST" \
    --result_dir="$RESULT_DIR" \
    --forward_t=$FORWARD_T \
    --backward_t=$BACKWARD_T
