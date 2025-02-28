#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ]; then
    echo "DCFIR_OUTPATH is not defined. Please set it manually before running this script."
    exit 1
fi

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MODEL_PATH="pretrained_models/ace/celebahq-ddpm.pt"
CONFIDENCE_THRESHOLD="0.05"
IMAGE_FOLDER="datasets/CelebAMask-HQ"
IMAGE_SIZE="256"

# Attack parameters
if [ "$#" -gt 5 ]; then
    echo "Usage: $0 <attack_method=PGD> <attack_step=1.0> <dist_l1=0.0> <dist_l2=0.0> <rmodel_path>"
    exit 1
fi

ATTACK_METHOD=${1:-PGD}
ATACK_STEP=${2:-1.0}
DIST_L1=${3:-0.0} # Dist does not work well, no real results if enabled
DIST_L2=${4:-0.0}
RMODEL_PATH=${5:-"$DCFIR_OUTPATH/models/regressors/celebahq/version_0/checkpoints/last.ckpt"}
RORACLE_PATH="$DCFIR_OUTPATH/models/regressors/celebahq_oracle/version_0/checkpoints/last.ckpt"
if [[ "$RMODEL_PATH" == *"linear_only"* ]]; then
    LINEAR_ONLY=1
else
    LINEAR_ONLY=0
fi

NUM_SAMPLES=3000
MAX_STEPS=200
BATCH_SIZE=8

NAME="CelebaHQ_val-method=${ATTACK_METHOD}-step=${ATACK_STEP}-dist_l1=${DIST_L1}-dist_l2=${DIST_L2}-rmodel_linear=${LINEAR_ONLY}"
OUTPUT_PATH="$DCFIR_OUTPATH/ac-re/celebahq/$NAME"

echo "Runnning $NAME"
# Run the Python script with the arguments
python main_regression_celebahq.py $MODEL_FLAGS \
    --model_path=$MODEL_PATH \
    --rmodel_path=$RMODEL_PATH \
    --roracle_path=$RORACLE_PATH \
    --batch_size=$BATCH_SIZE \
    --attack_step=$ATACK_STEP \
    --confidence_threshold=$CONFIDENCE_THRESHOLD \
    --image_folder=$IMAGE_FOLDER \
    --image_size=$IMAGE_SIZE \
    --output_path=$OUTPUT_PATH \
    --num_samples=$NUM_SAMPLES \
    --exp_name=$NAME \
    --attack_method=$ATTACK_METHOD \
    --attack_iterations=$MAX_STEPS \
    --attack_joint=True \
    --dist_l1=$DIST_L1 \
    --dist_l2=$DIST_L2 \
    --timestep_respacing=25 \
    --sampling_time_fraction=0.2
