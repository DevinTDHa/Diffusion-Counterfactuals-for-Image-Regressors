#!/bin/bash
#SBATCH --job-name=ace_celebahq
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/ACE/

# clip_denoised=True,  # Clipping noise
# batch_size=16,  # Batch size
# gpu="0",  # GPU index, should only be 1 gpu
# save_images=False,  # Saving all images
# num_samples=1,  # useful to sample few examples
# cudnn_deterministic=False,  # setting this to true will slow the computation time but will have identic results when using the checkpoint backwards
# # PATH ARGS
# model_path="",  # DDPM weights path
# exp_name="exp",  # Experiment name (will store the results at Output/Results/exp_name)
# # ATTACK ARGS
# seed=0,  # Random seed
# attack_method="PGD",  # Attack method (currently 'PGD', 'C&W', 'GD' and 'None' supported)
# attack_iterations=50,  # Attack iterations updates
# attack_epsilon=255,  # L inf epsilon bound (will be devided by 255)
# attack_step=1.0,  # Attack update step (will be devided by 255)
# attack_joint=True,  # Set to false to generate adversarial attacks
# attack_joint_checkpoint=False,  # use checkpoint method for backward. Beware, this will substancially slow down the CE generation!
# attack_checkpoint_backward_steps=1,  # number of DDPM iterations per backward process. We highly recommend have a larger backward steps than batch size (e.g have 2 backward steps and batch size of 1 than 1 backward step and batch size 2)
# attack_joint_shortcut=False,  # Use DiME shortcut to transfer gradients. We do not recommend it.
# # DIST ARGS
# dist_l1=0.0,  # l1 scaling factor
# dist_l2=0.0,  # l2 scaling factor
# dist_schedule="none",  # schedule for the distance loss. We did not used any for our results
# # FILTERING ARGS
# sampling_time_fraction=0.1,  # fraction of noise steps (e.g. 0.1 for 1000 smpling steps would be 100 out of 1000)
# sampling_stochastic=True,  # Set to False to remove the noise when sampling
# # POST PROCESSING
# sampling_inpaint=0.15,  # Inpainting threshold
# sampling_dilation=15,  # Dilation size for the mask generation
# # QUERY AND TARGET LABEL
# # DATASET
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MODEL_PATH="/home/tha/ACE/pretrained/celebahq-ddpm.pt"
CONFIDENCE_THRESHOLD="0.05"
IMAGE_FOLDER="/data/CelebAMask-HQ"
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
RMODEL_PATH=${5:-"/home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt"}
RORACLE_PATH="/home/tha/thesis_runs/regressor/imdb_wiki_densenet_fullft-256/version_0/checkpoints/last.ckpt"

NUM_SAMPLES=3000
MAX_STEPS=200
BATCH_SIZE=8

NAME="CelebaHQ_FR-method=${ATTACK_METHOD}-step=${ATACK_STEP}-dist_l1=${DIST_L1}-dist_l2=${DIST_L2}"
OUTPUT_PATH="/home/tha/thesis_runs/ace/celebahq/$NAME"

echo "Runnning $NAME"
# Run the Python script with the arguments
apptainer run \
    -B /home/space/datasets:/home/space/datasets \
    -B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/ \
    --nv \
    ~/apptainers/thesis.sif \
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
