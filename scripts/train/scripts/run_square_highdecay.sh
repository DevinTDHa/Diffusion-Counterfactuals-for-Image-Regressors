#!/bin/bash
#SBATCH --job-name=reg_squares_highdecay
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/train
set -x

NAME="square3_highdecay"
if [ -n "$2" ]; then
	ORACLE="--oracle"
fi

source /home/tha/hydra.env

apptainer run \
	-B /home/space/datasets:/home/space/datasets \
	-B /home/tha/datasets/squashed/square3.sqfs:/data/square3:image-src=/ \
	--nv \
	~/apptainers/thesis.sif \
	python train_resnet_square3.py \
	--folder_path /data/square3/ \
	--name $NAME \
	--weight_decay 0.01 \
	$ORACLE
