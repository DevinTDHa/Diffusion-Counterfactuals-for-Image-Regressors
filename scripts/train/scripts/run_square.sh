#!/bin/bash
#SBATCH --job-name=reg_squares
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/train
set -x

if [ -z "$1" ]; then
	echo "Usage: $0 <name> [oracle]"
	exit 1
fi

NAME=$1
if [ -n "$2" ]; then
	echo "Running for oracle"
	ORACLE="--oracle"
	NAME="${NAME}_oracle"
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
	$ORACLE
