#!/bin/bash
#SBATCH --job-name=reg_basf
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/train
set -x

if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Usage: $0 <name> <size> [oracle]"
	exit 1
fi

NAME="$1"
SIZE="$2"
if [ -n "$3" ]; then
	echo "Running for oracle"
	ORACLE="--oracle"
	NAME="${NAME}_oracle"
fi

apptainer run \
	-B /home/tha/datasets/squashed/basf_resize512.sqfs:/data/basf_resize512:image-src=/ \
	--nv \
	~/apptainers/thesis.sif \
	python train_resnet_basf.py \
	--folder_path /data/basf_resize512 \
	--name "$NAME-$SIZE" \
	--image_size "$SIZE" \
	$ORACLE
