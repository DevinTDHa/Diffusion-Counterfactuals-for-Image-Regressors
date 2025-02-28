#!/bin/bash
#SBATCH --job-name=reg_red
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/train
set -x

if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Usage: $0 <name> <size>"
	exit 1
fi

NAME=$1
SIZE=$2
if [ -n "$3" ]; then
	ORACLE="--oracle"
fi

apptainer run \
	-B /home/space/datasets-sqfs/celeba.sqfs:/data/celeba:image-src=/ \
	--nv \
	~/apptainers/thesis.sif \
	python train_resnet_red.py \
	--folder_path /data/celeba/img_align_celeba/ \
	--name "$NAME-$SIZE" \
	--image_size "$SIZE" \
	--batch_size 32 \
	$ORACLE