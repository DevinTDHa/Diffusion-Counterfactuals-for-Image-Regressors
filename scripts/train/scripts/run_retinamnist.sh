#!/bin/bash
#SBATCH --job-name=reg_retinaMNIST
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%x-%j.out
#SBATCH --chdir=/home/tha/master-thesis-xai/thesis_utils/scripts/train
set -x

if [ -z "$1" ]; then
	echo "Usage: $0 <name> [oracle]"
	exit 1
fi

NAME="$1"
SIZE=128
if [ -n "$2" ]; then
	echo "Running for oracle"
	ORACLE="--oracle"
	NAME="${NAME}_oracle"
fi

apptainer run \
	--nv \
	~/apptainers/thesis.sif \
	python train_resnet_retinamnist.py \
	--name "$NAME-$SIZE" \
	$ORACLE
