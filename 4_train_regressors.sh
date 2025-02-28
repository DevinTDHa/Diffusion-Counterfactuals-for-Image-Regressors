#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ]; then
    echo "DCFIR_OUTPATH is not defined. Please set it manually before running this script."
    exit 1
fi

# Will be automatically saved in $DCFIR_OUTPATH/models/regressors
SQUARE_PATH=$DCFIR_OUTPATH/datasets/square
IMDB_CLEAN_PATH=$DCFIR_OUTPATH/datasets/imdb-clean

# Train Square Regressor
echo "Training Square Regressor..."
python scripts/train/train_resnet_square.py --folder_path $SQUARE_PATH --name square
# Results in $DCFIR_OUTPATH/models/regressors/square/version_0/checkpoints/last.ckpt

echo "Training CelebAHq regressor and oracle..."
# Weights to finetune
WEIGHTS_PATH=pretrained_models/pretrained/decision_densenet/celebamaskhq/checkpoint.tar
# Train CelebAHQ Regressor
python train_imdb_clean.py \
    --folder_path $IMDB_CLEAN_PATH \
    --name "celebahq" \
    --densenet_weights "$WEIGHTS_PATH" \
    --image_size 256
# Results in $DCFIR_OUTPATH/models/regressors/celebahq/version_0/checkpoints/last.ckpt

# Train CelebAHQ Oracle
python train_imdb_clean.py \
    --folder_path $IMDB_CLEAN_PATH \
    --name "celebahq_oracle" \
    --densenet_weights "$WEIGHTS_PATH" \
    --image_size 256 \
    $FULL_FINETUNE $ORACLE
# Results in $DCFIR_OUTPATH/models/regressors/celebahq_oracle/version_0/checkpoints/last.ckpt
