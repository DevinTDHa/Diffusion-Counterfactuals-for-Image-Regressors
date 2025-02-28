#!/bin/bash

python run_oracle.py \
    --folder "fake_ref/square3_cf/ace_default_square3_mirror_lower" \
    --folder "fake_ref/square3_cf/ace_default_square3_mirror_upper" \
    --size 64 \
    --model "/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/square3_m1/version_0/checkpoints/epoch=0099-val_loss=0.00.ckpt" \
    --oracle "/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/square3_m2/version_0/checkpoints/epoch=0099-val_loss=0.00.ckpt" \
    --batch_size 64
