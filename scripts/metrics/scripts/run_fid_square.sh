#!/bin/bash

python run_metrics.py \
    --real_folder="real_dist/square3" \
    --fake_folder="fake_dist/square3/full_copy" \
    --fake_folder="fake_dist/square3/subset" \
    --fake_folder="fake_dist/square3/just_white" \
    --fake_folder="fake_dist/square3/just_black" \
    --limit=100000 \
    --size=64 \
    --batch_size=128 \
    --metric_type=distribution
