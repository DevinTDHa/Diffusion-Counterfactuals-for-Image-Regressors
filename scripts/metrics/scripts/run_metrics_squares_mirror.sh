#!/bin/bash
IMGS_FOLDER="/home/tha/datasets/square3_mirror/squares_lower/imgs"
FAKE_FOLDER_ACE="/home/tha/thesis_runs/chosen_results/ace/square/square3_mirror_lower/cf"
FAKE_FOLDER_DAE="/home/tha/thesis_runs/chosen_results/dae/square/square3_mirror_lower/cf"
OUTPUT_FOLDER="/home/tha/thesis_runs/metrics/square/square3_mirror_lower"

echo "Running reference metrics for $IMGS_FOLDER"
apptainer run \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics_square.py \
    --real_folder="$IMGS_FOLDER" \
    --fake_folder $FAKE_FOLDER_ACE \
    --fake_folder $FAKE_FOLDER_DAE \
    $OUTPUT_FOLDER

IMGS_FOLDER="/home/tha/datasets/square3_mirror/squares_upper/imgs"
FAKE_FOLDER_ACE="/home/tha/thesis_runs/chosen_results/ace/square/square3_mirror_upper/cf"
FAKE_FOLDER_DAE="/home/tha/thesis_runs/chosen_results/dae/square/square3_mirror_upper/cf"
OUTPUT_FOLDER="/home/tha/thesis_runs/metrics/square/square3_mirror_upper"

echo "Running reference metrics for $IMGS_FOLDER"
apptainer run \
    --nv \
    ~/apptainers/thesis.sif \
    python run_metrics_square.py \
    --real_folder="$IMGS_FOLDER" \
    --fake_folder $FAKE_FOLDER_ACE \
    --fake_folder $FAKE_FOLDER_DAE \
    $OUTPUT_FOLDER
