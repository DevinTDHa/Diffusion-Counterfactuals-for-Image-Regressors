#!/bin/bash

python /home/tha/diffae/dae_counterfactuals/diffeo_cf.py \
    --image_path /home/tha/diffae/imgs_align/sandy.png \
    --resize 256 \
    --rmodel_path "" \
    --rmodel_type red \
    --result_dir="dcf_red"
