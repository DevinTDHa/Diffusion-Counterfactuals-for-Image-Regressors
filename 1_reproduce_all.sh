#!/bin/bash

export DCFIR_OUTPATH="$PWD/diff_cf_ir_results"

# TODO: Python paths for the other two modules?

bash 2_get_datasets.sh
bash 3_train_square_generators.sh
bash 4_train_regressors.sh
bash 5_produce_results.sh
