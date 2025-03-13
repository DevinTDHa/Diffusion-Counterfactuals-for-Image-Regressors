#!/bin/bash
set -e
if [ -z "$DCFIR_OUTPATH" ] || [ -z "$DCFIR_HOME" ]; then
    echo "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    exit 1
fi

# ---------------- AC-RE ----------------
echo "Running AC-RE experiments..."
cd $DCFIR_HOME/related_work/ACE || exit

echo "Running Square experiments..."
bash related_work/ACE/scripts/regression/run_square_mirror.sh

echo "Running CelebAHQ experiments..."
bash $DCFIR_HOME/related_work/ACE/scripts/regression/run_age_celebahq_allval.sh

echo "Running fine-grained reference values..."
bash $DCFIR_HOME/related_work/ACE/scripts/regression/run_age_imdbwiki_multitarget.sh

# ---------------- DIFF-AE-RE ----------------
echo "Running Diff-AE-RE experiments..."
cd $DCFIR_HOME || exit
echo "Running Square experiments..."
bash $DCFIR_HOME/scripts/cf/squares/scripts/run_diffeocf_dae_square_mirror.sh

echo "Running CelebAHQ experiments..."
bash $DCFIR_HOME/scripts/cf/celebahq/scripts/run_diffeocf_dae_celebahq.sh

echo "Running fine-grained reference values..."
bash $DCFIR_HOME/scripts/cf/imdb-wiki-clean/scripts/run_multitarget.sh

# ---------------- Plots ----------------
echo "Creating Square Plots"
SQUARE_REGRESSOR="$DCFIR_OUTPATH/regressors/square/version_0/checkpoints/last.ckpt"

echo "Upper Squares"
MASKS_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_upper"
RESULT_FOLDER_ACRE="$DCFIR_OUTPATH/ac-re/square_mirror_upper"
RESULT_FOLDER_DIFFAERE="$DCFIR_OUTPATH/diffae-re/square_mirror_upper"
OUTPUT_FOLDER="$DCFIR_OUTPATH/plots/square_mirror"
python $DCFIR_HOME/scripts/plots/square_viz.py $RESULT_FOLDER_ACRE $MASKS_FOLDER $SQUARE_REGRESSOR $OUTPUT_FOLDER/acre_upper.svg
# Result in /tmp/diff_cf_ir_results/plots/square_mirror/acre_upper.svg
python $DCFIR_HOME/scripts/plots/square_viz.py $RESULT_FOLDER_DIFFAERE $MASKS_FOLDER $SQUARE_REGRESSOR $OUTPUT_FOLDER/diffaere_upper.svg
# Result in /tmp/diff_cf_ir_results/plots/square_mirror/diffaere_upper.svg

echo "Lower Squares"
MASKS_FOLDER="$DCFIR_OUTPATH/datasets/square_val/squares_lower"
RESULT_FOLDER_ACRE="$DCFIR_OUTPATH/ac-re/square_mirror_lower"
RESULT_FOLDER_DIFFAERE="$DCFIR_OUTPATH/diffae-re/square_mirror_lower"
OUTPUT_FOLDER="$DCFIR_OUTPATH/plots/square_mirror"
python $DCFIR_HOME/scripts/plots/square_viz.py $RESULT_FOLDER_ACRE $MASKS_FOLDER $SQUARE_REGRESSOR $OUTPUT_FOLDER/acre_lower.svg
# Result in /tmp/diff_cf_ir_results/plots/square_mirror/acre_lower.svg
python $DCFIR_HOME/scripts/plots/square_viz.py $RESULT_FOLDER_DIFFAERE $MASKS_FOLDER $SQUARE_REGRESSOR $OUTPUT_FOLDER/diffaere_lower.svg
# Result in /tmp/diff_cf_ir_results/plots/square_mirror/diffaere_lower.svg


# ---------------- Ablation ----------------
# TODO
