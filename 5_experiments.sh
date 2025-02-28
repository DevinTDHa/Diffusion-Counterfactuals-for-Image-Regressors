#!/bin/bash
if [ -z "$DCFIR_OUTPATH" ]; then
    echo "DCFIR_OUTPATH is not defined. Please set it manually before running this script."
    exit 1
fi

# ---------------- AC-RE ----------------
echo "Running AC-RE experiments..."
echo "Running Square experiments..."
bash related_work/ACE/scripts/regression/run_square_mirror.sh

echo "Running CelebAHQ experiments..."
bash related_work/ACE/scripts/regression/run_age_celebahq_allval.sh

echo "Running fine-grained reference values..."
bash related_work/ACE/scripts/regression/run_age_celebahq_multitarget.sh

# ---------------- DIFF-AE-RE ----------------
echo "Running Diff-AE-RE experiments..."
echo "Running Square experiments..."

echo "Running Square experiments..."
bash scripts/run_diffeocf_dae_square_mirror.sh

echo "Running CelebAHQ experiments..."
bash scripts/cf/celebahq/scripts/run_diffeocf_dae_celebahq_multitarget.sh
