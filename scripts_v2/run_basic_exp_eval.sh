#!/bin/bash
set -e

# Constants
N_EDITS=$1
DATASET=$2
N_LIM=$3


# Run configurations
MODEL_NAME="gpt2-xl"
ALG_NAMES=(
        "SHIFT" "FT" "MEND" "ROME" "MEMIT" "DUMMY"
        )
HPARAMS_FNAMES=(
    gpt2-xl_v1.json gpt2-xl_unconstr.json "gpt2-xl.json" "gpt2-xl.json" "gpt2-xl.json" "gpt2-xl.json"
    )

# Execute
for i in ${!ALG_NAMES[@]}
do
    alg_name=${ALG_NAMES[$i]}
    hparams_fname=${HPARAMS_FNAMES[$i]}

    echo "Running evals for $alg_name..."

    python -m experiments.evaluate --alg_name=$alg_name --model_name=$MODEL_NAME --hparams_fname=$hparams_fname --num_edits=$N_EDITS --dataset_size_limit=$N_LIM --ds_name=$DATASET --skip_generation_tests
done

exit 0
