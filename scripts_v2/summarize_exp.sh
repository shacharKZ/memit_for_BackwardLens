#!/bin/bash
set -e


ALG_NAMES=(
        "SHIFT" "FT" "MEND" "ROME" "MEMIT" "DUMMY"
        )

# Execute
for i in ${!ALG_NAMES[@]}
do
    alg_name_is_dir_name=${ALG_NAMES[$i]}

    echo "Running summarization for $alg_name_is_dir_name..."

    python3 -m experiments.summarize --dir_name=$alg_name_is_dir_name
done

exit 0
