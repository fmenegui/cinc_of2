#!/bin/bash

# CONFIG
# 4/II[X]; 4/II,V1[X]; 2/II[X]; 2/II,V1,V5[X]; 1/None[X]
COLUMNS="4"
FULL_MODE="II,V1"

# com texto
# OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.95 --random_add_header 0.8 -r 100 --random_grid_color --fully_random -se 10 --augment --num_columns $COLUMNS --full_mode $FULL_MODE"

# sem texto
OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -noise 40 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.95 --random_add_header 0.8 -r 100 --random_grid_color -se 10 --augment --num_columns $COLUMNS --full_mode $FULL_MODE"
echo "COLUMNS: $COLUMNS, FULL_MODE: $FULL_MODE"

# Base directory for CINC data
BASE_DIR="/mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_WFDB_FILES/INCORDB/"
FULL_MODE_SAVE=$(echo "$FULL_MODE" | sed 's/,/_/g')
SAVE_DIR="/mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_IMG/INCORDB_AMB/${COLUMNS}_${FULL_MODE_SAVE}"

declare -a ALL_PATHS
index=0
for SUBFOLDER in "$BASE_DIR"/*/; do
    for SUBSUBFOLDER in "$SUBFOLDER"*/; do
        ALL_PATHS[index]="${SUBSUBFOLDER%/}"  
        index=$((index+1))
    done
done

JOB_FILE="slurm_job_array_${COLUMNS}_${FULL_MODE_SAVE}.sh"
cat << EOF > "$JOB_FILE"
#!/bin/bash
#SBATCH --job-name=processing_array
#SBATCH --time=3-00:00:00
#SBATCH --mem=10240
#SBATCH --partition=dark
#SBATCH --nodelist=darkside1,darkside2
#SBATCH --array=0-$((${#ALL_PATHS[@]} - 1))%100

git config --global http.postBuffer 524288000

# Embed the ALL_PATHS array directly into the script
declare -a ALL_PATHS=(${ALL_PATHS[@]})

echo "Running job array task with SLURM_ARRAY_TASK_ID: \$SLURM_ARRAY_TASK_ID"

# Fetch the directory path from the array
DIR_PATH="\${ALL_PATHS[\$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="\${DIR_PATH/#${BASE_DIR//\//\\/}/$SAVE_DIR}"
mkdir -p "\$OUTPUT_DIR"

sleep \$((SLURM_ARRAY_TASK_ID * 20)) 

echo "Processing \${DIR_PATH} into \${OUTPUT_DIR}"
echo "Args: ${OTHER_ARGS}"
bash gen_img_base.sh --input "\${DIR_PATH}" --output "\${OUTPUT_DIR}" ${OTHER_ARGS}
EOF

# Submit the job array
sbatch "$JOB_FILE"
