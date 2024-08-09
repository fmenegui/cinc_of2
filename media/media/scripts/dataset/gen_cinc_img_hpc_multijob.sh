#!/bin/bash

# CONFIG
# 4/IIX; 4/II,V1X; 2/IIX; 2/II,V1,V5X; 1/NoneX
COLUMNS="4"
FULL_MODE="II"
echo "COLUMNS: $COLUMNS, FULL_MODE: $FULL_MODE"

# Base directory for PTB data
BASE_DIR="/mnt/processed1/signals/datasets/CINC_2021_RAW/training"
FULL_MODE_SAVE=$(echo "$FULL_MODE" | sed 's/,/_/g')
SAVE_DIR="/mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_IMG/CINC21/${COLUMNS}_${FULL_MODE_SAVE}"

# Corrected other fixed arguments (missing dashes and spaces)
OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.95 --random_add_header 0.8 -r 100 --random_grid_color --fully_random -se 10 --augment --num_columns $COLUMNS --full_mode $FULL_MODE"

# Iterate over each subfolder and submit a SLURM job for it
for SUBFOLDER in $(ls "$BASE_DIR"); do
    INPUT="$BASE_DIR/$SUBFOLDER"
    for SUBSUBFOLDER in $(ls "$INPUT"); do
        INPUT="$BASE_DIR/$SUBFOLDER/$SUBSUBFOLDER"
        OUTPUT="$SAVE_DIR/$SUBFOLDER/$SUBSUBFOLDER"  # Adjust this path based on your requirements
        mkdir -p "$OUTPUT"

        # Create a unique job file for each submission
        JOB_FILE="slurm_job_${COLUMNS}_${FULL_MODE_SAVE}_${SUBFOLDER}_${SUBSUBFOLDER}.sh"
        cat <<EOF > "$JOB_FILE"
#!/bin/bash
#SBATCH --job-name=processing_${COLUMNS}_${FULL_MODE_SAVE}_${SUBFOLDER}_${SUBSUBFOLDER}
#SBATCH --time=7-00:00:00  
#SBATCH --mem=10240
#SBATCH --partition=dark
#SBATCH --nodelist=darkside2

echo "Starting dataset setup..."
echo "--input $INPUT"
bash gen_img_base.sh --input "$INPUT" --output "$OUTPUT" --save-time 1 $OTHER_ARGS
EOF

        # Submit the job file
        sbatch "$JOB_FILE"
        sleep 20

        # Optional: Remove the job file after submission if not needed for debugging
        rm "$JOB_FILE"
    done
done
