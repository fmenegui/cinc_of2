#!/bin/bash

# CONFIG
COLUMNS="2"
FULL_MODE="V1"

# Base directory for PTB data
BASE_DIR="/mnt/experiments2/felipe.dias/MIMIC_IV_ECG/physionet.org/files/mimic-iv-ecg/1.0/files"
FULL_MODE_SAVE=$(echo $FULL_MODE | sed 's/,/_/g')
SAVE_DIR="/mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_IMG/MIMIC_ECG/${COLUMNS}_${FULL_MODE_SAVE}"

# Other fixed arguments
OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.95 --random_add_header 0.8 -r 100 --random_grid_color --fully_random -se 10 --augment --num_columns $COLUMNS --full_mode $FULL_MODE"

# Iterate over each subfolder and submit a SLURM job for it
for SUBFOLDER in $(ls $BASE_DIR); do
    INPUT="$BASE_DIR/$SUBFOLDER"
    OUTPUT="$SAVE_DIR/$SUBFOLDER"  # Adjust this path based on your requirements
    mkdir -p $PTB_MORE
    mkdir -p $OUTPUT

    # Create a unique job file for each submission
    JOB_FILE="slurm_job_${SUBFOLDER}.sh"
    cat << EOF > $JOB_FILE
#!/bin/bash
#SBATCH --job-name=processing_${SUBFOLDER}
#SBATCH --time=7-00:00:00  # Adjust based on expected runtime
#SBATCH --mem=10240
#SBATCH --partition=dark
#SBATCH --nodelist=darkside2


bash gen_img_base.sh --input $INPUT --output $OUTPUT $OTHER_ARGS
EOF

    # Submit the job file
    sbatch $JOB_FILE
    sleep 30

    # Optional: Remove the job file after submission if not needed for debugging
    # rm $JOB_FILE
done
