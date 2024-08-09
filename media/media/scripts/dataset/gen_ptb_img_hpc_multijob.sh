#!/bin/bash

# Base directory for PTB data
BASE_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/records100"
SAVE_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/GENERATED_IMG/NOISE_WITHOUT_ROTATION_COLUMNS_1"

# Other fixed arguments
PTBXL_DATABASE_CSV_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
PTBXL_SCP_STATEMENTS_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
# OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -rot 40 -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.8 --random_add_header 0.5 --random_resolution -r 200 --random_grid_color --fully_random -se 10"

# without rotation
# OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.8 --random_add_header 0.5 --random_resolution -r 200 --random_grid_color --fully_random -se 10"

# with rotation 1 column
OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -rot 40 -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.8 --random_add_header 0.5 --random_resolution -r 200 --random_grid_color --fully_random -se 10 --num_columns 1 --full_mode None"

# with rotation 2 columns
# OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.8 --random_add_header 0.5 --random_resolution -r 200 --random_grid_color --fully_random -se 10 --num_columns 1 --full_mode None"

# Iterate over each subfolder and submit a SLURM job for it
for SUBFOLDER in $(ls $BASE_DIR); do
    PTB_DIR="$BASE_DIR/$SUBFOLDER"
    PTB_MORE="$SAVE_DIR/more/$SUBFOLDER"  # Adjust this path based on your requirements
    PTB_MORE_IMG="$SAVE_DIR/more_img/$SUBFOLDER"  # Adjust this path based on your requirements
    mkdir -p $PTB_MORE
    mkdir -p $PTB_MORE_IMG

    # Create a unique job file for each submission
    JOB_FILE="slurm_job_${SUBFOLDER}.sh"
    cat << EOF > $JOB_FILE
#!/bin/bash
#SBATCH --job-name=ptb_processing_${SUBFOLDER}
#SBATCH --time=7-00:00:00  # Adjust based on expected runtime
#SBATCH --mem=10240
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 

bash gen_ptb_img.sh --ptb-dir $PTB_DIR --ptb-more $PTB_MORE --ptb-more-img $PTB_MORE_IMG --ptbxl-database-csv-dir $PTBXL_DATABASE_CSV_DIR --ptbxl-scp-statements-dir $PTBXL_SCP_STATEMENTS_DIR $OTHER_ARGS
EOF

    # Submit the job file
    sbatch $JOB_FILE

    # Optional: Remove the job file after submission if not needed for debugging
    # rm $JOB_FILE
done
