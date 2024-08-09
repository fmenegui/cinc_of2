#!/bin/bash

# Base directory for PTB data
BASE_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/records100"
SAVE_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/GENERATED_IMG/NOISE"

# Other fixed arguments
PTBXL_DATABASE_CSV_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
PTBXL_SCP_STATEMENTS_DIR="/mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
OTHER_ARGS="--store_text_bounding_box --store_config --bbox --augment -rot 40 -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.8 --random_add_header 0.5 --random_resolution -r 200 --random_grid_color --fully_random -se 10"

# Iterate over each subfolder and execute the script with adjusted arguments
for SUBFOLDER in $(ls $BASE_DIR); do
    PTB_DIR="$BASE_DIR/$SUBFOLDER"
    PTB_MORE="$SAVE_DIR/more/$SUBFOLDER"  # Adjust this path based on your requirements
    PTB_MORE_IMG="$SAVE_DIR/more_img/$SUBFOLDER"  # Adjust this path based on your requirements
    mkdir -p $PTB_MORE
    mkdir -p $PTB_MORE_IMG

    bash gen_ptb_img.sh --ptb-dir $PTB_DIR --ptb-more $PTB_MORE --ptb-more-img $PTB_MORE_IMG --ptbxl-database-csv-dir $PTBXL_DATABASE_CSV_DIR --ptbxl-scp-statements-dir $PTBXL_SCP_STATEMENTS_DIR $OTHER_ARGS
done
