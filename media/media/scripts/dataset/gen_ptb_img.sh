#!/bin/bash
# bash gen_ptb_img.sh --ptb-dir /home/fdias/data/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/records100/00000/ --ptb-more more --ptb-more-img more_img --ptbxl-database-csv-dir  /home/fdias/data/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv --ptbxl-scp-statements-dir /home/fdias/data/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
# bash gen_ptb_img.sh --ptb-dir /mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/records100 --ptb-more more --ptb-more-img more_img --ptbxl-database-csv-dir /mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv --ptbxl-scp-statements-dir /mnt/experiments2/felipe.dias/CINC_CHALLENGE_2024/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv --num_columns 1
# Script to set up the PTB-XL dataset and generate synthetic ECG images

# Set error handling
set -euo pipefail

# Initialize variables
PTB_DIR=""
PTB_MORE=""
PTB_MORE_IMG=""
PTBXL_DATABASE_CSV_DIR=""
PTBXL_SCP_STATEMENTS_DIR=""

# Function to display help guide
show_help() {
cat << EOF
Usage: ${0##*/} --ptb-dir PATH --ptb-more PATH --ptb-more-img PATH [OPTIONS]
Set up the PTB-XL dataset and generate synthetic ECG images.

    --ptb-dir PATH                   Path to the PTB-XL dataset directory.
    --ptb-more PATH                  Output directory for prepared data.
    --ptb-more-img PATH              Output directory for images.
    --ptbxl-database-csv-dir PATH    (Optional) Path to the PTB-XL database CSV file.
    --ptbxl-scp-statements-dir PATH  (Optional) Path to the PTB-XL SCP statements CSV file.
    [OPTIONS]                        Additional options for gen_ecg_images_from_data_batch.py.

    -h, --help                       Display this help and exit.

Examples:
    ${0##*/} --ptb-dir /path/to/ptb-xl --ptb-more /path/to/ptb-more --ptb-more-img /path/to/ptb-more-img --store-text-bounding-box --bbox --random-print 0.8
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ptb-dir)
            PTB_DIR="$2"
            shift 2
            ;;
        --ptb-more)
            PTB_MORE="$2"
            shift 2
            ;;
        --ptb-more-img)
            PTB_MORE_IMG="$2"
            shift 2
            ;;
        --ptbxl-database-csv-dir)
            PTBXL_DATABASE_CSV_DIR="$2"
            shift 2
            ;;
        --ptbxl-scp-statements-dir)
            PTBXL_SCP_STATEMENTS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit
            ;;
        *)
            break
            ;;
    esac
done

# Check if mandatory arguments are provided
if [[ -z "$PTB_DIR" || -z "$PTB_MORE" || -z "$PTB_MORE_IMG" ]]; then
    echo 'ERROR: Arguments "--ptb-dir", "--ptb-more", and "--ptb-more-img" are required.'
    show_help
    exit 1
fi

# Set default paths if not provided
PTBXL_DATABASE_CSV_DIR=${PTBXL_DATABASE_CSV_DIR:-"$PTB_DIR/ptbxl_database.csv"}
PTBXL_SCP_STATEMENTS_DIR=${PTBXL_SCP_STATEMENTS_DIR:-"$PTB_DIR/scp_statements.csv"}

# To absolute
PTB_DIR=$(realpath "$PTB_DIR")
PTB_MORE=$(realpath "$PTB_MORE")
PTB_MORE_IMG=$(realpath "$PTB_MORE_IMG")
PTBXL_DATABASE_CSV_DIR=$(realpath "$PTBXL_DATABASE_CSV_DIR")
PTBXL_SCP_STATEMENTS_DIR=$(realpath "$PTBXL_SCP_STATEMENTS_DIR")

# Start the dataset setup process
echo "Starting PTB-XL dataset setup..."
# conda activate myenv

# Step 1: Create a new folder `tmp` and cd into it
tmp_dir="tmp_$$_$(date +%Y%m%d_%H%M%S)_$RANDOM"
mkdir -p "$tmp_dir"
cd "$tmp_dir"
echo "Created temporary working directory."

# Step 2: Clone the required repositories
echo "Cloning the necessary repositories..."
git clone https://github.com/physionetchallenges/python-example-2024.git
# git clone https://github.com/alphanumericslab/ecg-image-kit.git
git clone -b fmenegui-change https://github.com/fmenegui/ecg-image-kit.git
# git clone -b fmenegui-change git@github.com:fmenegui/ecg-image-kit.git


# Step 3: cd to the python-example-2024 repository
cd python-example-2024
echo "Entered python-example-2024 repository."

# Step 4: Add ecg-image-generator to PATH
# ECG_IMAGE_GENERATOR_PATH="../ecg-image-kit/codes/ecg-image-generator"
# export PATH="$ECG_IMAGE_GENERATOR_PATH:$PATH"
# echo "Added ecg-image-generator to PATH."

# Step 5: Run prepare_ptbxl_data.py
echo "Running prepare_ptbxl_data.py..."
echo $PTB_DIR $PTB_MORE $PTBXL_DATABASE_CSV_DIR $PTBXL_SCP_STATEMENTS_DIR
python prepare_ptbxl_data.py -i "$PTB_DIR" -o "$PTB_MORE" --database_file "$PTBXL_DATABASE_CSV_DIR" --statements_file "$PTBXL_SCP_STATEMENTS_DIR"

# Step 6: Run gen_ecg_images_from_data_batch.py
cd ../ecg-image-kit/codes/ecg-image-generator
# echo "Installing requirements (myenv)..."
# pip install -r requirements.txt
echo "Running gen_ecg_images_from_data_batch.py..."
python gen_ecg_images_from_data_batch.py -i "$PTB_MORE" -o "$PTB_MORE_IMG" "$@"

# Step 7: Add filenames to header files (.hea)
cd ../../../python-example-2024
echo "Adding filenames to header files (.hea)..."
python add_image_filenames.py -i "$PTB_MORE_IMG" -o "$PTB_MORE_IMG"

# Clean up: Remove the temporary directory
cd ../..
rm -rf tmp
echo "Temporary files cleaned up."

# End of script
cd ../../../python-example-2024
echo "Adding filenames to header files (.hea)..."
python add_image_filenames.py -i "$PTB_MORE_IMG" -o "$PTB_MORE_IMG"

# Clean up: Remove the temporary directory
cd ../..
rm -rf "$tmp_dir"
echo "Temporary files cleaned up."

# End of script
