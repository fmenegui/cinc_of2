#!/bin/bash

# Set error handling
set -euo pipefail

# Initialize variables
INPUT=""
OUTPUT=""

# Function to display help guide
show_help() {
cat << EOF
Usage: ${0##*/} --input PATH --output PATH [OPTIONS]
Generate synthetic ECG images.

    --input PATH                   Path to the dataset directory.
    --output PATH                  Output directory for prepared data.
    [OPTIONS]                      Additional options for gen_ecg_images_from_data_batch.py.

    -h, --help                       Display this help and exit.

Examples:
    ${0##*/} --input /path/to/dataset --output /path/to/output --store-text-bounding-box --bbox --random-print 0.8
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
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
if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo 'ERROR: Arguments "--input" and "--output" are required.'
    show_help
    exit 1
fi


# To absolute
INPUT=$(realpath "$INPUT")
OUTPUT=$(realpath "$OUTPUT")

# Start the dataset setup process
echo "Starting dataset setup..."

# Create a new folder `tmp` and cd into it
tmp_dir="tmp_$$_$(date +%Y%m%d_%H%M%S)_$RANDOM"
mkdir -p "$tmp_dir"
cd "$tmp_dir"
echo "Created temporary working directory."


# Run gen_ecg_images_from_data_batch.py
# GIT_CURL_VERBOSE=1  git clone --depth 1 -b fmenegui-change-segmentation https://github.com/fmenegui/ecg-image-kit.git
cp -r '/home/f.dias/repositorios/gitlab/media/base_img_generator/ecg-image-kit' .
cd ecg-image-kit/codes/ecg-image-generator
echo "Running gen_ecg_images_from_data_batch.py..."
python gen_ecg_images_from_data_batch.py -i "$INPUT" -o "$OUTPUT" "$@"

# Remove tmp dir
cd ../../../../
echo "Local: "
echo "$tmp_dir"
rm -rf "$tmp_dir"
echo "Temporary files cleaned up."
# End of script
