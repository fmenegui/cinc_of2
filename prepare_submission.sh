#!/bin/bash
# bash prepare_submission.sh dados submissao

# Check if a folder input is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 folder_input destination_directory"
    exit 1
fi

# Assign the command line arguments to variables
CURRENT_DIR="$(pwd)"
FOLDER_INPUT="$1"
DESTINATION_DIR="$2"

SUBMISSION_DIR="${DESTINATION_DIR}/submission_$(date +%Y%m%d%H%M%S)"
mkdir -p "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"/code
mkdir -p "$SUBMISSION_DIR"/model

rsync -av --exclude-from='.ignoresubmission' ./ "$SUBMISSION_DIR"/code
echo "Files copied to ${SUBMISSION_DIR}/code"

cd "$CURRENT_DIR"
echo "Preparing challenge files..."
./prepare_challenge_files.sh "$SUBMISSION_DIR"/code

echo "Preparing model and config..."
cd "$CURRENT_DIR"
./prepare_model_and_config.sh "$FOLDER_INPUT" "$SUBMISSION_DIR"

cp Dockerfile "$SUBMISSION_DIR"
cp .dockerignore "$SUBMISSION_DIR"

echo "Submission preparation complete."



