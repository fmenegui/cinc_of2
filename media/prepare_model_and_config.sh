#!/bin/bash

# Check if the source folder is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 folder_input destination_directory"
    exit 1
fi

# Assign the command line argument to a variable
SOURCE_FOLDER="$1"

# Use the current directory as the destination
DESTINATION_FOLDER="$2"

# Define the file and directory to copy
FILE_TO_COPY="config_dx.py"
DIRECTORY_TO_COPY="model"

# Check if the source file exists
if [ ! -f "${SOURCE_FOLDER}/${FILE_TO_COPY}" ]; then
    echo "File ${SOURCE_FOLDER}/${FILE_TO_COPY} does not exist."
    exit 2
fi

# Check if the source directory exists
if [ ! -d "${SOURCE_FOLDER}/${DIRECTORY_TO_COPY}" ]; then
    echo "Directory ${SOURCE_FOLDER}/${DIRECTORY_TO_COPY} does not exist."
    exit 3
fi

# Copy the file
cp "${SOURCE_FOLDER}/${FILE_TO_COPY}" "${DESTINATION_FOLDER}/code/"

# Copy the directory
cp -r "${SOURCE_FOLDER}/${DIRECTORY_TO_COPY}" "${DESTINATION_FOLDER}/model/"

echo "config_dx.py and model folder successfully copied."
