#!/bin/bash

# Check if a directory argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Assign the first argument to a variable
DIRECTORY=$1

# Check if the provided argument is a directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: The path provided is not a directory."
    exit 1
fi

# Find and list PNG files in the specified directory and its subdirectories
find "$DIRECTORY" -type f -iname "*.png"
