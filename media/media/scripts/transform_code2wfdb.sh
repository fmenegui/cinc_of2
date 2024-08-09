#!/bin/bash
#SBATCH --job-name=code2wfdb
#SBATCH --time=7-00:00:00  # Adjust based on expected runtime
#SBATCH --mem=1024
#SBATCH --partition=cpu

python transform_code2wfdb.py
