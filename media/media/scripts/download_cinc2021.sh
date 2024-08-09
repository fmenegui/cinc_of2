#!/bin/bash
#SBATCH --job-name=cinc_down
#SBATCH --time=7-00:00:00  # Adjust based on expected runtime
#SBATCH --mem=1024
#SBATCH --partition=cpu


cd /mnt/experiments1/felipe.dias/raw
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
