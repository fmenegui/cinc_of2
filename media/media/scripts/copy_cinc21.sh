#!/bin/bash
#SBATCH --job-name=copy_cinc
#SBATCH --time=2-00:00:00  # Adjust based on expected runtime
#SBATCH --mem=1024
#SBATCH --partition=dark
#SBATCH --nodelist=darkside3

rsync -r --progress /mnt/processed1/signals/CINC_CHALLENGE_2024/GENERATED_IMG/CINC21 /mnt/processed2/signals/challenge_ecg_2024/CINC21

