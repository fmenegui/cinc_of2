#!/bin/bash
#SBATCH --job-name=ufmg_2ii
#SBATCH --time=4-00:00:00 
#SBATCH --mem=1024
#SBATCH --partition=dark
#SBATCH --nodelist=darkside3

echo 'Resize CODE15: 2_II'
python resize_images.py --input /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/CODE15/2_II --output /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/CODE15_RESIZED224/2_II --width 224 --height 224 --method resize
