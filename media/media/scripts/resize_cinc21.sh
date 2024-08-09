#!/bin/bash
#SBATCH --job-name=cinc_1_None_rz
#SBATCH --time=2-00:00:00  
#SBATCH --mem=1024
#SBATCH --partition=dark
#SBATCH --nodelist=darkside3

echo 'Resize CINC: 1_None'
python resize_images.py --input /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/CINC21/1_None --output /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/CINC21_RESIZED224_2/1_None --width 224 --height 224 --method resize
                                                      
