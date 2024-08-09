#!/bin/bash
#SBATCH --job-name=incor_rz
#SBATCH --time=2-00:00:00  # Adjust based on expected runtime
#SBATCH --mem=1024
#SBATCH --partition=dark
#SBATCH --nodelist=darkside3


python resize_images.py --input /mnt/experiments2/felipe.dias/ECGIA_UFAL_DATASET/ANON/data --output /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/INCORAMB_RESIZED224 --width 224 --height 224 --method resize
~                                                                                        
