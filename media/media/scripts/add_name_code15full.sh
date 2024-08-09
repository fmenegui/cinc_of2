#!/bin/bash
#SBATCH --job-name=addname_code
#SBATCH --time=7-00:00:00
#SBATCH --mem=10240
#SBATCH --partition=cpu
#SBATCH --nodelist=saci4cpu-0,saci4cpu-2,saci4cpu-3

python add_image_filenames.py -i /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/CODE15 -o /mnt/processed2/signals/cinc2021_image/documents/CINC_CHALLENGE_2024/GENERATED_IMG/CODE15