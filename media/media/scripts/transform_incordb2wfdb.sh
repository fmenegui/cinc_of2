#!/bin/bash
#SBATCH --job-name=incor2wfdb
#SBATCH --time=7-00:00:00  
#SBATCH --mem=1024
#SBATCH --partition=cpu

python transform_incordb2wfdb.py
