#!/bin/bash
#SBATCH --job-name=ML_pert
#SBATCH --output=/home/gdelrot/pathway_perturb/outputs/slurmouts/ML_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:30:00

source /home/gdelrot/pathway_perturb/venv/bin/activate

python -u /home/gdelrot/pathway_perturb/scripts/main.py