#!/bin/bash
#SBATCH --job-name=KNN
#SBATCH --output=/home/gdelrot/pathway_perturb/outputs/slurmouts/KNN_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-user=gauthier.delrot@u-bordeaux.fr
#SBATCH --mail-type=FAIL

# Load associate env
source /home/gdelrot/pathway_perturb/venv/bin/activate

# Run KNN script
python -u 03_signature_neighbors.py