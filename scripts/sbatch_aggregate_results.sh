#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --output=/home/gdelrot/pathway_perturb/outputs/slurmouts/aggregate_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

source /home/gdelrot/pathway_perturb/venv/bin/activate

python /home/gdelrot/pathway_perturb/scripts/aggregate_results.py