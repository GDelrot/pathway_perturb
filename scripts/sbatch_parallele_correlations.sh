#!/bin/bash
#SBATCH --job-name=CCLE
#SBATCH --output=/home/gdelrot/pathway_perturb/outputs/slurmouts/l1000_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-user=gauthier.delrot@u-bordeaux.fr
#SBATCH --mail-type=FAIL
# COMMAND TO LAUNCH THE SCRIPT IN THE README
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/gdelrot/pathway_perturb/venv/bin/activate

CELL_LINES=/home/gdelrot/pathway_perturb/data/cell_lines.txt

python /home/gdelrot/pathway_perturb/scripts/02_signature_correlations.py \
    $SLURM_ARRAY_TASK_ID \
    $CELL_LINES