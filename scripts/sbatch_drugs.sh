#!/bin/bash
#SBATCH --job-name=drug_features
#SBATCH --output=/home/gdelrot/pathway_perturb/outputs/slurmouts/drug_out_%A_%a.log
#SBATCH --mem=32GB                    
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=gauthier.delrot@u-bordeaux.fr
#SBATCH --mail-type=FAIL,ARRAY_TASKS

cd /home/gdelrot/pathway_perturb
source venv/bin/activate

# MOA file
MOA_FILE=/home/gdelrot/pathway_perturb/data/moa.txt

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    N=$(wc -l < "$MOA_FILE")
    echo "Submitting array of $N jobs..."
    sbatch --array=1-"$N"%25 "$0"   # $0 = path to this script itself
    exit 0
fi

# ── WORKER MODE (one task per cell line) ───────────────────────────────────────
source /home/gdelrot/pathway_perturb/venv/bin/activate

# Pick the Nth line matching this task's index (1-based)
MOA=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MOA_FILE")
echo "Task $SLURM_ARRAY_TASK_ID → processing: $MOA"
python -u scripts/drug_features.py "$MOA"
