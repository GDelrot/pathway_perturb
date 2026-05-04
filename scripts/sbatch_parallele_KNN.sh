#!/bin/bash
#SBATCH --job-name=KNN_parallele
#SBATCH --output=/home/gdelrot/pathway_perturb/outputs/slurmouts/KNN_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-user=gauthier.delrot@u-bordeaux.fr
#SBATCH --mail-type=FAIL

CELL_LINE_FILE=/home/gdelrot/pathway_perturb/data/cell_lines.txt

# ── LAUNCHER MODE ──────────────────────────────────────────────────────────────
# If SLURM_ARRAY_TASK_ID is not set, we're not inside an array yet.
# So we submit THIS script as an array and exit.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    N=$(wc -l < "$CELL_LINE_FILE")
    echo "Submitting array of $N jobs..."
    sbatch --array=1-"$N"%25 "$0"   # $0 = path to this script itself
    exit 0
fi

# ── WORKER MODE (one task per cell line) ───────────────────────────────────────
source /home/gdelrot/pathway_perturb/venv/bin/activate

# Pick the Nth line matching this task's index (1-based)
CELL_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELL_LINE_FILE")
echo "Task $SLURM_ARRAY_TASK_ID → processing: $CELL_LINE"

python -u 03_signature_neighbors_per_cell_line.py "$CELL_LINE"