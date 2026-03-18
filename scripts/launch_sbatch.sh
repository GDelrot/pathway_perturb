#!/bin/bash
#SBATCH --job-name=CCLE
#SBATCH --output=outputs/slurmouts/l1000_out_%j.log   # stdout
#SBATCH --error=outputs/slurmouts/l1000_err_%j.log     # stderr separately
#SBATCH --mem=32GB                        
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=gauthier.delrot@u-bordeaux.fr
#SBATCH --mail-type=FAIL

cd /home/gdelrot/pathway_perturb
source venv/bin/activate

# Run python script
python -u scripts/data_exploration.py

echo "Done on $(date)"