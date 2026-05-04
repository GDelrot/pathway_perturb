# pathway_perturb scripts
# loader.py -> handles CCLE/data loading for main script
# analysis.py -> holds analysis feature
# methods_ML.py -> holder all ML realted tasks
# pathways.py -> features related to pathways
# pathway coverage -> script to check coverage of pathways
# id_harmonizer -> pipeline for CCLE/L1000 cell lines normalization to cellosaurus
# 01_data_exploration -> script for first exploration of CCLE/L1000 data (PCAs..)
# 02_signature_correlations -> second intention exploration: try to identify signal within the data (drug patterns, etc...), correlation between groups of drugs/ comparison with null distribiution
# ── 2026-04-15  Precompute cell lines ──────────────────────────────────────
python scripts/precompute_cell_lines.py

# ── 2026-04-15  Launch correlation array ───────────────────────────────────
N=$(wc -l < /home/gdelrot/pathway_perturb/data/cell_lines.txt)
ARRAY_JOB_ID=$(sbatch --array=0-$((N-1))%25 sbatch_parallele_correlations.sh | awk '{print $4}')
echo "Array job ID: $ARRAY_JOB_ID"   # 129546

# ── 2026-04-15  Aggregation after array ────────────────────────────────────
sbatch --dependency=afterok:$ARRAY_JOB_ID aggregate_results.sh

# 03_signature_neighbors -> Script for computing clusters/ appreciate wether perturbations tend to clusters by MOAs/ smiles groups

