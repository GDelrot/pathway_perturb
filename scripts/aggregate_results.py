# aggregate_results.py
"""
After all array jobs complete, concatenate per-cell-line CSVs
into combined files, one per (outname, cor_type, pert_type) combination.
Cell line is used as index.
"""
import pandas as pd
from pathlib import Path

OUTPUT = Path('/home/gdelrot/pathway_perturb/outputs/corr_cl_subsample')

for pert_type in ['raw_pert', 'pathway_pert']:
    for outname in ['moa', 'smiles']:
        for cor_type in ['pearson', 'spearman']:

            pattern = f'correlations_{outname}_{cor_type}_*_{pert_type}.csv'
            files = sorted(OUTPUT.glob(pattern))

            if not files:
                print(f"  No files found for {pattern}")
                continue

            chunks = []
            for f in files:
                # stem: correlations_smiles_spearman_YAPC_raw_pert
                # strip left prefix and right pert_type suffix to get cell line
                stem = f.stem
                stem = stem.replace(f'correlations_{outname}_{cor_type}_', '')
                cell_line = stem.replace(f'_{pert_type}', '')

                df = pd.read_csv(f, index_col=0)
                df.index = [cell_line] * len(df)   # cell line as index
                df.index.name = 'cell_line'
                chunks.append(df)

            combined = pd.concat(chunks)
            out = OUTPUT / f'correlations_{outname}_{cor_type}_{pert_type}_ALL.csv'
            combined.to_csv(out, index=True)
            print(f"Saved {out}  shape={combined.shape}")