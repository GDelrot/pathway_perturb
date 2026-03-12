""" This script holds analysis function that will be applied onto
the l1000 data from sigcom lincs
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from typing import Dict,List
import gseapy as gp

def run_pca(matrix: pd.DataFrame, n_components: int = 50, batch_size: int = 10_000,
            save:bool = False,
            OUT:Path = Path(""),
            savepath:str=""):
    """Takes any matrix, returns scores + fitted model."""
    sample_ids = matrix.index
    scaler = StandardScaler(copy=False)
    scaled = scaler.fit_transform(matrix.astype(np.float32))
    
    ipca = IncrementalPCA(n_components=n_components)
    for start in range(0, scaled.shape[0], batch_size):
        ipca.partial_fit(scaled[start:start + batch_size])
    
    scores = np.vstack([
        ipca.transform(scaled[start:start + batch_size])
        for start in range(0, scaled.shape[0], batch_size)
    ])
    
    if save:
            # Save everything after PCA
        joblib.dump(ipca, OUT / "ipca_model.joblib")
        np.save(OUT / f"{savepath}_pca_scores.npy", scores)
        np.save(OUT / f"{savepath}_pca_explained_variance.npy", ipca.explained_variance_ratio_)
        np.save(OUT / f"{savepath}_pca_components.npy", ipca.components_)  # the actual axes (genes × PC)

    return pd.DataFrame(scores, index=sample_ids,
                        columns=[f"PC{i+1}" for i in range(n_components)]), ipca
    
def run_gsea(gene_matrix: pd.DataFrame, pathways: Dict) -> pd.DataFrame:
    
    # Init res df: rows=samples, cols=pathways
    gsea_res = pd.DataFrame(index=gene_matrix.columns,
                            columns=pathways.keys(),
                            dtype=float)
    for pert in gene_matrix.columns:
        # Sort descending: most upregulated on top, most downregulated at bottom
        ranked = gene_matrix.loc[:, pert].sort_values(ascending=False)
        pre_res = gp.prerank(
            rnk=ranked,
            gene_sets=pathways,
            threads=8,
            permutation_num=0
        )

        # res2d has one row per pathway — index it by Term then grab NES
        nes_series = pre_res.res2d.set_index('Term')['ES']
        print(nes_series)
        # Fill the row for this sample
        # .reindex aligns on pathway names and puts NaN where no result
        gsea_res.loc[pert] = nes_series.reindex(pathways.keys())

    return gsea_res
    