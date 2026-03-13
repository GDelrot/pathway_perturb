""" This script holds analysis function that will be applied onto
the l1000 data from sigcom lincs
"""
from typing import Dict, List, cast

import gseapy as gp
import numpy as np
import pandas as pd
from sklearn.decomposition import  PCA
from sklearn.preprocessing import StandardScaler

def run_pca(matrix: pd.DataFrame, n_components: int = 10):
    """Takes any matrix, returns scores + fitted model."""

    sample_ids = matrix.index
    scaler = StandardScaler(copy=False)
    scaled = scaler.fit_transform(matrix.astype(np.float32))

    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(scaled)

    return pd.DataFrame(pca_scores, index=sample_ids,
                        columns=[f"PC{i+1}" for i in range(n_components)]), pca

def run_gsea(gene_matrix: pd.DataFrame, pathways: Dict) -> pd.DataFrame:
    """Run preranked GSEA for each perturbation in the gene matrix.

    For each column (perturbation) in `gene_matrix`, genes are ranked by their
    expression values (descending) and passed to `gseapy.prerank`. The resulting
    enrichment scores (ES) are collected into a summary DataFrame.

    Args:
        gene_matrix: DataFrame of shape (genes, perturbations). Each column is
            a perturbation profile; values are used directly as the ranking metric.
        pathways: Mapping of pathway name to gene set (list of gene symbols),
            as expected by `gseapy.prerank`.

    Returns:
        DataFrame of shape (perturbations, pathways) containing ES values.
        NaN where a pathway produced no result for a given perturbation.
    """
    # Init res df: rows=samples, cols=pathways
    pathway_cols = [k for k in pathways.keys()]
    gsea_res = pd.DataFrame(index=gene_matrix.columns,
                            columns=pathway_cols,
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
        nes_series =cast(pd.DataFrame,pre_res.res2d).set_index('Term')['ES']
        print(nes_series)
        # Fill the row for this sample
        # .reindex aligns on pathway names and puts NaN where no result
        gsea_res.loc[pert] = nes_series.reindex(pathway_cols)

    return gsea_res

def run_gsva(omics_matrix: pd.DataFrame,
             pathways_dict: Dict[str, List[str]],
             min_size: int,
             max_size: int) -> pd.DataFrame:
    """Run GSVA.
    Args:
        omics_matrix: DataFrame of shape (samples, metabolites).
        pathways_dict: {pathway_id: [member_ids]}.
        min_size: Minimum set size after filtering.
        max_size: Maximum set size after filtering.
    Returns:
        DataFrame of shape (samples, pathways) with GSVA scores.
    """
    res = gp.gsva(
        data=omics_matrix.T,
        gene_sets=pathways_dict,#type:ignore
        min_size=min_size,
        max_size=max_size,
        kcdf="Gaussian",
    )
    gsva_matrix = cast(pd.DataFrame,res.res2d).pivot(index='Name', columns='Term', values='ES')

    return gsva_matrix
