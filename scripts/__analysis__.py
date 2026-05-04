""" This script holds analysis function that will be applied onto
the l1000 data from sigcom lincs
"""
import itertools
import random
from typing import Dict, List, cast
from venv import logger

import gseapy as gp
import numpy as np
import pandas as pd
import gc
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.decomposition import  PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from __loader__ import Loader

def run_pca(matrix: pd.DataFrame, n_components: int = 10):
    """Takes any matrix, returns scores + fitted model."""

    sample_ids = matrix.index
    scaler = StandardScaler(copy=False)
    scaled = scaler.fit_transform(matrix.astype(np.float32))

    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(scaled)

    return pd.DataFrame(pca_scores, index=sample_ids,
                        columns=[f"PC{i+1}" for i in range(n_components)]), pca

def run_umap(matrix: pd.DataFrame,
            n_components: int = 2,n_neighbors: int = 15,
            min_dist: float = 0.1):
    """
    Takes any matrix, returns scores + fitted model (UMAP version).
    
    Args:
        matrix: pd.DataFrame with samples as rows, features as columns
        n_components: number of dimensions (usually 2 for visualization)
        n_neighbors: size of local neighborhood (5-50, default 15)
        min_dist: minimum distance between points (0.0-0.99, default 0.1)
    
    Returns:
        umap_scores: DataFrame with UMAP coordinates
        umap_model: fitted UMAP object
    """
    sample_ids = matrix.index

    # Scale data (same as PCA)
    scaler = StandardScaler(copy=False)
    scaled = scaler.fit_transform(matrix.astype(np.float32))

    # Fit UMAP
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42  # For reproducibility
    )
    umap_scores = umap_model.fit_transform(scaled)

    return pd.DataFrame(
        umap_scores, 
        index=sample_ids,
        columns=[f"UMAP{i+1}" for i in range(n_components)]
    ), umap_model

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
             max_size: int,
             omics:str) -> pd.DataFrame:
    """Run GSVA.
    Args:
        omics_matrix: DataFrame of shape (samples, metabolites).
        pathways_dict: {pathway_id: [member_ids]}.
        min_size: Minimum set size after filtering.
        max_size: Maximum set size after filtering.
    Returns:
        DataFrame of shape (samples, pathways) with GSVA scores.
    """
    if omics == 'metabolomics':
        kcdf = 'Gaussian'

    elif omics == "transcriptomics":
        kcdf = 'Poisson'

    else:
        raise ValueError(f"Unsupported omics type: {omics!r}. Expected 'metabolomics' or 'transcriptomics'.")

    res = gp.gsva(
        data=omics_matrix.T,
        gene_sets=pathways_dict,#type:ignore
        min_size=min_size,
        max_size=max_size,
        kcdf=kcdf,
    )
    gsva_matrix = cast(pd.DataFrame,res.res2d).pivot(index='Name', columns='Term', values='ES')

    return gsva_matrix

def annotate_drug_info(perturbations:pd.DataFrame,
                       l1000_obj:Loader,
                       drug:str='moa'):
    """
    Annotates the l1000 pathway scores with drug metadata

    Args:
        drug (str, optional): _description_. Defaults to 'moa'.
    """
    annotated_l1000 = pd.DataFrame()

    # Duplicates in chemical tables due to several targets per drugs
    chemical_table = l1000_obj.compound_info.copy()
    sig_info = l1000_obj.sig_info.loc[:,['pert_id','sig_id']]

    chemical_table = chemical_table.drop(columns=['target']).drop_duplicates(subset='pert_id').copy()
    drug_metadata = pd.merge(sig_info, chemical_table, on='pert_id',how = 'left')
    drug_metadata = drug_metadata.dropna(subset=drug)
    drug_metadata = drug_metadata.set_index('sig_id')[drug]

    pert_drugs = set(drug_metadata.index)
    l1000_index = set(perturbations.index)

    common_samples = l1000_index.intersection(pert_drugs)
    l1000_pathway_data = perturbations.loc[list(common_samples)].copy()
    annotated_l1000 = l1000_pathway_data.join(drug_metadata,how='left')

    del l1000_pathway_data; gc.collect()

    return annotated_l1000

def annotate_cell_info(l1000_drug_ann:pd.DataFrame,
                       l1000_obj:Loader,
                       filter_cell_line=None):
    """Merge cell line annotations onto a drug-annotated DataFrame.

    Joins sig_info and cell_info on cell_iname, then left-joins onto
    l1000_drug_ann by sig_id. Optionally filters to a subset of cell lines.
    """
    cell_info = l1000_obj.cell_info.copy()
    sig_info = l1000_obj.sig_info.copy()

    cell_annotations = pd.merge(sig_info,cell_info,on='cell_iname')
    cell_annotations = cell_annotations.dropna(subset='cell_iname')
    cell_annotations = cell_annotations.set_index('sig_id')['cell_iname']

    annotated_df = l1000_drug_ann.join(cell_annotations,how='left')
    del l1000_drug_ann;gc.collect()

    if filter_cell_line:
        annotated_df = annotated_df[annotated_df['cell_iname'].isin(filter_cell_line)]

    return annotated_df

def run_cor(vector_a:np.array,
            vector_b:np.array,
            cor_type:str='pearson'):
    """Compute pairwise correlation between two vectors.

    Returns the correlation statistic (float). Supports 'pearson' and 'spearman'.
    """
    # Drop metadata columns, keep only pathway score columns
    a_vec = vector_a
    b_vec = vector_b

    if cor_type == 'pearson':

        cor_stat = np.abs(pearsonr(a_vec,b_vec).statistic)
    elif cor_type == 'spearman':

        cor_stat = np.abs(spearmanr(a_vec,b_vec).statistic)
    else:
        raise ValueError(f'Unsupported cor type: {cor_type}')

    return cor_stat

def sample_inter_moa_correlations(l1000_subset,metadata, feature_cols, n_samples, cor_type='pearson'):
    """
    Sample random pairs of perturbations from DIFFERENT MOAs.
    n_samples should match the total number of intra-MOA pairs for comparability.
    """
    indexed_by_moa = l1000_subset.groupby(metadata).groups  # {moa: [sig_ids]}
    moas = list(indexed_by_moa.keys())

    if len(moas) < 2:
        return [] 

    inter_cors = []
    features = l1000_subset[feature_cols]
    
    for _ in range(n_samples):
        # Pick two different MOAs
        moa_a, moa_b = random.sample(moas, 2)
        # Pick one random perturbation from each
        sig_a = random.choice(indexed_by_moa[moa_a])
        sig_b = random.choice(indexed_by_moa[moa_b])

        cor = run_cor(features.loc[sig_a], features.loc[sig_b], cor_type=cor_type)
        inter_cors.append(cor)

    return inter_cors

def run_moa_signature_correlations_chunked(
    perturbations:pd.DataFrame,
    l1000_obj: Loader,
    cell_line: str,
    metadata: str = 'moa',
    cor_type: str = 'pearson',
    max_pairs:int= 500
) -> tuple[pd.DataFrame | None, dict | None]:
    """Compute intra- vs inter-MOA correlations for a SINGLE cell line.
    Args:
        max_pairs: If set, randomly subsample each MOA's pairs to at most
                   this many. Useful when cell lines have highly variable
                   numbers of perturbations per MOA.

    Returns (summary_row, corr_dict) or (None, None) if no data.
    summary_row is a 1-row DataFrame (index=cell_line, cols=MOA names).
    """
    l1000_drug_annotated = annotate_drug_info(perturbations = perturbations, l1000_obj=l1000_obj, drug=metadata)
    pair_annotated = annotate_cell_info(
        l1000_drug_ann=l1000_drug_annotated,
        filter_cell_line=[cell_line],
        l1000_obj=l1000_obj,
    )
    if pair_annotated.empty:
        return None, None
    metadata_cols = [metadata, 'cell_iname']
    feature_cols = [c for c in pair_annotated.columns if c not in metadata_cols]
    logger.info(pair_annotated.head())
    cl_data = pair_annotated[pair_annotated['cell_iname'] == cell_line]
    if cl_data.shape[0] < 2:
        return None, None

    corr_dict: dict[str, list[float]] = {}

    for unique_meta in cl_data[metadata].unique():
        
        moa_subset = cl_data[cl_data[metadata] == unique_meta]
        logger.debug('Subset daframe before correlations: \n %s of shape %s',
                     moa_subset.head(),moa_subset.shape)
        if moa_subset.shape[0] < 2:
            continue
        logger.info('N different perturbations for combo %s, %s: %s\n',
                    cell_line, unique_meta, moa_subset.shape[0])
        features = moa_subset[feature_cols].values  
        
        pairs = list(itertools.combinations(range(moa_subset.shape[0]), r=2))

        # ── Subsampling ────────────────────────────────────────────────────
        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)
        # ──────────────────────────────────────────────────────────────────

        logger.info(
            "MOA %s | cell %s | %d perturbations → %d pairs (max_pairs=%s)",
          unique_meta, cell_line, moa_subset.shape[0], len(pairs), max_pairs,
        )
        
        corr_dict[unique_meta] = [
            _fast_cor(features[a], features[b], cor_type) for a, b in pairs
        ]

    n_intra = sum(len(v) for v in corr_dict.values())
    if n_intra == 0:
        return None, None

    inter_cors = sample_inter_moa_correlations(
        cl_data, metadata, feature_cols, n_samples=n_intra, cor_type=cor_type
    )
    corr_dict['__inter_moa__'] = inter_cors

    summary = {
        moa: np.median(np.abs(cors)) if cors else np.nan
        for moa, cors in corr_dict.items()
    }
    summary_row = pd.DataFrame(summary, index=[cell_line])

    n_above = (summary_row > summary_row['__inter_moa__'].iloc[0]).sum().sum()

    logger.info(
        '\n Fraction of medians above inter threshold: %s',
        n_above/(summary_row.shape[1]-1))

    return summary_row, corr_dict

def _fast_cor(a: np.ndarray, b: np.ndarray, cor_type: str) -> float:
    """Correlation between two 1-D arrays.
    Uses numpy for Pearson (avoids pandas overhead).
    Falls back to scipy for Spearman.
    """
    if cor_type == 'pearson':
        return float(np.corrcoef(a, b)[0, 1])
    else:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
