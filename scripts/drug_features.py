
"""
smiles_similarity.py
Compute pairwise Tanimoto similarity between drug SMILES,
then cluster them via hierarchical clustering.
Adapted and simplified from johaGL / smpath (CBiB).
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loader import Loader
from loader import CCLEpaths
from loader import LINCSpaths

# Paths
INPUT = Path('/home/gdelrot/pathway_perturb/data')
OUTPUT = Path('/home/gdelrot/pathway_perturb/outputs/out_exploration_2')
MNT_L1000 = Path('/mnt/cbib/l1000/data/')

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SMILES  →  Morgan fingerprints
# ──────────────────────────────────────────────────────────────────────────────

def smiles_to_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[list[str], list]:
    """
    Convert a list of SMILES strings to RDKit Morgan fingerprints.
    Invalid SMILES are silently dropped.
 
    Returns
    -------
    valid_smiles : list[str]   – SMILES that parsed correctly
    fps          : list        – corresponding RDKit ExplicitBitVect objects
    """
    valid_smiles, fps = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        valid_smiles.append(smi)
        fps.append(fp)
    return valid_smiles, fps
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 2.  Pairwise Tanimoto similarity matrix
# ──────────────────────────────────────────────────────────────────────────────
 
def compute_tanimoto_matrix(fps: list) -> np.ndarray:
    """
    Compute the symmetric N×N Tanimoto similarity matrix from a list of
    RDKit fingerprints.  Diagonal is set to 1 (self-similarity).
    """
    n = len(fps)
    arr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            arr[i, j] = sim
            arr[j, i] = sim
    np.fill_diagonal(arr, 1.0)
    return arr
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 3.  Hierarchical clustering
# ──────────────────────────────────────────────────────────────────────────────
 
def cluster_by_similarity(
    similarity_matrix: np.ndarray,
    cutoff: float = 0.4,
    linkage_method: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a similarity matrix to a distance matrix, run hierarchical
    clustering, and cut the dendrogram at `cutoff`.
 
    Parameters
    ----------
    similarity_matrix : np.ndarray  – N×N symmetric matrix, values in [0, 1]
    cutoff            : float       – distance threshold to cut the tree
                                      (0 = all same cluster, 1 = all separate)
    linkage_method    : str         – 'ward', 'average', 'complete', etc.
 
    Returns
    -------
    Z        : np.ndarray – linkage matrix (for plotting)
    clusters : np.ndarray – integer cluster label per molecule
    """
    distance_matrix = 1.0 - similarity_matrix
    # squareform converts N×N → condensed upper-triangle vector (required by linkage)
    Z = linkage(squareform(distance_matrix), method=linkage_method)
    clusters = fcluster(Z, t=cutoff, criterion="distance")
    return Z, clusters

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Convenience wrapper: SMILES list  →  annotated DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def smiles_to_cluster_df(
    smiles_series: pd.Series,
    names_series: Optional[pd.Series] = None,
    radius: int = 2,
    n_bits: int = 2048,
    cutoff: float = 0.4,
    linkage_method: str = "ward",
    threshold_similarity: Optional[float] = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Full pipeline: SMILES → fingerprints → similarity matrix → clusters.
 
    Parameters
    ----------
    smiles_series        : pd.Series  – raw SMILES strings
    names_series         : pd.Series  – optional drug names / IDs (same index)
    threshold_similarity : float|None – zero-out similarities below this value
                                        before clustering (sparsification)
 
    Returns
    -------
    result_df  : pd.DataFrame  – columns: smiles, name (opt), cluster_id
    sim_matrix : np.ndarray    – N×N similarity matrix
    Z          : np.ndarray    – linkage matrix
    """
    smiles_list = smiles_series.dropna().tolist()

    valid_smiles, fps = smiles_to_fingerprints(smiles_list, radius=radius, n_bits=n_bits)
    print(f"[INFO] {len(valid_smiles)} / {len(smiles_list)} SMILES parsed successfully.")

    sim_matrix = compute_tanimoto_matrix(fps)

    if threshold_similarity is not None:
        sim_matrix[sim_matrix < threshold_similarity] = 0.0

    Z, clusters = cluster_by_similarity(sim_matrix, cutoff=cutoff,
                                        linkage_method=linkage_method)

    result_df = pd.DataFrame({"canonical_smiles": valid_smiles, "cluster_id": clusters})

    # attach names if provided
    if names_series is not None:
        name_map = dict(zip(smiles_series.dropna().tolist(),
                            names_series[smiles_series.notna()].tolist()))
        result_df["name"] = result_df["canonical_smiles"].map(name_map)

    n_clusters = result_df["cluster_id"].nunique()
    print(f"[INFO] {n_clusters} clusters found at distance cutoff = {cutoff}.")
    return result_df, sim_matrix, Z

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Optional dendrogram plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_dendrogram(Z: np.ndarray, cutoff: float,
                    output_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=cutoff)
    ax.axhline(y=cutoff, color="red", linestyle="--", label=f"cutoff = {cutoff}")
    ax.set_title("Hierarchical clustering of drug SMILES (Tanimoto distance)")
    ax.set_ylabel("Distance (1 − Tanimoto)")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[INFO] Dendrogram saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    
    lincs_sigcom = LINCSpaths(
        gctx=str(MNT_L1000 / 'cp_coeff_mat.gctx'),
        pathway=str(INPUT / 'gsea_l1000.parquet'),
        sig_info= str( MNT_L1000 / 'siginfo_beta.txt'),
        gene_info=  str( MNT_L1000 / 'geneinfo_beta.txt.gz'),
        cell_info=  str( MNT_L1000 / 'cellinfo_beta.txt.gz'),
        compound_info= str (MNT_L1000/ 'compoundinfo_beta.txt.gz'),
        inst_info= str (MNT_L1000 / 'instinfo_beta.txt.gz')
    )
    ccle_data = CCLEpaths(
        transcriptomics= str(INPUT / 'CCLE_RNAseq_genes_rpkm_20180929.gct'),
        metabolomics= str(INPUT / 'CCLE_metabolomics_20190502.csv'),
        cell_annotations= str(INPUT / 'Cell_lines_annotations_20181226.txt'),
        metabo_mapping= str(INPUT / 'metabo_mapping.csv'),
        depmap_annotation = str (INPUT / 'Depmap_annotation.csv')
    )
    
    loader = Loader(lincs_paths=lincs_sigcom,ccle_paths=ccle_data)
    loader.load_l1000_metadata()
    
    compound_info = loader.compound_info
    compound_info = compound_info.dropna(subset='canonical_smiles')
    print(compound_info.shape)

    result_df, sim_matrix, Z = smiles_to_cluster_df(
        smiles_series=compound_info["canonical_smiles"],
        names_series=compound_info["cmap_name"],
        radius=2,
        n_bits=2048,
        cutoff=0.6,          # tune this: lower → fewer, bigger clusters
        linkage_method="average",
        threshold_similarity=None,
    )

    print(result_df.sort_values("cluster_id"))
    result_df.to_csv(OUTPUT / "drug_clusters.csv", index=False)

    plot_dendrogram(Z, cutoff=0.6, output_path=OUTPUT / "dendrogram.pdf")