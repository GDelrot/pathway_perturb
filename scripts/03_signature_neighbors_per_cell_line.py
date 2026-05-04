"""
Script to compute KNN on L1000 perturbations
"""
import sys
import gc
import time 
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import silhouette_score
from __loader__ import LINCSpaths, CCLEpaths, Loader
from __analysis__ import annotate_cell_info,annotate_drug_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

INPUT    = Path('/home/gdelrot/pathway_perturb/data')
OUTPUT   = Path('/home/gdelrot/pathway_perturb/outputs/parallele_KNN')
MNT_L1000 = Path('/mnt/cbib/l1000/data/')

OUTPUT.mkdir(parents=True, exist_ok=True)

def build_loader() -> Loader:
    """Build and return a Loader with L1000 metadata and pathway scores loaded.

    Configures paths for both L1000 (LINCS) and CCLE datasets, loads L1000
    metadata and GSEA-based pathway scores, drops columns with NaN values,
    and downcasts float64 columns to float32 to reduce memory usage.

    Returns
    -------
    Loader
        Loader instance with `l1000_pathway_data` ready for downstream analysis.
    """
    lincs_sigcom = LINCSpaths(
        gctx=str(MNT_L1000 / 'cp_coeff_mat.gctx'),
        pathway=str(INPUT / 'gsea_l1000.parquet'),
        sig_info=str(MNT_L1000 / 'siginfo_beta.txt'),
        gene_info=str(MNT_L1000 / 'geneinfo_beta.txt.gz'),
        cell_info=str(MNT_L1000 / 'cellinfo_beta.txt.gz'),
        compound_info=str(MNT_L1000 / 'compoundinfo_beta.txt.gz'),
        inst_info=str(MNT_L1000 / 'instinfo_beta.txt.gz'),
    )
    ccle_data = CCLEpaths(
        transcriptomics=str(INPUT / 'CCLE_RNAseq_genes_rpkm_20180929.gct'),
        metabolomics=str(INPUT / 'CCLE_metabolomics_20190502.csv'),
        cell_annotations=str(INPUT / 'Cell_lines_annotations_20181226.txt'),
        metabo_mapping=str(INPUT / 'metabo_mapping.csv'),
        depmap_annotation=str(INPUT / 'Depmap_annotation.csv'),
    )
    loader = Loader(lincs_paths=lincs_sigcom, ccle_paths=ccle_data)
    loader.load_l1000_metadata()

    # Gene level signatures
    loader.extract_data_subset(n_subset=None)
    loader.l1000_exp_data = loader.l1000_exp_data.dropna(axis=1) # Already float32

    # Pathway level signatures
    loader.load_l1000_pathway_scores()
    loader.l1000_pathway_data = loader.l1000_pathway_data.dropna(axis=1)
    float_cols = loader.l1000_pathway_data.select_dtypes('float64').columns
    loader.l1000_pathway_data[float_cols] = (
        loader.l1000_pathway_data[float_cols].astype('float32')
    )
    return loader

def run_KNN_clustering(perturbation_data:pd.DataFrame,
                       n_neighbors:int=5,
                       algorithm:str='auto'):
    """Fit a nearest-neighbour model on standardised perturbation pathway scores.

    Scales `perturbation_data` with `StandardScaler`, then fits a
    `NearestNeighbors` model and retrieves the `n_neighbors` closest samples
    for every point (self inclusive).

    Parameters
    ----------
    perturbation_data : pd.DataFrame
        Samples × pathway-score feature matrix.
    n_neighbors : int, optional
        Number of neighbours to retrieve (default 5).

    Returns
    -------
    dist : np.ndarray, shape (n_samples, n_neighbors)
        Euclidean distances to each neighbour.
    indices : np.ndarray, shape (n_samples, n_neighbors)
        Row indices of each neighbour in `perturbation_data`.
    """
    start_time = time.time()
    logger.info('Beginning KNN clustering at %s... \n',start_time)
    logger.info('Shape of the perturbation dataset of perturb: %s',perturbation_data.shape)

    # Scale the data
    scaler = StandardScaler()
    scaled_perturbations = scaler.fit_transform(perturbation_data)
    
    # KNN
    neigh = NearestNeighbors(n_neighbors=n_neighbors,algorithm=algorithm)
    neigh.fit(scaled_perturbations)
    
    # Retrieve distances and indices
    dist, indices = neigh.kneighbors(scaled_perturbations,
                                    n_neighbors=n_neighbors)
    logger.info('\n Clustering took %s',
                time.time() - start_time)
    return dist, indices

def run_neighbors_statistics(distances: np.ndarray,
                             indices: np.ndarray,
                             annotations: pd.DataFrame,
                             metadata: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute neighbourhood label purity for every observation.

    For each sample, measures the fraction of its k nearest neighbours
    (self excluded) that share the same label as the query point.

    Parameters
    ----------
    distances : np.ndarray, shape (n_samples, n_neighbors)
        Euclidean distances returned by NearestNeighbors.kneighbors().
    indices : np.ndarray, shape (n_samples, n_neighbors)
        Row-indices of each neighbour (col 0 = self).
    annotations : pd.DataFrame
        Must contain a column named `metadata` aligned with perturbation_data rows.
    metadata : str
        Column name used as label (e.g. 'moa', 'cluster_id').

    Returns
    -------
    sample_stats : pd.DataFrame
        Original annotations + 'nn_purity' column (float in [0, 1]).
    class_stats : pd.DataFrame
        Per-class mean, std and count of nn_purity, plus random baseline.
    """
    labels = annotations[metadata].to_numpy(dtype=str, na_value='unknown')         # (n_samples,)  ← 1-D label array

    # --- skip self (column 0 is always the query point itself) ---
    neighbor_idx = indices[:, 1:]                  # (n_samples, k-1)

    # --- vectorised label lookup ---
    neighbor_labels = labels[neighbor_idx]         # (n_samples, k-1)

    # --- broadcasting comparison ---
    # labels[:, None] → (n_samples, 1)  broadcasts against (n_samples, k-1)
    same_label_mask:np.ndarray = neighbor_labels == labels[:, None]   # bool (n_samples, k-1)

    # fraction of neighbours sharing the label, per sample
    purity_per_sample = same_label_mask.mean(axis=1)       # (n_samples,)

    # --- build result frames ---
    sample_stats = annotations.copy()
    sample_stats['nn_purity'] = purity_per_sample

    class_stats = (
        sample_stats
        .groupby(metadata)['nn_purity']
        .agg(mean='mean', std='std', count='count')
        .sort_values('mean', ascending=False)
    ) # type: ignore

    # random baseline: if labels were shuffled, expected purity = 1 / n_classes
    n_classes   = annotations[metadata].nunique()
    baseline    = 1.0 / n_classes

    class_stats['random_baseline'] = baseline
    class_stats['lift']            = class_stats['mean'] / baseline   # how many × better than chance

    # --- logging ---
    logger.info(
        '\n[%s] Global NN purity : %.3f  |  Random baseline : %.3f  |  Lift : %.2fx',
        metadata, purity_per_sample.mean(), baseline,
        purity_per_sample.mean() / baseline
    )
    logger.info('\nPer-class purity :\n%s', class_stats.to_string())

    return sample_stats, class_stats

def filter_cell_line(loader_obj: Loader,
                     l1000_data: pd.DataFrame,
                     cell_line: str) -> pd.DataFrame:
    """Subset L1000 perturbation data to a single cell line.

    Joins `cell_iname` from the loader's cell metadata onto `l1000_data`
    (by sig_id index) and keeps only rows matching `cell_line`.

    Parameters
    ----------
    loader_obj : Loader
        Loader instance whose `cell_info` table has a 'sig_id' column and
        a 'cell_iname' column.
    l1000_data : pd.DataFrame
        Perturbation feature matrix indexed by sig_id.
    cell_line : str
        Cell line name to retain (e.g. 'MCF7').

    Returns
    -------
    pd.DataFrame
        Subset of `l1000_data` rows belonging to `cell_line`, with the
        'cell_iname' column appended.
    """
    cell_info = loader_obj.cell_info.copy()
    sig_info = loader_obj.sig_info.copy()

    cell_annotations = pd.merge(sig_info,cell_info,on='cell_iname')
    cell_annotations = cell_annotations.dropna(subset='cell_iname')
    cell_annotations = cell_annotations.set_index('sig_id')['cell_iname']
    l1000_annotated = l1000_data.join(cell_annotations, how='left')
    return l1000_annotated[l1000_annotated['cell_iname'] == cell_line].drop(labels='cell_iname')

def main():
    
    logger.info('Loading datasets...')
    loader = build_loader()
    logger.info('Processing cell line: %s',
                sys.argv[1])
    for pert_type in ['raw_pert','pathway_pert']:
        if pert_type == 'raw_pert':
            pert_data = loader.l1000_exp_data
        elif pert_type == 'pathway_pert':
            pert_data = loader.l1000_pathway_data

        else:
            raise ValueError('Expecting one of raw_pert/pathway_pert for pert_type')

        # annotate dataset
        for metadata in ['moa','cluster_id']:
            logger.info('\n Starting annotation of cells and compounds: \n')
            logger.info('Initial size: %s',pert_data.shape[0])

            l1000_drug_annotated = annotate_drug_info(
                perturbations=pert_data,
                l1000_obj=loader,
                drug=metadata
                )
            pair_annotated = annotate_cell_info(
                l1000_drug_ann=l1000_drug_annotated,
                filter_cell_line=sys.argv[1],
                l1000_obj=loader,
            )
            # Subset for trying the code
            annotations = pair_annotated[['cell_iname',metadata]]
            pair_annotated = pair_annotated.drop(labels=['cell_iname',metadata],axis=1)
            logger.info('\n %s samples with annotation for %s and cell lines',
                        pair_annotated.shape[0],metadata)

            # Run KNN
            dist, indices = run_KNN_clustering(perturbation_data=pair_annotated,
                            n_neighbors=5,
                            algorithm='auto')
            
            logger.info('\n Computed distance with shape: %s, of form: \n %s \n'
                        'Computed indices with shape %s, of form %s',
                        dist.shape,dist[0:5,0:5],indices.shape,indices[0:5])
            
            sample_stats, class_stats = run_neighbors_statistics(
                distances=dist,
                indices=indices,
                annotations=annotations,
                metadata=metadata
            )

            # persist
            sample_stats.to_parquet(OUTPUT / f'nn_purity_samples_{sys.argv[1]}_{pert_type}_{metadata}.parquet')
            class_stats.to_csv(OUTPUT   / f'nn_purity_classes_{sys.argv[1]}_{pert_type}_{metadata}.csv')

    
if __name__ == '__main__':
    main()

    