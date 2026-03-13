"""
This script is the main script to perform pathway-level prediction 
of drug responses
"""
import pandas as pd
from pathlib import Path
from typing import cast
from pathways import Pathways
from loader import LINCSpaths
from loader import CCLEpaths
from loader import Loader
from viz import plot_metabo_histogram
from viz import plot_pca
from analysis import run_pca
from analysis import run_gsva

# Paths
INPUT = Path('/home/gdelrot/pathway_perturb/data')
OUTPUT = Path('/home/gdelrot/pathway_perturb/outputs')
MNT_L1000 = Path('/mnt/cbib/l1000/data/')

def main()->None:

    # Init the classes
    lincs_sigcom = LINCSpaths(
        gctx=str(MNT_L1000 / 'cp_coeff_mat.gctx'),
        pathway=str(MNT_L1000 / 'gsea_l1000.parquet'),
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
        metabo_mapping= str(INPUT / 'metabo_mapping.csv')
    )

    loader = Loader(lincs_paths=lincs_sigcom,ccle_paths=ccle_data)
    loader.load_ccle_data()

    # Load pathways
    pathways = Pathways()
    pathways.load_gmt(path= str(INPUT / 'KEGG_hsa_pathways_compounds_R117.gmt'),
                    omics = 'metabolomics')

    # --- Metabolomics preprocessing ------------------------------------------
    # Convert metabolite ids
    loader.preprocess_metabolomics(remove_lipids=True,
                                   convert_ids=True)

    metabo_proc = loader.ccle_metabolomics
    plot_metabo_histogram(metabo_proc,
                          title='Distribution of metabolomics values',
                          figname=str(OUTPUT/'metabo_values.png'))
    # --- Check PCAs
    HUE_COLS = ['Pathology','Site_Primary','Histology']
    met_pca_scores, met_pca = run_pca(matrix=metabo_proc)

    for hue in HUE_COLS:
        plot_pca(pca_plot=met_pca_scores.join(loader.ccle_annotation[hue]),
                explained_var=met_pca.explained_variance_ratio_,
                hue_col=hue,
                out_path=str(OUTPUT/f'metabolites_PCA_nolipids_{hue}.png'),
                title=f'PCA scatter plots of metabolites colored by : {hue}')

    # --- Checks PCAs of metabo pathway scores for different thresholds
    for met_threshold in [2,3,4,5,6]:
        print('\n','-'*80)
        print(f'\n Working with threshold: {met_threshold}')
        gsva_scores = run_gsva(
            omics_matrix=metabo_proc,
            pathways_dict=pathways.kegg_metabolomics.pathways_dict, #type:ignore
            min_size=met_threshold,
            max_size = 1000
        )
        pathways_kept = gsva_scores.shape[1]
        print(f'\n {pathways_kept} pathways have been kept')
        gsva_pca_scores, gsva_pca = run_pca(matrix=gsva_scores)

        for hue in HUE_COLS:
            plot_pca(pca_plot=gsva_pca_scores.join(loader.ccle_annotation[hue]),
                    explained_var=gsva_pca.explained_variance_ratio_,
                    hue_col=hue,
                    out_path=str(OUTPUT/f'gsva_PCA_threshold_{met_threshold}_{hue}.png'),
                    title=f'PCA scatter plots of gsva scores for threshold: {met_threshold} based on {pathways_kept} pathways')


if __name__ == '__main__':

    main()
