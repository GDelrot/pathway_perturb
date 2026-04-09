"""
This script is meant to explore the data as first intention analysis
"""
from pathlib import Path

from loader import LINCSpaths
from loader import CCLEpaths
from loader import Loader
from viz import plot_moa_correlations
from analysis import run_moa_signature_correlations
import pandas as pd
from drug_features import smiles_to_cluster_df

# Paths
INPUT = Path('/home/gdelrot/pathway_perturb/data')
OUTPUT = Path('/home/gdelrot/pathway_perturb/outputs/out_exploration_2')
MNT_L1000 = Path('/mnt/cbib/l1000/data/')

def main()->None:

    # Init the classes
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

    # Load CCLE metabo/ transcripto
    loader = Loader(lincs_paths=lincs_sigcom,ccle_paths=ccle_data)
    loader.load_l1000_metadata()
    loader.load_l1000_pathway_scores()

    loader.l1000_pathway_data = loader.l1000_pathway_data.dropna(axis=1)
    
    # --- Compute SMILES clusters and correlations
    result_df, _, Z = smiles_to_cluster_df(
        smiles_series=loader.compound_info["canonical_smiles"],
        names_series=loader.compound_info["cmap_name"],
        radius=2,
        n_bits=2048,
        cutoff=0.6,          # tune this: lower → fewer, bigger clusters
        linkage_method="average",
        threshold_similarity=None,
    )
    result_df = result_df[['canonical_smiles','cluster_id']]
    loader.compound_info = pd.merge(
        loader.compound_info, result_df, on = 'canonical_smiles', how = 'left'
        )
    print(loader.compound_info.head())
    print(loader.compound_info.columns)
    # Compute correlations by moa
    correlations_df, corr_dict = run_moa_signature_correlations(
        l1000_obj=loader,
        filter_cell_line= ['OVK18'],
        metadata='cluster_id',
        cor_type='pearson')
    correlations_df.to_csv(OUTPUT / 'correlations_smiles_pearson_MCF7.csv')
    plot_moa_correlations(cor_dict=corr_dict,
                        figname = str(OUTPUT / 'corplot_smiles_pearson.png'))

    # Compute correlations by moa
    correlations_df, corr_dict = run_moa_signature_correlations(
        l1000_obj=loader,
        filter_cell_line= ['OVK18'],
        metadata='moa',
        cor_type='pearson')

    correlations_df.to_csv(OUTPUT / 'correlations_moa_pearson_MCF7.csv')
    plot_moa_correlations(cor_dict=corr_dict,
                        figname = str(OUTPUT / 'corplot_pearson.png'))

    # correlations_df, corr_dict = run_moa_signature_correlations(
    #     l1000_obj=loader,
    #     filter_cell_line= ['MCF7'],
    #     metadata='moa',
    #     cor_type='spearman')

    # correlations_df.to_csv(OUTPUT / 'correlations_moa_spearman_MCF7.csv')
    # plot_moa_correlations(cor_dict=corr_dict,
    #                     figname = str(OUTPUT / 'corplot_spearman.png'))
if __name__ == '__main__':

    main()
