"""
This script is the main script to perform pathway-level prediction 
of drug responses
"""
from pathlib import Path
import pandas as pd

from pathways import Pathways
from loader import LINCSpaths
from loader import CCLEpaths
from loader import Loader
import gseapy as gp

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
        cell_annotations= str(INPUT / 'Cell_lines_annotations_20181226.txt')
    )
    
    loader = Loader(lincs_paths=lincs_sigcom,ccle_paths=ccle_data)
    loader.load_ccle_data()

    # Load pathways
    pathways = Pathways()
    pathways.load_gmt(path= str(INPUT / 'KEGG_hsa_pathways_compounds_R117.gmt'),
                    omics = 'metabolomics')

    print(pathways.kegg_metabolomics.gmt)
    print(pathways.kegg_metabolomics.gmt.shape)
    print(pathways.kegg_metabolomics.gmt.columns)

    compound_names = loader.ccle_metabolomics.columns.to_list()
    conversion_table = pathways.convert_metabolite_ids(input_type='name',
                                                       compound_list=compound_names)
    
    print(conversion_table)
    
if __name__ == '__main__':

    main()
