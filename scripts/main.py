"""
This script is the main script to perform pathway-level prediction 
of drug responses
"""
from pathlib import Path
from pathways import Pathways
from loader import LINCSpaths
from loader import CCLEpaths
from loader import Loader
from id_harmonizer import harmonize_ids
from methods_ML import run_full_pipeline, per_pathway_summary, feature_importance_df

# Paths
INPUT = Path('/home/gdelrot/pathway_perturb/data')
OUTPUT = Path('/home/gdelrot/pathway_perturb/outputs')
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
    loader.load_ccle_data()

    # Load l1000
    loader.load_l1000_pathway_scores()
    loader.load_l1000_metadata()

    # Load pathways
    # pathways = Pathways()
    # pathways.load_gmt(path= str(INPUT / 'KEGG_hsa_pathways_compounds_R117.gmt'),
    #                   omics = 'metabolomics')
    # pathways.load_gmt(path=str(INPUT / 'KEGG_hsa_pathways_transcriptomics_R117.gmt'),
    #                   omics = 'transcriptomics')

    # ===== METABOLOMICS ANALYSIS =====
    out_metabo = OUTPUT / 'out_metabo'
    if not out_metabo.is_dir():
        out_metabo.mkdir()

    out_transcripto = OUTPUT / 'out_rna'
    if not out_transcripto.is_dir():
        out_transcripto.mkdir()

    data_proc, _ = loader.preprocess_metabolomics(split_lipids=False,
                                                   convert_ids=True,
                                                   remove_unmapped=True)

    # Convert Ensembl IDs to gene symbols
    transcripts = loader.ccle_transcriptomics.columns.to_list()
    loader.ccle_transcriptomics.columns = [rna.split('.')[0] for rna in transcripts]

    # conversion = pathways.convert_gene_ids(
    #     input_ids=list(loader.ccle_transcriptomics.columns),
    #     source='ensembl.gene',
    #     target='symbol'
    # )
    # mapped = [sym for sym, eid in conversion.items() if eid is not None]
    # loader.ccle_transcriptomics = loader.ccle_transcriptomics[mapped]
    # loader.ccle_transcriptomics.columns = [conversion[sym] for sym in mapped]
    # print(f'\n Shape of annotated genes: {loader.ccle_transcriptomics.shape}')

    # # ===== PATHWAY INTERSECTIONS =====
    # common_pathways = pathways.pathway_intersections(
    #     rna_pathways=pathways.kegg_transcriptomics.pathways_dict,
    #     metabo_pathways=pathways.kegg_metabolomics.pathways_dict,
    #     metabolite_ms=data_proc.columns.to_list(),
    #     rna_ms=loader.ccle_transcriptomics.columns.to_list(),
    #     l1000_pathways=loader.l1000_pathway_data.columns.to_list(),
    #     metabo_thresholds=[2, 3, 4, 5]
    # )

    # ===== ID HARMONIZATION =====
    # Maps everything to Cellosaurus (CVCL_XXXX) as common identifier.
    # CCLE index ("DMS53_LUNG") → CVCL via DepMap RRID
    # L1000 sig_id → cell_iname via sig_info → CVCL via cell_info

    trans_h, metab_h, l1000_h, id_stats = harmonize_ids(
        ccle_transcriptomics=loader.ccle_transcriptomics,
        ccle_metabolomics=data_proc,
        l1000_pathway_data=loader.l1000_pathway_data,
        sig_info=loader.sig_info,          # ← adapt attribute name if different
        cell_info=loader.cell_info,        # ← adapt attribute name if different
        depmap_annotation=loader.depmap_annotation,
        ccle_annotation=loader.ccle_annotation,
    )

    # ===== ML PIPELINE =====
    # Now trans_h, metab_h are indexed by CVCL_XXXX
    # and l1000_h has 'cell_id' (=CVCL) and 'drug_id' columns.

    out_ml = OUTPUT / 'out_ml'
    if not out_ml.is_dir():
        out_ml.mkdir()

    for strategy in ['random', 'cell_line', 'drug']:
        results, data, summary = run_full_pipeline(
            l1000_df=l1000_h,
            ccle_transcriptomics=trans_h,
            ccle_metabolomics=metab_h,
            cell_col='cell_id',
            drug_col='drug_id',
            split_strategy=strategy,
            test_size=0.2,
        )
        summary.to_csv(out_ml / f'summary_{strategy}.csv', index=False)

    # Per-pathway R² and feature importances (from last run = drug holdout,
    # rerun random for diagnostics)
    results_rand, data_rand, _ = run_full_pipeline(
        l1000_df=l1000_h,
        ccle_transcriptomics=trans_h,
        ccle_metabolomics=metab_h,
        cell_col='cell_id',
        drug_col='drug_id',
        split_strategy='random',
        test_size=0.2,
    )
    per_pathway_summary(results_rand).to_csv(out_ml / 'per_pathway_r2.csv')

    fi = feature_importance_df(results_rand['RandomForest'], data_rand.feature_names, top_n=30)
    if fi is not None:
        fi.to_csv(out_ml / 'feature_importances_rf.csv', index=False)


if __name__ == '__main__':
    main()