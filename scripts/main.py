"""
This script is the main script to perform pathway-level prediction 
of drug responses
"""
from pathlib import Path
from pathways import Pathways
from __loader__ import LINCSpaths
from __loader__ import CCLEpaths
from __loader__ import Loader
from id_harmonizer import harmonize_ids
from methods_ML import run_full_pipeline, per_pathway_summary, feature_importance_df
from __analysis__ import run_gsva

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
    pathways = Pathways()
    pathways.load_gmt(path= str(INPUT / 'KEGG_hsa_pathways_compounds_R117.gmt'),
                      omics = 'metabolomics')
    pathways.load_gmt(path=str(INPUT / 'KEGG_hsa_pathways_transcriptomics_R117.gmt'),
                      omics = 'transcriptomics')

    metabo_proc, _ = loader.preprocess_metabolomics(split_lipids=False,
                                                   convert_ids=True,
                                                   remove_unmapped=True)

    # Convert Ensembl IDs to gene symbols
    transcripts = loader.ccle_transcriptomics.columns.to_list()
    loader.ccle_transcriptomics.columns = [rna.split('.')[0] for rna in transcripts]

    conversion = pathways.convert_gene_ids(
        input_ids=list(loader.ccle_transcriptomics.columns),
        source='ensembl.gene',
        target='symbol'
    )
    mapped = [sym for sym, eid in conversion.items() if eid is not None]
    loader.ccle_transcriptomics = loader.ccle_transcriptomics[mapped]
    loader.ccle_transcriptomics.columns = [conversion[sym] for sym in mapped]
    print(f'\n Shape of annotated genes: {loader.ccle_transcriptomics.shape}')

    # # ===== PATHWAY INTERSECTIONS =====
    common_pathways = pathways.pathway_intersections(
        rna_pathways=pathways.kegg_transcriptomics.pathways_dict,
        metabo_pathways=pathways.kegg_metabolomics.pathways_dict,
        metabolite_ms=metabo_proc.columns.to_list(),
        rna_ms=loader.ccle_transcriptomics.columns.to_list(),
        l1000_pathways=loader.l1000_pathway_data.columns.to_list(),
        metabo_thresholds=[3]
    )

    # Run GSVA on CCLE datasets
    gsva_metabo = run_gsva(
            omics_matrix=metabo_proc,
            pathways_dict=pathways.kegg_metabolomics.pathways_dict,
            min_size=3,
            max_size=1000,
            omics='metabolomics'
        )

    gsva_rna = run_gsva(
            omics_matrix=loader.ccle_transcriptomics,
            pathways_dict=pathways.kegg_transcriptomics.pathways_dict,
            min_size=15,
            max_size=1000,
            omics='transcriptomics'
        )

    # Filter data
    gold_samples = loader.tas_filtering(tas_threshold=0.2)
    l1000_filtered = loader.l1000_pathway_data.loc[gold_samples].copy()
    l1000_filtered = l1000_filtered.dropna(axis=1)

    # ===== ID HARMONIZATION =====
    trans_h, metab_h, l1000_h, id_stats = harmonize_ids(
        ccle_transcriptomics=gsva_rna,
        ccle_metabolomics=gsva_metabo,
        l1000_pathway_data=l1000_filtered,
        sig_info=loader.sig_info,
        cell_info=loader.cell_info,
        depmap_annotation=loader.depmap_annotation,
        ccle_annotation=loader.ccle_annotation,
    )

    trans_h.to_csv(INPUT / 'harmonized_transcriptomics.csv')
    metab_h.to_csv(INPUT / 'harmonized_metabolomics.csv')
    l1000_h.to_csv(INPUT / 'harmonized_l1000.csv')

    # ML pipeline
    from drug_encoding import compute_morgan_fingerprints
    
    drug_encoder = compute_morgan_fingerprints(
        compound_info=loader.compound_info,  # from your Loader class
        drug_id_col="pert_id",               # adjust to match your drug_col
        smiles_col="canonical_smiles",       # adjust to match your compound_info columns
        radius=2,                            # ECFP4
        n_bits=2048,
    )
    
    # ── Run pipeline with fingerprints ──────────────────────────────────────
    out_tas = OUTPUT / 'out_ml_fingerprint'
    out_tas.mkdir(exist_ok=True)
    
    for strategy in ['random', 'cell_line', 'drug']:
        results, data, summary = run_full_pipeline(
            l1000_df=l1000_h,
            ccle_transcriptomics=trans_h,
            ccle_metabolomics=metab_h,
            cell_col='cell_id',
            drug_col='drug_id',
            split_strategy=strategy,
            test_size=0.2,
            # ── NEW ─────────────────────────
            drug_encoding='fingerprint',
            drug_encoder=drug_encoder,      # pre-computed, not recomputed each loop
        )
        summary.to_csv(out_tas / f'summary_{strategy}_fp.csv', index=False)
    
    
    # ── Option B: compare encodings systematically ──────────────────────────
    from drug_encoding import compute_descriptors, compute_hybrid_encoding
    
    desc_encoder = compute_descriptors(
        compound_info=loader.compound_info,
        drug_id_col="pert_id",
        smiles_col="canonical_smiles",
    )
    
    hybrid_encoder = compute_hybrid_encoding(
        compound_info=loader.compound_info,
        drug_id_col="pert_id",
        smiles_col="canonical_smiles",
    )
    
    encoding_configs = {
        'onehot':      {'drug_encoding': 'onehot',      'drug_encoder': None},
        'fingerprint': {'drug_encoding': 'fingerprint',  'drug_encoder': drug_encoder},
        'descriptor':  {'drug_encoding': 'descriptor',   'drug_encoder': desc_encoder},
        'hybrid':      {'drug_encoding': 'hybrid',        'drug_encoder': hybrid_encoder},
    }
    
    comparison_rows = []
    for enc_name, enc_kwargs in encoding_configs.items():
        for strategy in ['random', 'cell_line', 'drug']:
            results, data, summary = run_full_pipeline(
                l1000_df=l1000_h,
                ccle_transcriptomics=trans_h,
                ccle_metabolomics=metab_h,
                cell_col='cell_id',
                drug_col='drug_id',
                split_strategy=strategy,
                test_size=0.2,
                **enc_kwargs,
            )
            for _, row in summary.iterrows():
                comparison_rows.append({
                    'encoding': enc_name,
                    'split': strategy,
                    'model': row['Model'],
                    'R2': row['R²'],
                    'MSE': row['MSE'],
                })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(out_tas / 'encoding_comparison.csv', index=False)


if __name__ == '__main__':
    main()
