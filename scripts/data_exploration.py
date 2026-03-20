"""
This script is the main script to perform pathway-level prediction 
of drug responses
"""
from pathlib import Path
from typing import cast, Dict, List
from pathways import Pathways
from loader import LINCSpaths
from loader import CCLEpaths
from loader import Loader
from viz import plot_metabo_histogram, plot_pca, plot_umap, plot_pca_centroids
from analysis import run_pca, run_gsva, run_umap

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
    loader.load_l1000_pathway_scores()
    # Load pathways
    pathways = Pathways()
    pathways.load_gmt(path= str(INPUT / 'KEGG_hsa_pathways_compounds_R117.gmt'),
                    omics = 'metabolomics')
    pathways.load_gmt(path=str(INPUT / 'KEGG_hsa_pathways_transcriptomics_R117.gmt'),
                    omics = 'transcriptomics')

        # ===== METABOLOMICS ANALYSIS =====

    out_metabo = OUTPUT / 'out_metabo'
    if not out_metabo.is_dir():
        out_metabo.mkdir()

    data_proc, _ = loader.preprocess_metabolomics(split_lipids=False,
                                   convert_ids=True,
                                   remove_unmapped=True)

    hue_cols = ['Pathology','Site_Primary','Histology']

    # # ===== FULL DATA (WITH LIPIDS) =====
    # # PCA on full data
    # met_pca_scores, met_pca = run_pca(matrix=data_proc)
    # for hue in hue_cols:
    #     plot_pca(pca_plot=met_pca_scores.join(loader.ccle_annotation[hue]),
    #             explained_var=met_pca.explained_variance_ratio_,
    #             hue_col=hue,
    #             out_path=str(out_metabo/f'metabolites_PCA_whole_{hue}.png'),
    #             title=f'PCA scatter plots of metabolites colored by : {hue}')

    #     # Complementary: group-level summary with centroids
    #     plot_pca_centroids(pca_df=met_pca_scores.join(loader.ccle_annotation[hue]),
    #                       explained_var=met_pca.explained_variance_ratio_,
    #                       hue_col=hue,
    #                       out_path=str(out_metabo/f'metabolites_PCA_whole_centroids_{hue}.png'),
    #                       title=f'PCA Centroids with 1-std ellipses of metabolites by : {hue}')

    # # UMAP on full data
    # met_umap_scores, met_umap = run_umap(matrix=data_proc)
    # for hue in hue_cols:
    #     plot_umap(umap_plot=met_umap_scores.join(loader.ccle_annotation[hue]),
    #             hue_col=hue,
    #             out_path=str(out_metabo/f'metabolites_UMAP_whole_{hue}.png'),
    #             title=f'UMAP scatter plots of metabolites colored by : {hue}')

    # # Split lipids for separate analysis
    # metabo_proc, lipid_proc = loader.preprocess_metabolomics(split_lipids=True,
    #                                convert_ids=True,
    #                                remove_unmapped=True)

    # plot_metabo_histogram(metabo_proc,
    #                       title='Distribution of metabolomics values',
    #                       figname=str(out_metabo/'metabo_values.png'))

    # plot_metabo_histogram(lipid_proc,
    #                       title='Distribution of lipid values',
    #                       figname=str(out_metabo/'lipid_values.png'))

    # # ===== METABOLITES ONLY (NO LIPIDS) =====
    # # PCA on metabolites
    # met_pca_scores, met_pca = run_pca(matrix=metabo_proc)
    # for hue in hue_cols:
    #     plot_pca(pca_plot=met_pca_scores.join(loader.ccle_annotation[hue]),
    #             explained_var=met_pca.explained_variance_ratio_,
    #             hue_col=hue,
    #             out_path=str(out_metabo/f'metabolites_PCA_nolipids_{hue}.png'),
    #             title=f'PCA scatter plots of {metabo_proc.shape[1]} metabolites colored by : {hue}')

    #     # Complementary: group-level summary with centroids
    #     plot_pca_centroids(pca_df=met_pca_scores.join(loader.ccle_annotation[hue]),
    #                       explained_var=met_pca.explained_variance_ratio_,
    #                       hue_col=hue,
    #                       out_path=str(out_metabo/f'metabolites_PCA_nolipids_centroids_{hue}.png'),
    #                       title=f'PCA Centroids with 1-std ellipses of {metabo_proc.shape[1]} metabolites by : {hue}')

    # # UMAP on metabolites
    # met_umap_scores, met_umap = run_umap(matrix=metabo_proc)
    # for hue in hue_cols:
    #     plot_umap(umap_plot=met_umap_scores.join(loader.ccle_annotation[hue]),
    #             hue_col=hue,
    #             out_path=str(out_metabo/f'metabolites_UMAP_nolipids_{hue}.png'),
    #             title=f'UMAP scatter plots of {metabo_proc.shape[1]} metabolites colored by : {hue}')

    # # ===== LIPIDS ONLY =====
    # # PCA on lipids
    # lipid_pca_scores, lipid_pca = run_pca(matrix=lipid_proc)
    # for hue in hue_cols:
    #     plot_pca(pca_plot=lipid_pca_scores.join(loader.ccle_annotation[hue]),
    #             explained_var=lipid_pca.explained_variance_ratio_,
    #             hue_col=hue,
    #             out_path=str(out_metabo/f'metabolites_PCA_only_lipids_{hue}.png'),
    #             title=f'PCA scatter plots of {lipid_proc.shape[1]} lipids colored by : {hue}')

    #     # Complementary: group-level summary with centroids
    #     plot_pca_centroids(pca_df=lipid_pca_scores.join(loader.ccle_annotation[hue]),
    #                       explained_var=lipid_pca.explained_variance_ratio_,
    #                       hue_col=hue,
    #                       out_path=str(out_metabo/f'metabolites_PCA_only_lipids_centroids_{hue}.png'),
    #                       title=f'PCA Centroids with 1-std ellipses of {lipid_proc.shape[1]} lipids by : {hue}')

    # # UMAP on lipids - ⚠️ FIXED: was "met_umap_scores" should be "lipid_umap_scores"
    # lipid_umap_scores, lipid_umap = run_umap(matrix=lipid_proc)
    # for hue in hue_cols:
    #     plot_umap(umap_plot=lipid_umap_scores.join(loader.ccle_annotation[hue]),
    #             hue_col=hue,
    #             out_path=str(out_metabo/f'metabolites_UMAP_only_lipids_{hue}.png'),
    #             title=f'UMAP scatter plots of {lipid_proc.shape[1]} lipids colored by : {hue}')

    # # ===== GSVA PATHWAY SCORES WITH VARIABLE THRESHOLDS =====
    # for met_threshold in [2, 3, 4, 5, 6]:
    #     print('\n' + '-'*80)
    #     print(f'Working with metabolomics threshold: {met_threshold}')

    #     gsva_scores = run_gsva(
    #         omics_matrix=metabo_proc,
    #         pathways_dict=pathways.kegg_metabolomics.pathways_dict,
    #         min_size=met_threshold,
    #         max_size=1000,
    #         omics='metabolomics'
    #     )
    #     pathways_kept = gsva_scores.shape[1]
    #     print(f'{pathways_kept} pathways kept with ≥{met_threshold} metabolites\n')

    #     # PCA on GSVA scores
    #     gsva_pca_scores, gsva_pca = run_pca(matrix=gsva_scores)
    #     for hue in hue_cols:
    #         plot_pca(pca_plot=gsva_pca_scores.join(loader.ccle_annotation[hue]),
    #                 explained_var=gsva_pca.explained_variance_ratio_,
    #                 hue_col=hue,
    #                 out_path=str(out_metabo/f'gsva_PCA_threshold_{met_threshold}_{hue}.png'),
    #                 title=f'PCA of GSVA scores (threshold={met_threshold}, {pathways_kept} pathways) by {hue}')

    #         # Complementary: group-level summary with centroids
    #         plot_pca_centroids(pca_df=gsva_pca_scores.join(loader.ccle_annotation[hue]),
    #                           explained_var=gsva_pca.explained_variance_ratio_,
    #                           hue_col=hue,
    #                           out_path=str(out_metabo/f'gsva_PCA_threshold_{met_threshold}_centroids_{hue}.png'),
    #                           title=f'PCA Centroids with 1-std ellipses of GSVA scores (threshold={met_threshold}, {pathways_kept} pathways) by {hue}')

    #     # UMAP on GSVA scores
    #     gsva_umap_scores, gsva_umap = run_umap(matrix=gsva_scores)
    #     for hue in hue_cols:
    #         plot_umap(umap_plot=gsva_umap_scores.join(loader.ccle_annotation[hue]),
    #                 hue_col=hue,
    #                 out_path=str(out_metabo/f'gsva_UMAP_threshold_{met_threshold}_{hue}.png'),
    #                 title=f'UMAP of GSVA scores (threshold={met_threshold}, {pathways_kept} pathways) by {hue}')

    ###########################################################################
    # --- Transcriptomics exploration -----------------------------------------
    ###########################################################################

    out_transcripto = OUTPUT / 'out_rna'
    if not out_transcripto.is_dir():
        out_transcripto.mkdir()

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

    # ===== PATHWAY INTERSECTIONS =====
    common_pathways = pathways.pathway_intersections(
        rna_pathways=pathways.kegg_transcriptomics.pathways_dict,
        metabo_pathways=pathways.kegg_metabolomics.pathways_dict,
        metabolite_ms=data_proc.columns.to_list(),
        rna_ms=loader.ccle_transcriptomics.columns.to_list(),
        l1000_pathways=loader.l1000_pathway_data.columns.to_list(),
        metabo_thresholds=[2, 3, 4, 5]
    )

    # Print with friendly names
    print("\n" + "="*60)
    print("PATHWAY INTERSECTIONS WITH NAMES")
    print("="*60)

    for threshold, data in common_pathways.items():
        print(f"\nThreshold: ≥{threshold} metabolites")
        print(f"Count: {data['count']} pathways\n")

        if data['pathways']:
            for pathway_id in sorted(data['pathways']):
                friendly_name = pathways.pathway_names.get(pathway_id, "Unknown")
                print(f"  • {pathway_id}: {friendly_name}")

        else:
            print("  (No intersecting pathways)")

    # # ===== RAW RNA DATA =====
    rna_rpkm = loader.ccle_transcriptomics

    # plot_metabo_histogram(
    #     rna_rpkm,
    #     title='Distribution of rna RPKM (raw from ccle) values',
    #     figname=str(out_transcripto/'rna_values.png')
    # )

    # # PCA on raw RNA
    # rna_pca_scores, rna_pca = run_pca(matrix=rna_rpkm)
    # for hue in hue_cols:
    #     plot_pca(
    #         pca_plot=rna_pca_scores.join(loader.ccle_annotation[hue]),
    #         explained_var=rna_pca.explained_variance_ratio_,
    #         hue_col=hue,
    #         out_path=str(out_transcripto/f'rna_PCA_{hue}.png'),
    #         title=f'PCA scatter plots of RNA colored by : {hue}'
    #     )

    #     # Complementary: group-level summary with centroids
    #     plot_pca_centroids(
    #         pca_df=rna_pca_scores.join(loader.ccle_annotation[hue]),
    #         explained_var=rna_pca.explained_variance_ratio_,
    #         hue_col=hue,
    #         out_path=str(out_transcripto/f'rna_PCA_centroids_{hue}.png'),
    #         title=f'PCA Centroids with 1-std ellipses of RNA by : {hue}'
    #     )

    # # UMAP on raw RNA - ✅ ADDED
    # rna_umap_scores, rna_umap = run_umap(matrix=rna_rpkm)
    # for hue in hue_cols:
    #     plot_umap(
    #         umap_plot=rna_umap_scores.join(loader.ccle_annotation[hue]),
    #         hue_col=hue,
    #         out_path=str(out_transcripto/f'rna_UMAP_{hue}.png'),
    #         title=f'UMAP scatter plots of RNA colored by : {hue}'
    #     )

    # # ===== GSVA PATHWAY SCORES WITH VARIABLE THRESHOLDS =====
    # for rna_threshold in [10, 15, 20]:
    #     print('\n' + '-'*80)
    #     print(f'Working with transcriptomics threshold: {rna_threshold}')

    #     gsva_scores = run_gsva(
    #         omics_matrix=rna_rpkm,
    #         pathways_dict=pathways.kegg_transcriptomics.pathways_dict,
    #         min_size=rna_threshold,
    #         max_size=1000,
    #         omics='transcriptomics'
    #     )
    #     pathways_kept = gsva_scores.shape[1]
    #     print(f'{pathways_kept} pathways kept with ≥{rna_threshold} genes\n')

    #     # PCA on GSVA scores
    #     gsva_pca_scores, gsva_pca = run_pca(matrix=gsva_scores)
    #     for hue in hue_cols:
    #         plot_pca(
    #             pca_plot=gsva_pca_scores.join(loader.ccle_annotation[hue]),
    #             explained_var=gsva_pca.explained_variance_ratio_,
    #             hue_col=hue,
    #             out_path=str(out_transcripto/f'gsva_PCA_threshold_{rna_threshold}_{hue}.png'),
    #             title=f'PCA of GSVA scores (threshold={rna_threshold}, {pathways_kept} pathways) by {hue}'
    #         )

    #         # Complementary: group-level summary with centroids
    #         plot_pca_centroids(
    #             pca_df=gsva_pca_scores.join(loader.ccle_annotation[hue]),
    #             explained_var=gsva_pca.explained_variance_ratio_,
    #             hue_col=hue,
    #             out_path=str(out_transcripto/f'gsva_PCA_threshold_{rna_threshold}_centroids_{hue}.png'),
    #             title=f'PCA Centroids with 1-std ellipses of GSVA scores (threshold={rna_threshold}, {pathways_kept} pathways) by {hue}'
    #         )

    #     # UMAP on GSVA scores - ✅ ADDED
    #     gsva_umap_scores, gsva_umap = run_umap(matrix=gsva_scores)
    #     for hue in hue_cols:
    #         plot_umap(
    #             umap_plot=gsva_umap_scores.join(loader.ccle_annotation[hue]),
    #             hue_col=hue,
    #             out_path=str(out_transcripto/f'gsva_UMAP_threshold_{rna_threshold}_{hue}.png'),
    #             title=f'UMAP of GSVA scores (threshold={rna_threshold}, {pathways_kept} pathways) by {hue}'
    #         )
        # Check PCAs onto different common pathways thresholds
    for threshold, data in common_pathways.items():
        print(f"\nThreshold: ≥{threshold} metabolites")
        print(f"Count: {data['count']} pathways\n")

        if data['pathways']:
            # Complementary: group-level summary with centroids
            gsva_scores = run_gsva(
                omics_matrix=rna_rpkm,
                pathways_dict=pathways.kegg_transcriptomics.pathways_dict,
                min_size=15,
                max_size=1000,
                omics='transcriptomics'
            )
            common_path = list(data['pathways'])
            gsva_scores = gsva_scores.loc[:,common_path]
            gsva_pca_scores, gsva_pca = run_pca(matrix=gsva_scores)
            for hue in hue_cols:
                plot_pca(
                    pca_plot=gsva_pca_scores.join(loader.ccle_annotation[hue]),
                    explained_var=gsva_pca.explained_variance_ratio_,
                    hue_col=hue,
                    out_path=str(out_transcripto/f'gsva_PCA_threshold_{threshold}_{hue}_common.png'),
                    title=f'PCA of GSVA scores (threshold={threshold}, {len(common_path)} pathways) by {hue}'
                )

                # Complementary: group-level summary with centroids
                plot_pca_centroids(
                    pca_df=gsva_pca_scores.join(loader.ccle_annotation[hue]),
                    explained_var=gsva_pca.explained_variance_ratio_,
                    hue_col=hue,
                    out_path=str(out_transcripto/f'gsva_PCA_threshold_{threshold}_centroids_{hue}_common.png'),
                    title=f'PCA Centroids with 1-std ellipses of GSVA scores (threshold={threshold}, {len(common_path) }pathways) by {hue}'
                )

            # UMAP on GSVA scores - ✅ ADDED
            gsva_umap_scores, gsva_umap = run_umap(matrix=gsva_scores)
            for hue in hue_cols:
                plot_umap(
                    umap_plot=gsva_umap_scores.join(loader.ccle_annotation[hue]),
                    hue_col=hue,
                    out_path=str(out_transcripto/f'gsva_UMAP_threshold_{threshold}_{hue}_common.png'),
                    title=f'UMAP of GSVA scores (threshold={threshold}, {len(common_path)} pathways) by {hue}'
                )

        else:
            print("  (No intersecting pathways)")

if __name__ == '__main__':

    main()