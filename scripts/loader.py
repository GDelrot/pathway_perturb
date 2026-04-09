"""
LINCS and CCLE Data Loader
==========================
Provides dataclasses for file path configuration and a Loader class that
handles ingestion of two complementary datasets:

  - **LINCS L1000** (via SigComLINCS): level-5 compound-perturbation expression
    data (.gctx), pre-computed GSEA pathway scores (.parquet), and associated
    metadata (sig_info, inst_info, gene_info, cell_info, compound_info).
    Filtering by Transcriptional Activity Score (TAS) and perturbation type
    restricts the working set to high-quality drug treatments only.

  - **CCLE**: RNA-seq transcriptomics (.gct), metabolomics (.csv), and cell-line
    annotations (.tsv).

Typical usage
-------------
    lincs = LINCSpaths(gctx=..., pathway=..., sig_info=..., inst_info=...)
    ccle  = CCLEpaths(transcriptomics=..., metabolomics=..., cell_annotations=...)

    loader = Loader(lincs, ccle)
    loader.load_l1000_metadata()
    valid_ids = loader.tas_filtering(tas_threshold=0.2)
    loader.extract_data_ids(valid_ids)
    loader.load_ccle_data()
"""
from dataclasses import dataclass
from typing import cast, Optional

import pandas as pd
from cmapPy.pandasGEXpress.parse_gct import parse as parse_gct
from cmapPy.pandasGEXpress.parse_gctx import parse

@dataclass
class LINCSpaths:
    """
    Paths to LINCS dataset files from SigComLINCS.

    Attributes
    ----------
    gctx : str
        Path to the Level 5 expression data file (.gctx).
    pathway : str
    Path to the GSEA scores derived from level 5 data data file (.gctx).
    sig_info : str
        Path to the signature metadata file (sig_info.txt).
    inst_info : str
        Path to the instance metadata file (inst_info.txt.gz).
    gene_info : str, optional
        Path to the gene annotation file (gene_info.txt.gz).
    cell_info : str, optional
        Path to the cell line annotation file (cell_info.txt.gz).
    compound_info : str, optional
        Path to the compound annotation file (compound_info.txt.gz).
    """
    gctx: str
    pathway: str
    sig_info: str
    inst_info: str
    gene_info: Optional[str] = None
    cell_info: Optional[str] = None
    compound_info: Optional[str] = None

@dataclass
class CCLEpaths:
    """
    Paths to CCLE dataset files.

    Attributes
    ----------
    transcriptomics : str
        Path to the CCLE RNA-seq expression data file.
    metabolomics : str
        Path to the CCLE metabolomics data file.
    cell_annotations : str
        Path to the cell line annotation file.
    """
    transcriptomics: str
    metabolomics: str
    cell_annotations: str
    metabo_mapping: str
    depmap_annotation: str

class Loader:
    """
    Extracts and organizes LINCS data for drug/cell line perturbations ONLY.
    
    Why this class?
    ---------------
    The LINCS dataset contains many perturbation types (shRNA, overexpression, etc.)
    but you only want compound (drug) treatments. This class filters everything else out
    and gives you clean drug × cell line data.
    """

    def __init__(self, lincs_paths: LINCSpaths, ccle_paths: CCLEpaths): # type: ignore
        """
        Initialize with paths to your downloaded files.
        
        Parameters:
        -----------
        gctx_path : str
            Path to GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx
        inst_info_path : str
            Path to GSE92742_Broad_LINCS_inst_info.txt
        gene_info_path : str, optional
            Path to GSE92742_Broad_LINCS_gene_info.txt
        cell_info_path : str, optional
            Path to GSE92742_Broad_LINCS_cell_info.txt
        """
        self.lincs_paths = lincs_paths
        self.ccle_paths = ccle_paths

        # L1000
        self.inst_info = pd.DataFrame()
        self.gene_info = pd.DataFrame()
        self.cell_info = pd.DataFrame()
        self.drug_samples = pd.DataFrame()  # Filtered for drugs only
        self.sig_info = pd.DataFrame()
        self.l1000_metadata = pd.DataFrame()
        self.compound_info = pd.DataFrame()
        self.l1000_exp_data = pd.DataFrame()
        self.l1000_pathway_data = pd.DataFrame()

        # CCLE
        self.ccle_transcriptomics = pd.DataFrame()
        self.ccle_metabolomics = pd.DataFrame()
        self.ccle_annotation = pd.DataFrame()
        self.metabo_mapping = pd.DataFrame()
        self.depmap_annotation = pd.DataFrame()

    def load_l1000_metadata(self):
        """
        Load all metadata files.
        
        Understanding the files:
        ------------------------

        """
        print("Loading metadata nested in gctx file...")

        self.l1000_metadata = cast(pd.DataFrame,parse(self.lincs_paths.gctx,
                col_meta_only=True))
        print(self.l1000_metadata.head())
        print(f' \n\n Loaded metadata with shape: {self.l1000_metadata.shape}')

        # Load instance info
        print("  - Loading sig_info...") # Key file working with lvl 5 data
        self.sig_info = pd.read_csv(
            self.lincs_paths.sig_info, sep='\t', low_memory=False)
        print(f"    Loaded {len(self.sig_info):,} total samples")

        # Load instance info
        print("  - Loading inst_info...")
        self.inst_info = pd.read_csv(
            self.lincs_paths.inst_info, sep='\t', low_memory=False, compression='gzip')
        print(f"    Loaded {len(self.inst_info):,} total samples")

        # Load gene info (if provided)
        if self.lincs_paths.gene_info:
            print("  - Loading gene_info...")
            self.gene_info = pd.read_csv(
                self.lincs_paths.gene_info, sep='\t', low_memory=False, compression='gzip')
            print(f"    Loaded {len(self.gene_info):,} genes")

        # Load cell info (if provided)
        if self.lincs_paths.cell_info:
            print("  - Loading cell_info...")
            self.cell_info = pd.read_csv(
                self.lincs_paths.cell_info, sep='\t', low_memory=False, compression='gzip')
            print(f"    Loaded {len(self.cell_info):,} cell lines")

        # Load compound info (if provided)
        if self.lincs_paths.compound_info:
            print("  - Loading cell_info...")
            self.compound_info = pd.read_csv(
                self.lincs_paths.compound_info, sep='\t', low_memory=False, compression='gzip')
            print(f"    Loaded {len(self.compound_info):,} cell lines")

    def check_metadata(self):
        """
        Print a report of metadata fields
        """
        for field in self.l1000_metadata.columns:

            print(f'Metadata field: {field} \n')
            print(f'Unique values for this field: {self.l1000_metadata[field].nunique()}\n')
            print([f'{unique_value} \n' for unique_value in self.l1000_metadata[field].unique()])

    def extract_data_subset(self, n_subset):
        """
        Args:
            n_subset (int, optional): int to N subset. Defaults to 0.
        """
        if n_subset :
            # Load metadata first to get sample IDs, then parse only those
            subset_cids = self.l1000_metadata.index[:n_subset].tolist()
            gctoo = parse(self.lincs_paths.gctx, cid=subset_cids)
        else:
            gctoo = parse(self.lincs_paths.gctx)

        self.l1000_exp_data = gctoo.data_df.T

    def load_l1000_pathway_scores(self):
        """
        load the pre-computed GSEA pathway scores derived from l1000 level 
        5 data from SigComLINCS (CD) (computed with min_size = 15)
        """
        print(f'\n Loading data from file: {self.lincs_paths.pathway}')
        self.l1000_pathway_data= pd.read_parquet(
            path=self.lincs_paths.pathway,
            engine = "pyarrow"
            )
        print(f'\n Loaded GSEA pathway scores with shape: {self.l1000_pathway_data.shape}')

    def extract_data_ids(self, ids: list):
        """
        Load expression data for a specific list of sample IDs from the gctx file.

        Parameters
        ----------
        ids : list
            List of column IDs (sig_ids) to extract from the gctx file.

        Returns
        -------
        pd.DataFrame or None
            Samples × genes expression DataFrame (transposed from gctx format),
            or None if the gctx file could not be read.
        """
        try:
            gctoo = parse(self.lincs_paths.gctx, cid=ids)
        except OSError as e:
            print(f'Following error happened: {e}')
            return None
        self.l1000_exp_data = gctoo.data_df.T
        return self.l1000_exp_data

    def tas_filtering(self, tas_threshold: float):
        """
        Filter signatures by Transcriptional Activity Score (TAS) and perturbation type.

        Retains only compound perturbations (`pert_type == 'trt_cp'`) with a TAS above
        the given threshold, then cross-references against IDs actually present in the
        gctx file. Prints a summary of cell line and drug counts after filtering.

        Parameters
        ----------
        tas_threshold : float
            Minimum TAS value (exclusive) a signature must have to be retained.

        Returns
        -------
        list
            List of sig_ids that pass the TAS filter and exist in the gctx file.
        """
        sig_info = self.sig_info.copy()

        # IDs available in the actual gctx file
        available_ids = set(self.l1000_metadata.index)

        # Filter by TAS threshold
        sig_info = sig_info[(sig_info['tas'] > tas_threshold) & (sig_info['pert_type'] == 'trt_cp')]
        candidate_ids = sig_info['sig_id'].to_list()

        # Only keep IDs that actually exist in gctx
        valid_ids = [i for i in candidate_ids if i in available_ids]

        # Diagnostic prints
        print(f"After TAS filter     : {len(candidate_ids)} IDs")
        print(f"Found in gctx        : {len(valid_ids)} IDs")
        print(f"Dropped (not in gctx): {len(candidate_ids) - len(valid_ids)} IDs")

        sig_info = sig_info[sig_info['sig_id'].isin(valid_ids)]

        print(f"N unique cell lines after filtering at TAS = {tas_threshold} : \n")
        print(sig_info['cell_iname'].nunique())
        print(f"\n N unique drugs after filtering at TAS = {tas_threshold} : \n")
        print(sig_info['cmap_name'].nunique())
        return valid_ids

    def load_ccle_data(self):
        """
        Load CCLE transcriptomics, metabolomics, and cell annotation data.

        Populates
        ---------
        self.ccle_transcriptomics : pd.DataFrame
            Gene expression matrix parsed from the GCT file.
        self.ccle_metabolomics : pd.DataFrame
            Metabolite abundance table loaded from CSV.
        self.ccle_annotation : pd.DataFrame
            Cell line annotation table loaded from a tab-separated file.
        """
        # Transcriptomics data
        print(f'\n Loading CCLE transcriptomics data file :{self.ccle_paths.transcriptomics}')
        self.ccle_transcriptomics = cast(pd.DataFrame, parse_gct(
            file_path=self.ccle_paths.transcriptomics
        ).data_df.T)
        print(self.ccle_transcriptomics.head())

        # Metabolomics data
        print(f'\n Loading CCLE metabolomics data file :{self.ccle_paths.metabolomics}')
        self.ccle_metabolomics = pd.read_csv(
            self.ccle_paths.metabolomics,
            index_col= 0
        )
        self.ccle_metabolomics = self.ccle_metabolomics.drop(labels='DepMap_ID', axis = 1)
        print(self.ccle_metabolomics.head())

        # Cell annotation file
        print(f'\n Loading CCLE annotations data file :{self.ccle_paths.cell_annotations}')
        self.ccle_annotation = pd.read_table(
            self.ccle_paths.cell_annotations,
            index_col=0,
            sep = '\t'
        )
        print(self.ccle_annotation.head())
        print(('\n Loading CCLE metabolite annotations '
                f'data file :{self.ccle_paths.metabo_mapping}'))
        self.metabo_mapping = pd.read_csv(
            self.ccle_paths.metabo_mapping
        )
        print(self.metabo_mapping.head())
        print(('\n Loading depmap annotations '
                f'data file :{self.ccle_paths.depmap_annotation}'))
        self.depmap_annotation = pd.read_csv(
            self.ccle_paths.depmap_annotation
        )
        print(self.depmap_annotation.head())

    def preprocess_metabolomics(self,
                                split_lipids: bool,
                                convert_ids:bool,
                                remove_unmapped:bool):
        """
        Filter the metabolomics matrix to retain only KEGG-mapped metabolites.

        Drops columns whose Query name has no KEGG ID in ``self.metabo_mapping``.
        Optionally removes lipid columns — defined as any column not present in
        the ``Query`` column of the mapping table.

        Parameters
        ----------
        remove_lipids : bool
            If True, additionally drop all metabolites that have no entry in
            ``self.metabo_mapping['Query']`` (i.e. lipids not covered by the
            KEGG mapping file).

        Modifies
        --------
        self.ccle_metabolomics : pd.DataFrame
            Updated in-place with unmapped (and optionally lipid) columns removed.
        """
        lipids_proc = pd.DataFrame()
        metabo_proc = self.ccle_metabolomics

        print(self.metabo_mapping['KEGG'].isna().sum())
        name_mapping = dict(zip(self.metabo_mapping['Query'], self.metabo_mapping['KEGG']))
        unmapped = [k for k, v in name_mapping.items() if pd.isna(v)]
        print(f'N metabolites with no kegg ids: {len(unmapped)}')

        print(metabo_proc.columns)
        if remove_unmapped:
            metabo_proc = metabo_proc.drop(labels=unmapped,axis=1)
        print(metabo_proc.shape)

        if split_lipids:
            query_set = set(self.metabo_mapping['Query'])
            kegg_set  = set(self.metabo_mapping['KEGG'].dropna())

            lipids = [
                col for col in metabo_proc.columns
                if col not in query_set and col not in kegg_set
            ]
            lipids_proc = metabo_proc.loc[:,lipids]
            metabo_proc = metabo_proc.drop(columns=lipids)
            print(f'\n Removed n lipids: {len(lipids)}')
            print(f'\n Metabolomics matrix shape: {metabo_proc.shape}')

        if convert_ids:
            metabo_proc = metabo_proc.rename(columns=name_mapping)
            print(metabo_proc.columns)

        return metabo_proc, lipids_proc
