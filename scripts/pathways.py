""" This script holds the class that handles all features
related to pathways
Portions of this file are adapted from the sspa library:
#   Wieder, C. et al. (2022). sspa: A Python package for single-sample pathway analysis.
#   GitHub: https://github.com/cwieder/py-sspa
#   License: MIT
"""
from dataclasses import dataclass
from typing import Dict, Optional,List
import re
import requests
from tqdm import tqdm
import pandas as pd
import mygene

@dataclass
class PathwayData:
    """
    Bundles the two representations of a pathway set.

    Why a dataclass? It gives you:
    - A clear, named container (better than a raw tuple or dict)
    - Easy to extend later (just add a field)
    - Auto-generated __repr__ for debugging

    Attributes:
        pathways_dict: {pathway_id: [gene/compound list]}
        gmt: GMT-style DataFrame (pathway_id as index, Pathway_name + gene columns)
    """
    pathways_dict: Optional[Dict[str, list]] = None
    gmt: Optional[pd.DataFrame] = None

class Pathways():
    """
    This class handles downloading pathways,
    filtering, statistics, id conversion...
    """

    VALID_OMICS_TYPES = {"metabolomics", "transcriptomics", "multiomics"}

    def __init__(self):
        # Each omics type gets its own PathwayData container.
        # Accessing: self.kegg_metabolomics.gmt or self.kegg_metabolomics.pathways_dict
        self.kegg_metabolomics: PathwayData = PathwayData()
        self.kegg_transcriptomics: PathwayData = PathwayData()
        self.kegg_multiomics: PathwayData = PathwayData()
        self.pathway_names : Dict

    def _get_pathway_data(self, omics_type: str) -> PathwayData:
        """
        Returns the PathwayData object for a given omics type.
        Raises ValueError early if the type is invalid.
        """
        if omics_type not in self.VALID_OMICS_TYPES:
            raise ValueError(
                f"Unknown omics_type '{omics_type}'. "
                f"Must be one of: {self.VALID_OMICS_TYPES}"
            )
        return getattr(self, f"kegg_{omics_type}")

    def _store_pathways(self, omics_type: str, pathways_dict: dict, gmt: pd.DataFrame):
        """
        Stores both representations into the correct PathwayData object.
        """
        pathway_data = self._get_pathway_data(omics_type)
        pathway_data.pathways_dict = pathways_dict
        pathway_data.gmt = gmt

    def download_kegg(self,
                      organism: str = "hsa",
                      filepath=None,
                      omics_type='transcriptomics'):
        '''
        Adapted from sspa.utils.download_KEGG():
        https://github.com/cwieder/py-sspa/blob/main/sspa/utils.py
        Function for KEGG pathway download 
        Args:
            organism (str): KEGG 3 letter organism code
            filepath (str): filepath to save pathway file to, default is None - save to variable
            omics_type(str): type of omics pathways to download
            (metabolomics, transcriptomics, or multiomics)
        Returns: 
            GMT-like pd.DataFrame containing KEGG pathways
        '''
        # Validate early — fail fast before any network requests
        self._get_pathway_data(omics_type)

        print("Beginning KEGG download...")
        url = 'http://rest.kegg.jp/list/pathway/' + organism
        data = requests.get(url, timeout=600)
        pathways = data.text
        pathways = pathways.split("\n")
        pathways = filter(None, pathways)
        pathway_dict = dict()

        for path in pathways:
            path = path.split("\t")
            name = path[1]
            pathid = re.search(r"(.*)", path[0]).group(1)  # type:ignore
            pathway_dict[pathid] = name

        base_url = 'http://rest.kegg.jp/get/'
        pathway_ids = [*pathway_dict]

        # get release details
        release_data = requests.get('http://rest.kegg.jp/info/kegg', timeout=600)
        version_no = release_data.text.split()[9][0:3]

        if omics_type == 'metabolomics':
            pathway_compound_mapping = dict()

            for _, i in enumerate(tqdm(pathway_ids)):
                complist = []
                current_url = base_url + i
                page = requests.get(current_url, timeout=600)
                lines = page.text.split("\n")

                try:
                    cpds_start = [lines.index(i) for i in lines if i.startswith("COMPOUND")][0]
                    reference_start = [lines.index(i) for i in lines if i.startswith("REFERENCE") or i.startswith("REL_PATHWAY")][0]
                    cpds_lines = lines[cpds_start:reference_start]
                    first_cpd = cpds_lines.pop(0).split()[1]
                    complist.append(first_cpd)
                    complist = complist + [i.split()[0] for i in cpds_lines]
                    pathway_compound_mapping[i] = list(set(complist))
                except IndexError:
                    pathway_compound_mapping[i] = []

            pathway_compound_mapping = {k: v for k, v in pathway_compound_mapping.items() if v}

            df = pd.DataFrame.from_dict(pathway_compound_mapping, orient='index')
            df.insert(0, 'Pathway_name', df.index.map(pathway_dict.get))

            if filepath:
                fpath = filepath + "/KEGG_" + organism + "_pathways_compounds_R" + str(version_no) + ".gmt"
                df.to_csv(fpath, sep="\t", header=False)
                print("KEGG DB file saved to " + fpath)
            print("Complete!")

            self._store_pathways(omics_type, pathway_compound_mapping, df)
            return df

        if omics_type == 'multiomics':
            pathway_mapping = dict()

            for _, i in enumerate(tqdm(pathway_ids)):
                complist = []
                genelist = []
                current_url = base_url + i

                page = requests.get(current_url, timeout=600)
                lines = page.text.split("\n")

                try:
                    genes_start = [lines.index(i) for i in lines if i.startswith("GENE")][0]
                    cpds_start = [lines.index(i) for i in lines if i.startswith("COMPOUND")][0]
                    reference_start = [lines.index(i) for i in lines if i.startswith("REFERENCE") or i.startswith("REL_PATHWAY")][0]
                    genes_lines = lines[genes_start:cpds_start]
                    cpds_lines = lines[cpds_start:reference_start]

                    first_cpd = cpds_lines.pop(0).split()[1]
                    complist.append(first_cpd)
                    complist = complist + [i.split()[0] for i in cpds_lines]
                    first_gene = genes_lines.pop(0).split()[1]
                    genelist.append(first_gene)
                    genelist = genelist + [i.split()[0] for i in genes_lines]
                    pathway_mapping[i] = list(set(complist)) + list(set(genelist))
                except IndexError:
                    pathway_mapping[i] = []

            pathway_mapping = {k: v for k, v in pathway_mapping.items() if v}

            df = pd.DataFrame.from_dict(pathway_mapping, orient='index')
            df.insert(0, 'Pathway_name', df.index.map(pathway_dict.get))

            if filepath:
                fpath = filepath + "/KEGG_" + organism + "_pathways_multiomics_R" + str(version_no) + ".gmt"
                df.to_csv(fpath, sep="\t", header=False)
                print("KEGG DB file saved to " + fpath)
            print("Complete!")

            self._store_pathways(omics_type, pathway_mapping, df)
            return df

        if omics_type == 'transcriptomics':
            pathway_mapping = dict()

            for index, i in enumerate(tqdm(pathway_ids)):
                genelist = []
                current_url = base_url + i
                page = requests.get(current_url,timeout=300)
                lines = page.text.split("\n")

                try:
                    genes_start = [lines.index(i) for i in lines if i.startswith("GENE")][0]
                    cpds_start = [lines.index(i) for i in lines if i.startswith("COMPOUND")][0]
                    reference_start = [lines.index(i) for i in lines if i.startswith("REFERENCE") or i.startswith("REL_PATHWAY")][0]
                    genes_lines = lines[genes_start:cpds_start]

                    first_gene = genes_lines.pop(0).split()[1]
                    genelist.append(first_gene)
                    genelist = genelist + [i.split()[0] for i in genes_lines]
                    pathway_mapping[i] = list(set(genelist))
                except IndexError:
                    pathway_mapping[i] = []

            pathway_mapping = {k: v for k, v in pathway_mapping.items() if v}

            df = pd.DataFrame.from_dict(pathway_mapping, orient='index')
            df.insert(0, 'Pathway_name', df.index.map(pathway_dict.get))

            if filepath:
                fpath = str(filepath) + "/KEGG_" + organism + "_pathways_transcriptomics_R" + str(version_no) + ".gmt"
                df.to_csv(fpath, sep="\t", header=False)
                print("KEGG DB file saved to " + fpath)
            print("Complete!")

            self._store_pathways(omics_type, pathway_mapping, df)
            return df

    def convert_gene_ids(self,
                    input_ids: list,
                    source: str = "ensembl.gene",
                    target: str = "symbol",
                    species: str = "human") -> dict:
        """
        Converts gene IDs from one format to another using MyGene.info.

        Args:
            input_ids (list): List of gene identifiers to convert.
            source (str): The field type of your input IDs.
                        Common values: 'ensembl.gene', 'symbol', 'entrezgene', 'refseq'
            target (str): The field(s) you want returned.
                        Can be a comma-separated string: 'symbol,entrezgene'
            species (str): Species filter. E.g. 'human', 'mouse', 'rat'

        Returns:
            dict: Mapping of input_id -> converted value(s)
        """
        mg = mygene.MyGeneInfo()

        results = mg.querymany(
            input_ids,
            scopes=source,
            fields=target,
            species=species,
            as_dataframe=False,
            returnall=False
        )

        converted = {}
        for hit in results:
            query_id = hit["query"]
            if "notfound" in hit:
                converted[query_id] = None
            else:
                converted[query_id] = hit.get(target, None)

        return converted

    def load_gmt(self, path: str, omics: str):
        """
        Loads an existing pathway gmt file and stores both the dict
        and DataFrame representations.
        Converts gene IDs to a common format (symbols for transcriptomics).
        
        Args:
            path (str): path to existing gmt
            omics (str): omics type ('metabolomics', 'transcriptomics', or 'multiomics')
        """
        # Load as strings
        gmt = pd.read_csv(path, sep='\t', header=None, dtype=str)
        self.pathway_names = dict(zip(gmt.loc[:,gmt.columns[0]],gmt.loc[:,gmt.columns[1]]))
        gmt.drop(columns=gmt.columns[1], inplace=True)
        gmt.set_index(keys=gmt.columns[0], inplace=True, drop=True)
        
        pathway_dict = {}
        
        if omics == 'transcriptomics':
            # Step 1: Extract all Entrez gene IDs from GMT
            all_entrez_ids = set()
            for pathway in gmt.index:
                genes = gmt.loc[pathway].dropna()
                # Convert to int then str to clean up decimals
                entrez_ids = [str(int(float(g))) for g in genes]
                all_entrez_ids.update(entrez_ids)
            
            print(f"Converting {len(all_entrez_ids)} Entrez IDs to symbols...")

            # Step 2: Batch convert Entrez IDs → symbols
            conversion = self.convert_gene_ids(
                input_ids=list(all_entrez_ids),
                source='entrezgene',
                target='symbol',
                species='human'
            )

            # Step 3: Build pathway dict with converted symbols
            for pathway in gmt.index:
                genes = gmt.loc[pathway].dropna()
                entrez_ids = [str(int(float(g))) for g in genes]
                # Convert to symbols, skip any that didn't convert (None values)
                symbols = [
                    conversion[eid] for eid in entrez_ids
                    if conversion.get(eid) is not None
                ]
                if symbols:  # Only keep pathways with at least one mapped gene
                    pathway_dict[pathway] = symbols
            
            print(f"Converted {len(pathway_dict)} pathways with symbol mapping")
            
        elif omics == 'metabolomics':
            for pathway in gmt.index:
                met = gmt.loc[pathway].dropna()
                pathway_dict[pathway] = [str(m).strip() for m in met]
        
        self._store_pathways(omics, pathway_dict, gmt)
        
    def pathway_intersections(self,
                          rna_pathways: Dict,
                          metabo_pathways: Dict,
                          metabolite_ms: List,
                          rna_ms: List,
                          l1000_pathways: List,
                          metabo_thresholds: List):
        """
        Find pathway intersections across three datasets at different metabolite thresholds.
        """
        # Convert to sets once (for performance)
        metabolite_set = set(metabolite_ms)
        rna_set = set(rna_ms)
        l1000_set = set(l1000_pathways)
        print(l1000_pathways)
        # ===== FILTER RNA PATHWAYS (FIXED THRESHOLD) =====
        rna_path_measured = {
            k: [rna for rna in v if rna in rna_set]
            for k, v in rna_pathways.items()
        }

        rna_threshold = 15  # Make this a parameter if flexible
        rna_path_filtered = {
            k: v for k, v in rna_path_measured.items()
            if len(v) >= rna_threshold
        }
        rna_pathways_set = set(rna_path_filtered.keys())

        print("=" * 60)
        print("RNA PATHWAYS REPORT")
        print("=" * 60)
        print(f"RNA genes measured: {len(rna_set)}")
        print(f"RNA pathways with ≥{rna_threshold} genes: {len(rna_pathways_set)}\n")

        # ===== FILTER METABOLOMICS PATHWAYS (VARIABLE THRESHOLDS) =====
        results = {}

        print("=" * 60)
        print("METABOLOMICS PATHWAYS BY THRESHOLD")
        print("=" * 60)
        print(f"Metabolites measured: {len(metabolite_set)}")
        print(f"L1000 reference pathways: {len(l1000_set)}\n")

        for threshold in metabo_thresholds:
            # Filter metabolomics pathways
            met_path_measured = {
                k: [met for met in v if met in metabolite_set]
                for k, v in metabo_pathways.items()
            }

            met_path_filtered = {
                k: v for k, v in met_path_measured.items()
                if len(v) >= threshold
            }

            met_pathways_set = set(met_path_filtered.keys())

            # Triple intersection
            pathway_inter = met_pathways_set & rna_pathways_set & l1000_set

            # Store results
            results[threshold] = {
                'pathways': pathway_inter,
                'count': len(pathway_inter),
                'met_pathways_count': len(met_pathways_set)
            }

            # Print report for this threshold
            print(f"Threshold: ≥{threshold} metabolites")
            print(f"  ├─ Metabolomics pathways: {len(met_pathways_set)}")
            print(f"  ├─ Intersecting with RNA: {len(met_pathways_set & rna_pathways_set)}")
            print(f"  └─ Triple intersection (Metabo ∩ RNA ∩ L1000): {len(pathway_inter)}")

            if pathway_inter:
                print(f"     Pathways: {sorted(pathway_inter)}\n")
            else:
                print(f"     (No intersecting pathways)\n")

        return results