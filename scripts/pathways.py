""" This script holds the class that handles all features
related to pathways
Portions of this file are adapted from the sspa library:
#   Wieder, C. et al. (2022). sspa: A Python package for single-sample pathway analysis.
#   GitHub: https://github.com/cwieder/py-sspa
#   License: MIT
"""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
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
            omics_type(str): type of omics pathways to download (metabolomics, transcriptomics, or multiomics)
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

            for index, i in enumerate(tqdm(pathway_ids)):
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

            for index, i in enumerate(tqdm(pathway_ids)):
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
                page = requests.get(current_url)
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

    def load_gmt(self,
                 path: str,
                 omics: str):
        """
        Loads an existing pathway gmt file and stores both the dict
        and DataFrame representations.

        Args:
            path (str): path to existing gmt
            omics (str): omics type ('metabolomics', 'transcriptomics', or 'multiomics')
        """
        # Load the gmt as a DataFrame
        gmt = pd.read_csv(path, sep='\t',header=None)
        gmt.drop(columns=gmt.columns[1], inplace=True)
        gmt.set_index(keys=gmt.columns[0], inplace=True, drop=True)

        # Also build the dict representation
        pathway_dict = {}
        for pathway in gmt.index:
            genes = gmt.loc[pathway].dropna()
            pathway_dict[pathway] = [
                str(int(g)) if isinstance(g, float) else str(g)
                for g in genes
            ]

        # Store both
        self._store_pathways(omics, pathway_dict, gmt)

    def convert_metabolite_ids(self,
                            input_type: str,
                            compound_list: list) -> pd.DataFrame:
        """
        Use MetaboAnalyst API for metabolite identifier conversion.

        Adapted from sspa.utils.identifier_conversion():
        https://github.com/cwieder/py-sspa/blob/main/sspa/utils.py
        Wieder, C. et al. (2022). sspa: A Python package for single-sample
        pathway analysis. License: MIT

        Args:
            input_type (str): Identifier type present in input data.
                One of: 'name', 'hmdb', 'pubchem', 'chebi', 'metlin', 'kegg'
            compound_list (list): List of identifiers to convert.

        Returns:
            pd.DataFrame: DataFrame containing identifier matches.

        Raises:
            NotImplementedError: If input_type is not 'name'.
            requests.exceptions.RequestException: If the API call fails.
        """
        if input_type != 'name':
            raise NotImplementedError(
                "Currently the MetaboAnalyst API only converts "
                "from compound names to other identifiers."
            )

        print("Commencing ID conversion using MetaboAnalyst API...")

        url = "https://www.xialab.ca/api/mapcompounds"

        payload = json.dumps({
            "queryList": ";".join(compound_list) + ",",
            "inputType": input_type
        })

        headers = {
            "Content-Type": "application/json",
            "cache-control": "no-cache",
        }

        response = requests.post(url, data=payload, headers=headers, timeout=120)
        response.raise_for_status()  # raises an exception on HTTP 4xx/5xx

        return pd.DataFrame(response.json())
