"""
drug_encoding.py — Drug encoding strategies for ML pipeline
=============================================================
Provides fingerprint-based and hybrid drug encodings
that plug directly into the prepare_features() function.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DrugEncoder:
    """
    Container for a fitted drug encoder.
    Stores the encoding matrix and the mapping from drug_id → row index,
    so at inference time you can encode new observations by lookup.
    """
    encoding_matrix: np.ndarray       # shape (n_drugs, n_features)
    drug_to_idx: dict                 # drug_id → row in encoding_matrix
    feature_names: list               # column names for the encoding
    method: str                       # 'fingerprint', 'onehot', 'descriptor', 'hybrid'
    n_bits: int = 2048                # fingerprint length (if applicable)
    radius: int = 2                   # Morgan radius (if applicable)

    def transform(self, drug_ids: np.ndarray) -> np.ndarray:
        """
        Look up encodings for an array of drug IDs.

        Parameters
        ----------
        drug_ids : array of drug identifiers (same format as training)

        Returns
        -------
        np.ndarray of shape (len(drug_ids), n_features)
        """
        indices = []
        missing = set()
        for d in drug_ids:
            if d in self.drug_to_idx:
                indices.append(self.drug_to_idx[d])
            else:
                missing.add(d)
                indices.append(-1)

        if missing:
            logger.warning(
                f"  {len(missing)} drugs not found in encoder — will be zero-vectors. "
                f"Examples: {list(missing)[:5]}"
            )

        result = np.zeros((len(drug_ids), self.encoding_matrix.shape[1]))
        for i, idx in enumerate(indices):
            if idx >= 0:
                result[i] = self.encoding_matrix[idx]

        return result


def compute_morgan_fingerprints(
    compound_info: pd.DataFrame,
    drug_id_col: str = "pert_id",
    smiles_col: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> DrugEncoder:
    """
    Compute Morgan fingerprints for all drugs in compound_info.

    Parameters
    ----------
    compound_info : pd.DataFrame
        Must contain columns for drug ID and SMILES string.
    drug_id_col : str
        Column name for the drug identifier (must match drug_col in L1000).
    smiles_col : str
        Column name for the SMILES string.
    radius : int
        Morgan fingerprint radius. 2 = ECFP4, 3 = ECFP6.
    n_bits : int
        Length of the bit vector. 2048 is standard.

    Returns
    -------
    DrugEncoder with the fingerprint matrix and lookup dict.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Deduplicate: one fingerprint per unique drug
    df = compound_info[[drug_id_col, smiles_col]].drop_duplicates(subset=drug_id_col)
    df = df.dropna(subset=[smiles_col])

    drug_ids = []
    fp_matrix = []
    failed = []

    for _, row in df.iterrows():
        drug_id = row[drug_id_col]
        smiles = row[smiles_col]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(drug_id)
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fp_array = np.array(fp, dtype=np.float32)

        drug_ids.append(drug_id)
        fp_matrix.append(fp_array)

    if failed:
        logger.warning(f"  Failed to parse SMILES for {len(failed)} drugs: {failed[:5]}...")

    fp_matrix = np.vstack(fp_matrix)
    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}

    feature_names = [f"fp_{i}" for i in range(n_bits)]

    logger.info(
        f"  Morgan fingerprints computed: {len(drug_ids)} drugs, "
        f"radius={radius}, n_bits={n_bits}"
    )

    return DrugEncoder(
        encoding_matrix=fp_matrix,
        drug_to_idx=drug_to_idx,
        feature_names=feature_names,
        method="fingerprint",
        n_bits=n_bits,
        radius=radius,
    )


def compute_descriptors(
    compound_info: pd.DataFrame,
    drug_id_col: str = "pert_id",
    smiles_col: str = "canonical_smiles",
) -> DrugEncoder:
    """
    Compute RDKit physicochemical descriptors for all drugs.

    Returns a dense matrix of ~200 descriptors per drug.
    NaN values (some descriptors fail on some molecules) are filled with 0.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    descriptor_names = [name for name, _ in Descriptors.descList]
    calc = MolecularDescriptorCalculator(descriptor_names)

    df = compound_info[[drug_id_col, smiles_col]].drop_duplicates(subset=drug_id_col)
    df = df.dropna(subset=[smiles_col])

    drug_ids = []
    desc_matrix = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue
        desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        drug_ids.append(row[drug_id_col])
        desc_matrix.append(desc)

    desc_matrix = np.vstack(desc_matrix)

    # Replace NaN/Inf with 0
    desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
    feature_names = [f"desc_{n}" for n in descriptor_names]

    logger.info(f"  Physicochemical descriptors: {len(drug_ids)} drugs, {len(descriptor_names)} descriptors")

    return DrugEncoder(
        encoding_matrix=desc_matrix,
        drug_to_idx=drug_to_idx,
        feature_names=feature_names,
        method="descriptor",
    )


def compute_hybrid_encoding(
    compound_info: pd.DataFrame,
    drug_id_col: str = "pert_id",
    smiles_col: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> DrugEncoder:
    """
    Hybrid encoding: Morgan fingerprints + physicochemical descriptors concatenated.
    """
    fp_enc = compute_morgan_fingerprints(compound_info, drug_id_col, smiles_col, radius, n_bits)
    desc_enc = compute_descriptors(compound_info, drug_id_col, smiles_col)

    # Only keep drugs present in both
    common_drugs = sorted(set(fp_enc.drug_to_idx) & set(desc_enc.drug_to_idx))

    fp_rows = np.array([fp_enc.encoding_matrix[fp_enc.drug_to_idx[d]] for d in common_drugs])
    desc_rows = np.array([desc_enc.encoding_matrix[desc_enc.drug_to_idx[d]] for d in common_drugs])

    hybrid_matrix = np.hstack([fp_rows, desc_rows])
    drug_to_idx = {d: i for i, d in enumerate(common_drugs)}
    feature_names = fp_enc.feature_names + desc_enc.feature_names

    logger.info(f"  Hybrid encoding: {len(common_drugs)} drugs, {hybrid_matrix.shape[1]} features")

    return DrugEncoder(
        encoding_matrix=hybrid_matrix,
        drug_to_idx=drug_to_idx,
        feature_names=feature_names,
        method="hybrid",
    )