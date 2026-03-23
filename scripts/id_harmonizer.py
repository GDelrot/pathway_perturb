"""
id_harmonizer.py — Map all datasets to a common Cellosaurus ID
================================================================

THE PROBLEM
───────────
Each dataset uses a DIFFERENT cell line identifier:

    CCLE metabo/transcripto  →  "DMS53_LUNG"          (CCLE_ID / ccle_name)
    L1000 pathway scores     →  "ABY001_A375_XH:BRD-K66175015:10:24"  (sig_id)
    L1000 cell_info          →  "A375" (cell_iname) + "CVCL_0132" (cellosaurus_id)
    DepMap annotations       →  "NIHOVCAR3_OVARY" (CCLEName) + "CVCL_0465" (RRID)

To merge across datasets, we need ONE common identifier.
Cellosaurus (CVCL_XXXX) is the best choice because:
    1. It's a universal cell line registry (standardized globally)
    2. Both L1000 cell_info and DepMap annotations already carry it
    3. It resolves ambiguities (e.g., "293T" vs "HEK293T" → same CVCL)

THE STRATEGY
────────────
    CCLE index ("DMS53_LUNG")
         │
         ▼  join with depmap_annotation on CCLEName
    depmap RRID ("CVCL_0049")
         │
         ▼  this IS the cellosaurus ID
    ✓ CCLE mapped

    L1000 sig_id ("ABY001_A375_XH:BRD-K66175015:10:24")
         │
         ▼  join with sig_info to get cell_iname ("A375")
    cell_iname
         │
         ▼  join with cell_info to get cellosaurus_id ("CVCL_0132")
    ✓ L1000 mapped
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def build_ccle_to_cellosaurus(
    depmap_annotation: pd.DataFrame,
    ccle_annotation: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Build a mapping: CCLE_name → Cellosaurus ID (CVCL_XXXX).

    Uses DepMap's 'CCLEName' and 'RRID' columns as the primary bridge.

    PEDAGOGIC NOTE: Why DepMap and not CCLE annotations?
    ────────────────────────────────────────────────────
    CCLE annotations (Cell_lines_annotations) don't have a Cellosaurus column.
    DepMap annotations DO have 'RRID' which IS the Cellosaurus accession
    (e.g., CVCL_0465). DepMap also has 'CCLEName' (e.g., NIHOVCAR3_OVARY)
    which matches the CCLE index format.

    Parameters
    ----------
    depmap_annotation : pd.DataFrame
        Must contain 'CCLEName' and 'RRID' columns.
    ccle_annotation : pd.DataFrame, optional
        Fallback: has 'CCLE_ID'-style names + some identifiers.

    Returns
    -------
    pd.Series
        Index = CCLE name (e.g., "NIHOVCAR3_OVARY")
        Values = Cellosaurus ID (e.g., "CVCL_0465")
    """
    # ── Primary mapping from DepMap ─────────────────────────────────────
    # 'CCLEName' column matches the index format of CCLE transcriptomics/metabolomics
    # 'RRID' column contains the Cellosaurus accession

    dm = depmap_annotation[['CCLEName', 'RRID']].dropna(subset=['CCLEName', 'RRID'])

    # Clean: RRID sometimes has prefix "CVCL_" already, sometimes not
    # Also filter out empty strings and "N/A"
    dm = dm[dm['RRID'].str.startswith('CVCL_', na=False)].copy()
    dm = dm[dm['CCLEName'].str.len() > 0]

    # Drop duplicates: keep first occurrence (some cell lines have aliases)
    dm = dm.drop_duplicates(subset='CCLEName', keep='first')

    mapping = dm.set_index('CCLEName')['RRID']

    logger.info(f"  DepMap CCLE→Cellosaurus mapping: {len(mapping)} cell lines")

    return mapping


def build_l1000_sig_to_cellosaurus(
    sig_info: pd.DataFrame,
    cell_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a mapping: L1000 sig_id → (cell_iname, drug_id, cellosaurus_id).

    L1000 SIGNATURE ID ANATOMY
    ──────────────────────────
    A sig_id like "ABY001_A375_XH:BRD-K66175015:10:24" encodes:

        ABY001     = batch/plate info
        A375       = cell line name (but NOT always reliable to parse!)
        XH         = cell line variant/passage
        BRD-K66175015 = Broad drug ID
        10         = dose (µM)
        24         = timepoint (hours)

    BUT: parsing is fragile (underscores in cell names, variable formats).
    The SAFE approach is to join with sig_info which has clean columns:
        sig_id → cell_iname, pert_id, pert_iname, etc.

    Then join cell_iname with cell_info to get cellosaurus_id.

    Parameters
    ----------
    sig_info : pd.DataFrame
        Must contain 'sig_id' and 'cell_iname' columns.
    cell_info : pd.DataFrame
        Must contain 'cell_iname' and 'cellosaurus_id' columns.

    Returns
    -------
    pd.DataFrame
        Columns: ['sig_id', 'cell_iname', 'cellosaurus_id', 'pert_id', 'pert_iname']
    """
    # ── Step 1: sig_id → cell_iname + drug info ────────────────────────
    #
    # sig_info is the master metadata table for all L1000 signatures.
    # We extract the columns we need for the ML pipeline.
    print(sig_info)
    print(cell_info)
    needed_cols = ['sig_id', 'cell_iname']
    drug_cols = ['pert_id', 'cmap_name']

    # Add drug columns if available
    for col in drug_cols:
        if col in sig_info.columns:
            needed_cols.append(col)

    sig_meta = sig_info[needed_cols].copy()

    logger.info(f"  sig_info: {len(sig_meta)} signatures")

    # ── Step 2: cell_iname → cellosaurus_id ─────────────────────────────
    #
    # cell_info maps cell line short names to cellosaurus accessions.
    # Not all cell lines will have a cellosaurus ID (some are "N/A").

    cell_map = cell_info[['cell_iname', 'cellosaurus_id']].copy()

    # Clean cellosaurus IDs: keep only valid CVCL_ entries
    cell_map = cell_map[cell_map['cellosaurus_id'].str.startswith('CVCL_', na=False)]
    cell_map = cell_map.drop_duplicates(subset='cell_iname', keep='first')

    logger.info(f"  cell_info: {len(cell_map)} cell lines with Cellosaurus ID")

    # ── Step 3: Merge sig_meta ← cell_map ──────────────────────────────
    #
    # LEFT JOIN: keep all signatures, add cellosaurus where available.
    # Signatures from cell lines without cellosaurus will get NaN.

    merged = sig_meta.merge(cell_map, on='cell_iname', how='left')

    n_mapped = merged['cellosaurus_id'].notna().sum()
    n_total = len(merged)
    logger.info(f"  Signatures with Cellosaurus ID: {n_mapped}/{n_total} "
                f"({n_mapped/n_total*100:.1f}%)")

    return merged

def harmonize_ids(
    ccle_transcriptomics: pd.DataFrame,
    ccle_metabolomics: pd.DataFrame,
    l1000_pathway_data: pd.DataFrame,
    sig_info: pd.DataFrame,
    cell_info: pd.DataFrame,
    depmap_annotation: pd.DataFrame,
    ccle_annotation: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Main function: reindex ALL datasets on Cellosaurus IDs.

    WHAT THIS FUNCTION DOES (step by step)
    ───────────────────────────────────────
    1. Build CCLE_name → Cellosaurus mapping (via DepMap RRID)
    2. Build L1000 sig_id → Cellosaurus mapping (via sig_info → cell_info)
    3. Reindex CCLE transcriptomics: rows become CVCL_XXXX
    4. Reindex CCLE metabolomics: rows become CVCL_XXXX
    5. Add 'cell_id' (= cellosaurus) and 'drug_id' columns to L1000

    Parameters
    ----------
    ccle_transcriptomics : pd.DataFrame
        Index = CCLE names (e.g., "22RV1_PROSTATE")
    ccle_metabolomics : pd.DataFrame
        Index = CCLE names (e.g., "DMS53_LUNG")
    l1000_pathway_data : pd.DataFrame
        Index = sig_ids (e.g., "ABY001_A375_XH:BRD-K66175015:10:24")
        Columns = pathway scores
    sig_info, cell_info, depmap_annotation : pd.DataFrame
        Annotation tables from your Loader.

    Returns
    -------
    trans_harmonized : pd.DataFrame — CCLE transcriptomics indexed by CVCL
    metab_harmonized : pd.DataFrame — CCLE metabolomics indexed by CVCL
    l1000_harmonized : pd.DataFrame — L1000 with 'cell_id' (CVCL) and 'drug_id' columns
    stats            : dict         — mapping statistics for diagnostics
    """
    print("\n" + "=" * 70)
    print("ID HARMONIZATION → Cellosaurus")
    print("=" * 70)

    stats = {}

    # ═══════════════════════════════════════════════════════════════════
    # 1. CCLE → Cellosaurus
    # ═══════════════════════════════════════════════════════════════════

    print("\n--- CCLE mapping ---")
    ccle_to_cvcl = build_ccle_to_cellosaurus(depmap_annotation, ccle_annotation)

    # Reindex transcriptomics
    trans_idx = ccle_transcriptomics.index
    trans_mapped = trans_idx.map(lambda x: ccle_to_cvcl.get(x, np.nan))
    n_trans_mapped = trans_mapped.notna().sum()
    print(f"  Transcriptomics: {n_trans_mapped}/{len(trans_idx)} cell lines mapped "
          f"({n_trans_mapped/len(trans_idx)*100:.1f}%)")

    # Show unmapped examples for debugging
    unmapped_trans = trans_idx[trans_mapped.isna()]
    if len(unmapped_trans) > 0:
        print(f"  Unmapped examples: {list(unmapped_trans[:5])}")

    # Apply mapping: drop unmapped cell lines
    trans_harmonized = ccle_transcriptomics.copy()
    trans_harmonized.index = trans_mapped
    trans_harmonized = trans_harmonized[trans_harmonized.index.notna()]
    # Handle duplicate CVCL IDs (rare, but some cell lines map to same cellosaurus)
    trans_harmonized = trans_harmonized[~trans_harmonized.index.duplicated(keep='first')]

    # Reindex metabolomics (same logic)
    metab_idx = ccle_metabolomics.index
    metab_mapped = metab_idx.map(lambda x: ccle_to_cvcl.get(x, np.nan))
    n_metab_mapped = metab_mapped.notna().sum()
    print(f"  Metabolomics:    {n_metab_mapped}/{len(metab_idx)} cell lines mapped "
          f"({n_metab_mapped/len(metab_idx)*100:.1f}%)")

    unmapped_metab = metab_idx[metab_mapped.isna()]
    if len(unmapped_metab) > 0:
        print(f"  Unmapped examples: {list(unmapped_metab[:5])}")

    metab_harmonized = ccle_metabolomics.copy()
    metab_harmonized.index = metab_mapped
    metab_harmonized = metab_harmonized[metab_harmonized.index.notna()]
    metab_harmonized = metab_harmonized[~metab_harmonized.index.duplicated(keep='first')]

    stats['ccle_trans_mapped'] = n_trans_mapped
    stats['ccle_trans_total'] = len(trans_idx)
    stats['ccle_metab_mapped'] = n_metab_mapped
    stats['ccle_metab_total'] = len(metab_idx)

    # ═══════════════════════════════════════════════════════════════════
    # 2. L1000 → Cellosaurus
    # ═══════════════════════════════════════════════════════════════════

    print("\n--- L1000 mapping ---")
    l1000_meta = build_l1000_sig_to_cellosaurus(sig_info, cell_info)

    # The L1000 pathway data is indexed by sig_id (the 'cid' column).
    # We need to join it with our mapping.
    print(l1000_pathway_data.head())
    l1000_harmonized = l1000_pathway_data.copy()
    l1000_harmonized = l1000_harmonized.reset_index()

    # Identify the sig_id column (might be 'cid', 'index', or 'sig_id')
    sig_col = None
    for candidate in ['cid', 'sig_id', 'index']:
        if candidate in l1000_harmonized.columns:
            sig_col = candidate
            break

    if sig_col is None:
        # The reset_index might have created a generic column name
        sig_col = l1000_harmonized.columns[0]
        print(f"  ⚠ Using first column '{sig_col}' as sig_id")

    print(f"  L1000 sig_id column: '{sig_col}'")
    print(f"  L1000 observations before merge: {len(l1000_harmonized)}")

    # Merge L1000 pathway scores ← metadata (cell_id + drug_id + cellosaurus)
    l1000_harmonized = l1000_harmonized.merge(
        l1000_meta,
        left_on=sig_col,
        right_on='sig_id',
        how='left',
    )

    # Drop observations without cellosaurus mapping
    n_before = len(l1000_harmonized)
    l1000_harmonized = l1000_harmonized.dropna(subset=['cellosaurus_id'])
    n_after = len(l1000_harmonized)
    print(f"  L1000 observations after Cellosaurus filter: {n_after}/{n_before} "
          f"({n_after/n_before*100:.1f}%)")

    # Rename to standard columns expected by ML pipeline
    print(l1000_harmonized.columns)
    print(l1000_harmonized.head())
    l1000_harmonized = l1000_harmonized.rename(columns={
        'cellosaurus_id': 'cell_id',
    })
    print(l1000_harmonized.columns)
    # Set drug_id: prefer pert_iname (human-readable) over pert_id (Broad ID)
    if 'pert_iname' in l1000_harmonized.columns:
        l1000_harmonized = l1000_harmonized.rename(columns={'pert_iname': 'drug_id'})
    elif 'pert_id' in l1000_harmonized.columns:
        l1000_harmonized = l1000_harmonized.rename(columns={'pert_id': 'drug_id'})

    # Clean up: keep only cell_id, drug_id, and pathway score columns
    # Drop sig_id, cell_iname, and other metadata columns
    pathway_cols = [c for c in l1000_pathway_data.columns]
    keep_cols = ['cell_id', 'drug_id'] + [c for c in pathway_cols if c in l1000_harmonized.columns]
    l1000_harmonized = l1000_harmonized[keep_cols]

    stats['l1000_mapped'] = n_after
    stats['l1000_total'] = n_before

    # ═══════════════════════════════════════════════════════════════════
    # 3. OVERLAP SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    print("\n--- Overlap summary ---")
    trans_cells = set(trans_harmonized.index)
    metab_cells = set(metab_harmonized.index)
    l1000_cells = set(l1000_harmonized['cell_id'].unique())

    overlap_all = trans_cells & metab_cells & l1000_cells
    overlap_trans_l1000 = trans_cells & l1000_cells
    overlap_metab_l1000 = metab_cells & l1000_cells

    print(f"  Transcriptomics CVCL IDs: {len(trans_cells)}")
    print(f"  Metabolomics CVCL IDs:    {len(metab_cells)}")
    print(f"  L1000 CVCL IDs:           {len(l1000_cells)}")
    print(f"  Trans ∩ L1000:            {len(overlap_trans_l1000)}")
    print(f"  Metab ∩ L1000:            {len(overlap_metab_l1000)}")
    print(f"  ALL THREE:                {len(overlap_all)}")

    if len(overlap_all) == 0:
        print("\n  ⚠ WARNING: Zero overlap across all 3 datasets!")
        print("    Check ID formats — here are samples from each:")
        print(f"    Trans:  {list(trans_cells)[:3]}")
        print(f"    Metab:  {list(metab_cells)[:3]}")
        print(f"    L1000:  {list(l1000_cells)[:3]}")

    stats['overlap_all'] = len(overlap_all)
    stats['overlap_trans_l1000'] = len(overlap_trans_l1000)
    stats['overlap_metab_l1000'] = len(overlap_metab_l1000)

    print(f"\n  Final shapes:")
    print(f"    Transcriptomics: {trans_harmonized.shape}")
    print(f"    Metabolomics:    {metab_harmonized.shape}")
    print(f"    L1000:           {l1000_harmonized.shape}")

    return trans_harmonized, metab_harmonized, l1000_harmonized, stats