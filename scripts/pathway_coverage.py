"""
Pathway Coverage Analysis Module

This module calculates and visualizes how well our molecular datasets 
represent the molecules in known pathways.

Key concept: Coverage = (molecules from pathway present in our data) / (total molecules in pathway)
This metric tells us: "What fraction of each pathway can we actually study?"
"""

from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_pathway_coverage(
    pathway_dict: Dict[str, List[str]],
    available_molecules: List[str],
    pathway_name: str = "Pathway"
) -> Tuple[pd.Series, Dict]:
    """
    Calculate the coverage ratio for each pathway.
    
    Coverage = (molecules from pathway found in our dataset) / (total molecules in pathway)
    
    Args:
        pathway_dict: Dict mapping pathway_id -> list of molecule names in pathway
        available_molecules: List of molecules actually present in our dataset
        pathway_name: Name of the omics type (for reporting)
    
    Returns:
        coverage_ratios: Series of coverage values per pathway (index=pathway_id)
        coverage_stats: Dict with summary statistics
    """
    
    # Convert available molecules to a set for fast lookup
    available_set = set(available_molecules)
    
    coverage_data = {}
    
    # For each pathway, calculate what fraction we can study
    for pathway_id, molecules_in_pathway in pathway_dict.items():
        # How many molecules from this pathway are in our dataset?
        found_molecules = sum(1 for mol in molecules_in_pathway if mol in available_set)
        total_molecules = len(molecules_in_pathway)

        # Avoid division by zero
        if total_molecules > 0:
            coverage_ratio = found_molecules / total_molecules
            coverage_data[pathway_id] = coverage_ratio
    
    # Convert to pandas Series for easy manipulation and visualization
    coverage_ratios = pd.Series(coverage_data)
    
    # Calculate summary statistics
    coverage_stats = {
        'mean': coverage_ratios.mean(),
        'median': coverage_ratios.median(),
        'std': coverage_ratios.std(),
        'min': coverage_ratios.min(),
        'max': coverage_ratios.max(),
        'pathways_full_coverage': (coverage_ratios == 1.0).sum(),  # Pathways we have 100%
        'pathways_partial_coverage': ((coverage_ratios > 0) & (coverage_ratios < 1.0)).sum(),  # Partial
        'pathways_no_coverage': (coverage_ratios == 0).sum(),  # Pathways we have none of
        'n_pathways_total': len(coverage_ratios)
    }
    
    return coverage_ratios, coverage_stats


def plot_pathway_coverage_histogram(
    coverage_ratios: pd.Series,
    coverage_stats: Dict,
    omics_type: str,
    output_path: str,
    n_bins: int = 30
) -> None:
    """
    Create a histogram visualization of pathway coverage.
    
    This visualization shows the distribution of how well we can study each pathway.
    - Right side (1.0) = pathways where we have ALL molecules
    - Left side (0.0) = pathways where we have NO molecules
    
    Args:
        coverage_ratios: Series of coverage values per pathway
        coverage_stats: Dictionary with summary statistics
        omics_type: Name of omics type (e.g., 'Metabolomics', 'Transcriptomics')
        output_path: Where to save the figure
        n_bins: Number of histogram bins
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    ax.hist(
        coverage_ratios.values,
        bins=n_bins,
        color='steelblue',
        edgecolor='black',
        alpha=0.7
    )
    
    # Add vertical lines for summary statistics
    ax.axvline(coverage_stats['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {coverage_stats['mean']:.3f}")
    ax.axvline(coverage_stats['median'], color='orange', linestyle='--', 
               linewidth=2, label=f"Median: {coverage_stats['median']:.3f}")
    
    # Labels and formatting
    ax.set_xlabel('Pathway Coverage Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Pathways', fontsize=12, fontweight='bold')
    ax.set_title(f'{omics_type}: Pathway Coverage Distribution\n'
                 f'({coverage_stats["n_pathways_total"]} pathways analyzed)',
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_coverage_summary(
    coverage_stats: Dict,
    omics_type: str
) -> None:
    """
    Print human-readable summary of pathway coverage statistics.
    
    Args:
        coverage_stats: Dictionary with coverage statistics
        omics_type: Name of omics type
    """
    
    print(f"\n{'='*60}")
    print(f"{omics_type.upper()} - PATHWAY COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total pathways:           {coverage_stats['n_pathways_total']}")
    print(f"Full coverage (100%):     {coverage_stats['pathways_full_coverage']}")
    print(f"Partial coverage (0-100%): {coverage_stats['pathways_partial_coverage']}")
    print(f"No coverage (0%):         {coverage_stats['pathways_no_coverage']}")
    print(f"\nCoverage statistics:")
    print(f"  Mean coverage:          {coverage_stats['mean']:.1%}")
    print(f"  Median coverage:        {coverage_stats['median']:.1%}")
    print(f"  Std deviation:          {coverage_stats['std']:.3f}")
    print(f"  Range:                  {coverage_stats['min']:.1%} - {coverage_stats['max']:.1%}")
    print(f"{'='*60}\n")