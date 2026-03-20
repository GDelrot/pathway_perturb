"""
This script contains functions related to plotting features
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse

def plot_pca(
    pca_plot: pd.DataFrame,        # must contain PC1, PC2, and hue_col
    explained_var: np.ndarray,     # from ipca.explained_variance_ratio_
    hue_col: str = 'cell_line',    # what to color by
    out_path: str = '',         # if None, just show interactively
    title: str = 'PCA',
):
    """
    Scatter PCA plot colored by a categorical column.

    Saves to out_path if provided (non-empty string), otherwise displays interactively.
    Pass out_path=None to display interactively; pass out_path='' or a file path to save.
    """
    pc1_var = explained_var[0] * 100
    pc2_var = explained_var[1] * 100

    _, ax = plt.subplots(figsize=(14, 11))
    sns.scatterplot(
        data=pca_plot, x="PC1", y="PC2",
        hue=hue_col, alpha=0.8, s=10, linewidth=0, ax=ax
    )
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', markerscale=2)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()   # free memory — important if you call this many times in a loop
    else:
        plt.show()

def plot_umap(
    umap_plot: pd.DataFrame,       # must contain UMAP1, UMAP2, and hue_col
    hue_col: str = 'cell_line',   # what to color by
    out_path: str = '',            # if None, just show interactively
    title: str = 'UMAP',
):
    """
    Scatter UMAP plot colored by a categorical column.
    Mirrors plot_pca() structure.
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.scatterplot(
        data=umap_plot, x="UMAP1", y="UMAP2",
        hue=hue_col, alpha=0.8, s=10, linewidth=0, ax=ax
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pca_density(pca_df: pd.DataFrame, hue_col : str,explained_var,
                    title: str, out_path: Path) -> None:
    """
    Plots PCA as a 2D KDE density, with marginal KDEs on the sides.
    Better than scatter for large n.
    """
    _, ax = plt.subplots(figsize=(10, 8))

    # 2D density — far more readable than 60k points
    sns.kdeplot(
        data=pca_df, x='PC1', y='PC2',
        hue=hue_col,       # color by group
        fill=True, alpha=0.4,
        common_norm=False,     # each group normalized independently
        ax=ax
    )
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_pca_joint(pca_df: pd.DataFrame, explained_var,
                   title:str,hue_col:str,out_path: Path) -> None:
    """
    Jointplot: scatter in center, KDE distributions on margins.
    """
    g = sns.jointplot(
        data=pca_df, x='PC1', y='PC2',
        kind='scatter',
        hue=hue_col,         # color by group
        alpha=0.05,              # very transparent — essential at 60k points
        s=2,                     # tiny points
        marginal_kws={'fill': True, 'common_norm': False}
    )
    g.set_axis_labels(
        f'PC1 ({explained_var[0]*100:.1f}%)',
        f'PC2 ({explained_var[1]*100:.1f}%)'
    )
    plt.title(title)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_pca_centroids(pca_df: pd.DataFrame, explained_var,
                       title:str,hue_col:str,out_path: str) -> None:
    """
    Plots per-group centroids with std ellipses. 
    Summarizes 60k points into interpretable group positions.
    """
    _, ax = plt.subplots(figsize=(12, 9))

    groups = pca_df[hue_col].unique()
    palette = sns.color_palette('tab20', len(groups))

    for color, group in zip(palette, groups):
        subset = pca_df[pca_df[hue_col] == group]
        mx, my = subset['PC1'].mean(), subset['PC2'].mean()
        sx, sy = subset['PC1'].std(), subset['PC2'].std()

        # Centroid
        ax.scatter(mx, my, color=color, s=80, zorder=3, label=group)
        # 1-std ellipse
        ellipse = Ellipse((mx, my), width=2*sx, height=2*sy,
                          color=color, alpha=0.2, zorder=2)
        ax.add_patch(ellipse)

    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metabo_histogram(metabo_data:pd.DataFrame,
                          title:str,
                          figname:str)->None:
    """
    Plots a histogram with KDE overlay of all metabolite values.

    Flattens the entire DataFrame into a 1D array before plotting,
    so this shows the global value distribution across all features and samples.
    """
    metabo_values = metabo_data.to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.histplot(data=metabo_values.flatten(), kde=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figname)
    plt.close(fig)

    return

