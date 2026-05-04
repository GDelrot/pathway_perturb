# worker_cell_line.py
"""
SLURM array worker.
Called with:  python worker_cell_line.py <array_task_id> <cell_lines_file>
Processes a single cell line, writes its own output files.
"""
import sys
import gc
import logging
import time
import psutil
from pathlib import Path

from __loader__ import LINCSpaths, CCLEpaths, Loader
from __viz__      import plot_moa_correlations
from __analysis__ import run_moa_signature_correlations_chunked

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

INPUT    = Path('/home/gdelrot/pathway_perturb/data')
OUTPUT   = Path('/home/gdelrot/pathway_perturb/outputs/corr_cl_subsample')
MNT_L1000 = Path('/mnt/cbib/l1000/data/')

OUTPUT.mkdir(parents=True, exist_ok=True)

def log_memory(tag: str, t0: float ) -> float:
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
    elapsed = f"  +{time.time()-t0:.1f}s" if t0 else ""
    logger.info(f"[MEM {tag}] RSS = {rss_gb:.2f} GB{elapsed}")
    return time.time()

def resolve_cell_line(task_id: int, cell_lines_file: Path) -> str:
    """Map SLURM_ARRAY_TASK_ID (0-based) to a cell line name."""
    lines = cell_lines_file.read_text().splitlines()
    return lines[task_id]


def build_loader() -> Loader:

    lincs_sigcom = LINCSpaths(
        gctx=str(MNT_L1000 / 'cp_coeff_mat.gctx'),
        pathway=str(INPUT / 'gsea_l1000.parquet'),
        sig_info=str(MNT_L1000 / 'siginfo_beta.txt'),
        gene_info=str(MNT_L1000 / 'geneinfo_beta.txt.gz'),
        cell_info=str(MNT_L1000 / 'cellinfo_beta.txt.gz'),
        compound_info=str(MNT_L1000 / 'compoundinfo_beta.txt.gz'),
        inst_info=str(MNT_L1000 / 'instinfo_beta.txt.gz'),
    )
    ccle_data = CCLEpaths(
        transcriptomics=str(INPUT / 'CCLE_RNAseq_genes_rpkm_20180929.gct'),
        metabolomics=str(INPUT / 'CCLE_metabolomics_20190502.csv'),
        cell_annotations=str(INPUT / 'Cell_lines_annotations_20181226.txt'),
        metabo_mapping=str(INPUT / 'metabo_mapping.csv'),
        depmap_annotation=str(INPUT / 'Depmap_annotation.csv'),
    )
    loader = Loader(lincs_paths=lincs_sigcom, ccle_paths=ccle_data)
    loader.load_l1000_metadata()

    # Gene level signatures
    loader.extract_data_subset(n_subset=None)
    loader.l1000_exp_data = loader.l1000_exp_data.dropna(axis=1) # Already float32

    # Pathway level signatures
    loader.load_l1000_pathway_scores()
    loader.l1000_pathway_data = loader.l1000_pathway_data.dropna(axis=1)
    float_cols = loader.l1000_pathway_data.select_dtypes('float64').columns
    loader.l1000_pathway_data[float_cols] = (
        loader.l1000_pathway_data[float_cols].astype('float32')
    )
    return loader


def process_cell_line(loader: Loader, cl: str) -> None:
    
    for pert_type in ['raw_pert','pathway_pert']:
        if pert_type == 'raw_pert':
            pert_data = loader.l1000_exp_data
        elif pert_type == 'pathway_pert':
            pert_data = loader.l1000_pathway_data
        else:
            raise ValueError('Needed a value in ["raw_pert","pathway_pert"]')

        for metadata in ['moa', 'cluster_id']:
            outname = 'smiles' if metadata == 'cluster_id' else 'moa'

            for cor_type in ['pearson', 'spearman']:
                logger.info('Working with parameters: %s|%s|%s|%s',
                 pert_type,metadata,cl,cor_type)
                summary_row, corr_dict = run_moa_signature_correlations_chunked(
                    perturbations=pert_data,
                    l1000_obj=loader,
                    cell_line=cl,
                    metadata=metadata,
                    cor_type=cor_type,
                    max_pairs= 50 # Mode of the distribution of generated pairs across all cell lines per moa
                )
                if summary_row is None:
                    logger.warning(f"No data for {cl} / {metadata} / {cor_type} / {pert_type}")
                    continue

                # ── persist this cell line's summary immediately ──
                out_csv = OUTPUT / f'correlations_{outname}_{cor_type}_{cl}_{pert_type}.csv'
                summary_row.to_csv(out_csv, index=False)

                # plot_moa_correlations(
                #     median_cor=summary_row,
                #     cell_line=cl,
                #     metadata=metadata,
                #     figname=str(OUTPUT / f'corplot_{outname}_{cor_type}_{cl}_{pert_type}.png'),
                # )
                del summary_row, corr_dict
                gc.collect()
        if pert_type == 'raw_pert':
            del loader.l1000_exp_data
        elif pert_type =='pathway_pert':
            del loader.l1000_pathway_data
        del pert_data
        gc.collect()
                


def main() -> None:
    task_id        = int(sys.argv[1])               # from $SLURM_ARRAY_TASK_ID
    cell_lines_file = Path(sys.argv[2])             # path to cell_lines.txt

    cl = resolve_cell_line(task_id, cell_lines_file)
    logger.info(f"Task {task_id} → cell line: {cl}")

    loader = build_loader()
    process_cell_line(loader, cl)
    logger.info(f"Done: {cl}")


if __name__ == '__main__':
    main()