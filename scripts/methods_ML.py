"""
methods_ML.py — ML methods for pathway-level drug perturbation prediction
==========================================================================

CONCEPTUAL OVERVIEW
-------------------
We want to learn the mapping:

    f(basal_cell_state, drug) → pathway_perturbation

Where:
    - basal_cell_state = transcriptomic + metabolomic pathway scores (CCLE)
    - drug             = encoded drug identity
    - perturbation     = pathway-level response from L1000

ARCHITECTURE CHOICE: Multi-Output Regression
---------------------------------------------
Since we predict MULTIPLE pathways simultaneously, we use MultiOutputRegressor
or natively multi-output models (like RandomForest). Why?

    Option A: One model per pathway (independent models)
        → Ignores correlations between pathways
        → But simple and parallelizable

    Option B: Multi-output (one model, all pathways at once)
        → Can capture pathway-pathway correlations (for some models)
        → Single training call

We implement BOTH options depending on the model.
RandomForest/GradientBoosting are natively multi-output.
Ridge/ElasticNet need MultiOutputRegressor wrapper.

SPLIT STRATEGIES
----------------
    - 'random'   : classic random split (optimistic, good for debugging)
    - 'cell_line': hold out entire cell lines (tests generalization to new cells)
    - 'drug'     : hold out entire drugs (tests generalization to new drugs)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── Sklearn imports ──────────────────────────────────────────────────────
# TIP: import only what you need — keeps namespace clean and helps you
# understand exactly which tools you're using.
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    cross_val_score,
    KFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
import warnings
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PreparedData:
    """
    Container for ML-ready data.

    WHY a dataclass?
    -----------------
    Instead of passing 10 separate variables around, we bundle everything
    into one object. This is cleaner than a dict (you get autocomplete,
    type hints) and lighter than a full class.

    Attributes
    ----------
    X_train, X_test : np.ndarray
        Feature matrices [n_samples, n_features]
    y_train, y_test : np.ndarray
        Target matrices [n_samples, n_pathways]
    feature_names   : list of str
    pathway_names   : list of str
    train_idx, test_idx : np.ndarray
        Original indices (useful for tracing back to cell lines / drugs)
    split_strategy  : str
        Which split was used ('random', 'cell_line', 'drug')
    drug_encoder    : object
        Fitted encoder for drugs (needed at inference time)
    scaler          : object
        Fitted scaler for features
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    pathway_names: list
    train_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    split_strategy: str = "random"
    drug_encoder: object = None
    scaler: object = None

def prepare_features(
    l1000_df: pd.DataFrame,
    ccle_transcriptomics: pd.DataFrame,
    ccle_metabolomics: pd.DataFrame,
    cell_col: str = "cell_id",
    drug_col: str = "drug_id",
    drug_encoding: str = "onehot",
    max_drug_categories: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, list, list, object]:
    """
    Build the feature matrix X and target matrix Y from raw datasets.

    STEP-BY-STEP LOGIC
    -------------------
    1. Identify pathway columns (common across datasets)
    2. For each L1000 observation (cell_i, drug_j):
       - Look up basal transcriptomic pathways from CCLE → features
       - Look up basal metabolomic pathways from CCLE  → features
       - Encode drug_j                                 → features
       - L1000 pathway scores                          → targets
    3. Concatenate all features → X
    4. L1000 pathway scores    → Y

    Parameters
    ----------
    l1000_df : pd.DataFrame
        Perturbation data. Must contain `cell_col`, `drug_col`, and pathway columns.
    ccle_transcriptomics : pd.DataFrame
        Rows = cell lines, Columns = pathway scores. Index = cell line IDs.
    ccle_metabolomics : pd.DataFrame
        Same structure as above.
    cell_col, drug_col : str
        Column names in l1000_df identifying cell lines and drugs.
    drug_encoding : str
        'onehot' or 'label'. Onehot is better for linear models;
        label encoding is more memory-efficient for tree models.
    max_drug_categories : int
        If #drugs > this, fall back to label encoding (onehot would explode
        the feature space — e.g., 5000 drugs = 5000 extra columns).

    Returns
    -------
    X_df, Y_df, feature_names, pathway_names, drug_encoder
    """
    logger.info("Preparing features...")

    # ── 1. Find common pathway columns ──────────────────────────────────
    # PEDAGOGIC NOTE: We need columns that are pathway scores in ALL datasets.
    # Metadata columns (cell_id, drug_id, dose, etc.) must be excluded.
    metadata_cols = {cell_col, drug_col, "dose", "time", "pert_type", "sig_id",
                     "pert_id", "pert_iname", "cell_mfc_name", "det_plate"}

    l1000_pathway_cols = [c for c in l1000_df.columns if c not in metadata_cols]
    trans_pathway_cols = list(ccle_transcriptomics.columns)
    metab_pathway_cols = list(ccle_metabolomics.columns)

    # Intersection: only pathways measured in ALL three datasets
    common_pathways = sorted(
        set(l1000_pathway_cols) & set(trans_pathway_cols) & set(metab_pathway_cols)
    )
    logger.info(f"  Common pathways across all datasets: {len(common_pathways)}")

    if len(common_pathways) == 0:
        raise ValueError(
            "No common pathway columns found! Check that your column names match "
            "across datasets (e.g., 'HALLMARK_APOPTOSIS' vs 'Apoptosis')."
        )

    # ── 2. Filter L1000 to cell lines present in BOTH CCLE datasets ────
    common_cells = (
        set(l1000_df[cell_col].unique())
        & set(ccle_transcriptomics.index)
        & set(ccle_metabolomics.index)
    )
    logger.info(f"  Cell lines in all 3 datasets: {len(common_cells)}")

    l1000_filtered = l1000_df[l1000_df[cell_col].isin(common_cells)].copy()
    logger.info(f"  L1000 observations after cell-line filtering: {len(l1000_filtered)}")

    # ── 3. Build basal feature matrix ───────────────────────────────────
    # For each L1000 row, look up the cell line's basal state.
    # This is a LEFT JOIN on cell_id.
    #
    # WHY .loc[cell_ids] and .values?
    # → We align CCLE rows to match L1000 row order via the cell_id column.
    #   .values strips the index so we get a plain numpy array for stacking.

    cell_ids = l1000_filtered[cell_col].values

    basal_trans = ccle_transcriptomics.loc[cell_ids, common_pathways].values
    basal_metab = ccle_metabolomics.loc[cell_ids, common_pathways].values

    # Feature names: prefix to distinguish source
    trans_feat_names = [f"trans_{p}" for p in common_pathways]
    metab_feat_names = [f"metab_{p}" for p in common_pathways]

    # ── 4. Encode drugs ─────────────────────────────────────────────────
    # PEDAGOGIC  on encoding strategies:
    #
    # ONE-HOT: Each drug gets its own binary column [0,0,1,0,...,0]
    #   Pros: No ordinal assumption, works well with linear models
    #   Cons: HUGE feature space if many drugs (5k drugs = 5k columns)
    #
    # LABEL ENCODING: Each drug gets a single integer (drug_A=0, drug_B=1, ...)
    #   Pros: Compact, memory efficient
    #   Cons: Implies ordinal relationship (drug 5 > drug 3?!) — BAD for linear
    #         models, OK for tree-based models (they split on thresholds anyway)
    #
    # DECISION: If #drugs is manageable, use onehot. Otherwise, label.

    n_drugs = l1000_filtered[drug_col].nunique()
    logger.info(f"  Unique drugs: {n_drugs}")

    if drug_encoding == "onehot" and n_drugs <= max_drug_categories:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        drug_features = encoder.fit_transform(l1000_filtered[[drug_col]])
        drug_feat_names = [f"drug_{c}" for c in encoder.get_feature_names_out()]
        logger.info(f"  Drug encoding: OneHot → {drug_features.shape[1]} features")
    else:
        if drug_encoding == "onehot":
            logger.info(
                f"  Too many drugs ({n_drugs}) for OneHot — falling back to label encoding. "
                f"Consider using drug fingerprints for a better representation."
            )
        encoder = LabelEncoder()
        drug_features = encoder.fit_transform(l1000_filtered[drug_col]).reshape(-1, 1)
        drug_feat_names = ["drug_label"]
        logger.info(f"  Drug encoding: Label → 1 feature")

    # ── 5. Assemble X and Y ─────────────────────────────────────────────
    X = np.hstack([basal_trans, basal_metab, drug_features])
    feature_names = trans_feat_names + metab_feat_names + drug_feat_names

    Y = l1000_filtered[common_pathways].values
    pathway_names = common_pathways

    # Store as DataFrames for traceability
    X_df = pd.DataFrame(X, columns=feature_names, index=l1000_filtered.index)
    Y_df = pd.DataFrame(Y, columns=pathway_names, index=l1000_filtered.index)

    # Attach metadata for splitting later
    X_df["__cell_id__"] = cell_ids
    X_df["__drug_id__"] = l1000_filtered[drug_col].values

    logger.info(f"  Final X shape: {X_df.shape}  |  Final Y shape: {Y_df.shape}")

    return X_df, Y_df, feature_names, pathway_names, encoder


def split_data(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    feature_names: list,
    pathway_names: list,
    drug_encoder: object,
    strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> PreparedData:
    """
    Split data into train/test with different strategies.

    PEDAGOGIC NOTE: Why different strategies matter
    ------------------------------------------------
    Imagine your model sees MCF7 (breast cancer cell line) in training.
    In a random split, MCF7 appears in BOTH train and test.
    The model might just memorize "MCF7 + drug_X → this response"
    without learning generalizable biology.

    A CELL-LINE HOLDOUT forces the model to predict for cell lines
    it has NEVER seen — this is much harder but more realistic
    (you want to predict for a NEW patient's tumor, after all).

    Parameters
    ----------
    strategy : str
        'random'    → standard random split
        'cell_line' → hold out entire cell lines
        'drug'      → hold out entire drugs
    scale : bool
        Whether to StandardScale features. Almost always yes for linear
        models (Ridge, ElasticNet). Less critical for trees but doesn't hurt.
    """
    logger.info(f"Splitting data with strategy='{strategy}', test_size={test_size}")

    # Extract metadata then drop it from features
    cell_ids = X_df["__cell_id__"].values
    drug_ids = X_df["__drug_id__"].values
    X_clean = X_df[feature_names].values
    Y = Y_df.values

    if strategy == "random":
        # ── Simple random split ─────────────────────────────────────
        # Every (cell, drug) pair is independently assigned to train or test.
        # Fast, easy, but OPTIMISTIC because same cell line leaks across splits.
        train_idx, test_idx = train_test_split(
            np.arange(len(X_clean)), test_size=test_size, random_state=random_state
        )

    elif strategy == "cell_line":
        # ── Cell-line holdout ───────────────────────────────────────
        # ALL observations from held-out cell lines go to test.
        # This prevents the model from memorizing cell-line-specific patterns.
        unique_cells = np.unique(cell_ids)
        train_cells, test_cells = train_test_split(
            unique_cells, test_size=test_size, random_state=random_state
        )
        train_idx = np.where(np.isin(cell_ids, train_cells))[0]
        test_idx = np.where(np.isin(cell_ids, test_cells))[0]
        logger.info(
            f"  Train cell lines: {len(train_cells)} | Test cell lines: {len(test_cells)}"
        )

    elif strategy == "drug":
        # ── Drug holdout ────────────────────────────────────────────
        # ALL observations for held-out drugs go to test.
        # Tests: "Can the model predict the effect of a NEVER-SEEN drug?"
        # NOTE: This is typically the HARDEST task and might require
        # richer drug representations (fingerprints, SMILES embeddings).
        unique_drugs = np.unique(drug_ids)
        train_drugs, test_drugs = train_test_split(
            unique_drugs, test_size=test_size, random_state=random_state
        )
        train_idx = np.where(np.isin(drug_ids, train_drugs))[0]
        test_idx = np.where(np.isin(drug_ids, test_drugs))[0]
        logger.info(
            f"  Train drugs: {len(train_drugs)} | Test drugs: {len(test_drugs)}"
        )

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'random', 'cell_line', or 'drug'.")

    X_train, X_test = X_clean[train_idx], X_clean[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]

    logger.info(f"  Train size: {len(train_idx)} | Test size: {len(test_idx)}")

    # ── Scaling ─────────────────────────────────────────────────────
    # WHY: Linear models (Ridge, ElasticNet) are sensitive to feature scale.
    # A pathway scored 0-1 and another scored 0-1000 would bias the model.
    # StandardScaler: z = (x - mean) / std → all features ~ N(0,1)
    #
    # CRITICAL: fit on TRAIN only, then transform both train and test.
    # Fitting on test would leak future information into your model.
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)   # fit + transform on train
        X_test = scaler.transform(X_test)          # transform only on test
        logger.info("  Features scaled (StandardScaler)")

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        pathway_names=pathway_names,
        train_idx=train_idx,
        test_idx=test_idx,
        split_strategy=strategy,
        drug_encoder=drug_encoder,
        scaler=scaler,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_models(n_targets: int = 1) -> dict:
    """
    Return a dictionary of named ML models ready for multi-output regression.

    PEDAGOGIC NOTES ON EACH MODEL
    ──────────────────────────────

    RIDGE REGRESSION (L2 regularization)
    → Adds penalty: loss = MSE + alpha * Σ(w²)
    → Shrinks ALL coefficients toward zero but never TO zero
    → Great for correlated features (pathway scores ARE correlated)
    → alpha=1.0 is a reasonable default; tune via CV

    ELASTICNET (L1 + L2 regularization)
    → loss = MSE + alpha * [l1_ratio * Σ|w| + (1-l1_ratio) * Σ(w²)]
    → l1_ratio=0.5 → balanced L1/L2
    → L1 component can zero out irrelevant features → built-in feature selection
    → Useful if you suspect many pathways are irrelevant as features

    RANDOM FOREST
    → Ensemble of decision trees, each trained on a bootstrap sample
    → Natively handles multi-output (one tree predicts all pathways)
    → Captures non-linear relationships and interactions
    → n_estimators=200: more trees = more stable predictions (diminishing returns >500)
    → max_depth=15: prevents overfitting (deep trees memorize training data)

    GRADIENT BOOSTING
    → Builds trees SEQUENTIALLY, each correcting the previous one's errors
    → Generally more accurate than RF but slower and easier to overfit
    → NOT natively multi-output → needs MultiOutputRegressor wrapper
    → n_estimators=200, learning_rate=0.05, max_depth=6: conservative defaults
    """

    models = {
        # ── Linear models (need MultiOutput wrapper) ────────────────
        "Ridge": MultiOutputRegressor(
            Ridge(alpha=1.0, random_state=42)
        ),
        "ElasticNet": MultiOutputRegressor(
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
        ),

        # ── Tree-based models ───────────────────────────────────────
        # RandomForest is NATIVELY multi-output (no wrapper needed)
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,    # regularization: don't split if <5 samples in leaf
            n_jobs=-1,             # use all CPU cores
            random_state=42,
        ),

        # GradientBoosting: sklearn's version is NOT natively multi-output
        # → wrap it. For large datasets, consider HistGradientBoosting (faster).
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,       # stochastic GB: use 80% of data per tree
                random_state=42,
            )
        ),
    }

    return models


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """Stores results for one model."""
    name: str
    model: object
    metrics: dict              # global metrics
    per_pathway_metrics: dict  # per-pathway breakdown
    train_time: float
    predictions: np.ndarray    # y_pred on test set


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pathway_names: list,
) -> tuple[dict, dict]:
    """
    Compute global and per-pathway metrics.

    METRICS EXPLAINED
    -----------------
    MSE  : Mean Squared Error → penalizes large errors heavily (squared!)
    MAE  : Mean Absolute Error → more robust to outliers
    R²   : Coefficient of determination → 1.0 = perfect, 0.0 = predicts mean,
            <0 = worse than predicting the mean (your model is harmful)

    WHY PER-PATHWAY?
    → Global R² can hide the fact that your model predicts apoptosis well
      (R²=0.8) but completely fails on, say, mTOR signaling (R²=-0.1).
    → Always look at per-pathway breakdown!
    """
    # ── Global metrics (averaged across all pathways) ───────────────
    global_metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),   # macro-averaged across outputs
    }

    # ── Per-pathway metrics ─────────────────────────────────────────
    per_pathway = {}
    for i, pw in enumerate(pathway_names):
        per_pathway[pw] = {
            "mse": mean_squared_error(y_true[:, i], y_pred[:, i]),
            "mae": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "r2": r2_score(y_true[:, i], y_pred[:, i]),
        }

    return global_metrics, per_pathway


def train_and_evaluate(
    data: PreparedData,
    models: Optional[dict] = None,
    verbose: bool = True,
) -> dict[str, ModelResult]:
    """
    Train all models and evaluate on test set.

    Parameters
    ----------
    data : PreparedData
        Output of split_data()
    models : dict, optional
        {name: model} dict. If None, uses get_models() defaults.

    Returns
    -------
    results : dict of {model_name: ModelResult}
    """
    if models is None:
        models = get_models(n_targets=data.y_train.shape[1])

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        t0 = time.time()

        # ── FIT ─────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence convergence warnings for first pass
            model.fit(data.X_train, data.y_train)

        train_time = time.time() - t0

        # ── PREDICT ─────────────────────────────────────────────────
        y_pred = model.predict(data.X_test)

        # ── EVALUATE ────────────────────────────────────────────────
        global_metrics, per_pathway = evaluate_predictions(
            data.y_test, y_pred, data.pathway_names
        )

        results[name] = ModelResult(
            name=name,
            model=model,
            metrics=global_metrics,
            per_pathway_metrics=per_pathway,
            train_time=train_time,
            predictions=y_pred,
        )

        if verbose:
            logger.info(
                f"  {name:20s} | R²={global_metrics['r2']:.4f} | "
                f"MSE={global_metrics['mse']:.4f} | "
                f"MAE={global_metrics['mae']:.4f} | "
                f"Time={train_time:.1f}s"
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. RESULTS SUMMARY & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def summarize_results(results: dict[str, ModelResult]) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing all models.
    """
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "R²": res.metrics["r2"],
            "MSE": res.metrics["mse"],
            "MAE": res.metrics["mae"],
            "Train Time (s)": res.train_time,
        })
    df = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    return df


def per_pathway_summary(results: dict[str, ModelResult]) -> pd.DataFrame:
    """
    Build a per-pathway R² comparison across models.

    OUTPUT SHAPE: rows=pathways, columns=models, values=R²

    This is your KEY diagnostic table. Look for:
    → Pathways where ALL models fail (R² < 0): maybe these pathways aren't
      predictable from basal state, or the L1000 signal is too noisy.
    → Pathways where trees >> linear: suggests non-linear biology.
    → Pathways where linear >> trees: simpler relationship, trees are overfitting.
    """
    pw_data = {}
    for name, res in results.items():
        pw_data[name] = {pw: m["r2"] for pw, m in res.per_pathway_metrics.items()}

    df = pd.DataFrame(pw_data)
    df.index.name = "Pathway"
    df = df.sort_values(list(results.keys())[0], ascending=False)
    return df


def get_feature_importances(result: ModelResult, top_n: int = 20) -> Optional[pd.Series]:
    """
    Extract feature importances from tree-based models.

    WHY THIS MATTERS
    -----------------
    Feature importances tell you WHICH basal features drive predictions.
    If 'trans_HALLMARK_APOPTOSIS' is the top feature, it means the basal
    apoptosis state of a cell line is the strongest predictor of drug response.

    NOTE: Only works for RandomForest (natively multi-output).
    For MultiOutputRegressor wrappers, importances are per-estimator
    (one per target pathway), so we average them.
    """
    model = result.model
    feature_names = result.name  # we'll pass feature_names separately

    if isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
    elif isinstance(model, MultiOutputRegressor):
        # Average importances across all sub-estimators
        if hasattr(model.estimators_[0], "feature_importances_"):
            imp_list = [est.feature_importances_ for est in model.estimators_]
            importances = np.mean(imp_list, axis=0)
        else:
            return None
    else:
        return None

    return importances


def feature_importance_df(
    result: ModelResult, feature_names: list, top_n: int = 20
) -> Optional[pd.DataFrame]:
    """Return a DataFrame of top feature importances."""
    importances = get_feature_importances(result)
    if importances is None:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION (more robust than single train/test split)
# ═══════════════════════════════════════════════════════════════════════════

def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_folds: int = 5,
    groups: Optional[np.ndarray] = None,
    scoring: str = "r2",
) -> dict:
    """
    Run k-fold cross-validation.

    PEDAGOGIC NOTE: Why CV over a single split?
    --------------------------------------------
    A single train/test split is ONE dice roll. You might get lucky or unlucky
    depending on which samples end up in test. CV averages over K splits:

        Fold 1: [TEST | train | train | train | train]
        Fold 2: [train | TEST | train | train | train]
        Fold 3: [train | train | TEST | train | train]
        ...

    Mean ± std of scores across folds gives you CONFIDENCE in your estimate.
    If std is huge, your model's performance is unstable.

    GROUPED CV:
    If groups is provided (e.g., cell_line IDs), uses GroupKFold
    → ensures all samples from one group stay together
    → equivalent to the 'cell_line' split strategy but repeated K times

    Parameters
    ----------
    groups : array-like, optional
        Group labels for GroupKFold (e.g., cell_line IDs)
    scoring : str
        Sklearn scoring metric. 'r2' for regression is standard.
        Note: sklearn returns NEGATIVE MSE for 'neg_mean_squared_error'.
    """
    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        # GroupKFold needs groups in the split call
        # cross_val_score doesn't support multi-output natively for all scorers,
        # so we do it manually
        splits = list(cv.split(X, y, groups=groups))
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))

    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale per fold (proper CV protocol)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        import copy
        fold_model = copy.deepcopy(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_tr, y_tr)
        y_pred = fold_model.predict(X_val)

        if scoring == "r2":
            score = r2_score(y_val, y_pred)
        elif scoring == "neg_mse":
            score = -mean_squared_error(y_val, y_pred)
        else:
            score = r2_score(y_val, y_pred)

        fold_scores.append(score)
        logger.info(f"  Fold {fold_i + 1}/{n_folds}: {scoring}={score:.4f}")

    return {
        "mean": np.mean(fold_scores),
        "std": np.std(fold_scores),
        "per_fold": fold_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONVENIENCE: RUN FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    l1000_df: pd.DataFrame,
    ccle_transcriptomics: pd.DataFrame,
    ccle_metabolomics: pd.DataFrame,
    cell_col: str = "cell_id",
    drug_col: str = "drug_id",
    split_strategy: str = "random",
    test_size: float = 0.2,
    run_cv: bool = False,
    cv_folds: int = 5,
) -> tuple[dict, PreparedData, pd.DataFrame]:
    """
    End-to-end pipeline: prepare → split → train → evaluate.

    Returns
    -------
    results      : dict of ModelResult
    data         : PreparedData
    summary_df   : comparison DataFrame
    """
    # Step 1: Prepare features
    X_df, Y_df, feat_names, pw_names, drug_enc = prepare_features(
        l1000_df, ccle_transcriptomics, ccle_metabolomics,
        cell_col=cell_col, drug_col=drug_col,
    )

    # Step 2: Split
    data = split_data(
        X_df, Y_df, feat_names, pw_names, drug_enc,
        strategy=split_strategy, test_size=test_size,
    )

    # Step 3: Train & evaluate
    results = train_and_evaluate(data)

    # Step 4: Summarize
    summary_df = summarize_results(results)
    logger.info("\n" + summary_df.to_string(index=False))

    # Step 5 (optional): Cross-validation
    if run_cv:
        logger.info("\n--- Cross-Validation ---")
        models = get_models()
        for name, model in models.items():
            logger.info(f"CV for {name}:")
            cv_result = cross_validate_model(
                data.X_train, data.y_train, model, n_folds=cv_folds
            )
            logger.info(
                f"  → Mean R²={cv_result['mean']:.4f} ± {cv_result['std']:.4f}"
            )

    return results, data, summary_df