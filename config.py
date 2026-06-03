# config.py - Centralized configuration for drought forecasting experiments

import torch
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path("EXPERIMENTS")
DATA_PATH = "data/pr_Area1.xlsx"

METRICS_DIR = BASE_DIR / "metrics"
CURVES_DIR = BASE_DIR / "curves"

def ensure_directories() -> None:
    """Create necessary experiment directories if they don't exist."""
    for d in [BASE_DIR, METRICS_DIR, CURVES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

ensure_directories()

# ============================================================================
# DATA SPLIT & TEMPORAL CONFIGURATION
# ============================================================================
TRAIN_END_YEAR = 2018
REF_DATE = "2024-12"
SPI_SCALE_FIXED = 3

# GRID SEARCH
P_VALUES = [3, 6, 9, 12]
Q_VALUES = [1, 3, 6, 9, 12]

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONVLSTM3D HYPERPARAMETERS
# ============================================================================
CONVLSTM3D_PARAMS = {
    "batch_size": 16,
    "epochs": 100,
    "lr": 0.0001,
    "hidden": (64, 32, 16),
    "dropout": 0.3,
    "patience": 10,
    "use_checkpoint": True,
}

# ============================================================================
# CLASSICAL MODELS (RF, XGBoost) CONFIGURATION
# ============================================================================
CLASSIC_PARAMS = {
    "n_iter": 100,           # Number of iterations for random search
    "cv": 5,                 # Cross-validation folds
    "sampling_rate": 0.1,    # Fraction of spatial pixels to sample
    "max_samples": 10000,    # Maximum total samples to generate
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
MIN_TEST_SAMPLES = 1
USE_VAL_AS_TEST_FALLBACK = True
EVAL_MODE = "last"           # "last" or "best"

# ============================================================================
# REPRODUCIBILITY & VISUALIZATION
# ============================================================================
RANDOM_SEED = 123
GENERATE_VISUALIZATIONS = True

# ============================================================================
# RANDOM FOREST HYPERPARAMETER SPACE
# ============================================================================
RF_SPACE = {
    "n_estimators": [200, 300, 500, 800, 1200],
    "max_depth": [None, 5, 8, 12, 16, 24, 32],
    "min_samples_split": [2, 5, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4, 8, 12],
    "max_features": [1.0, "sqrt", 0.3, 0.5, 0.7],
    "bootstrap": [True],
    "max_samples": [None, 0.5, 0.7, 0.9]
}

# ============================================================================
# XGBOOST HYPERPARAMETER SPACE
# ============================================================================
XGB_SPACE = {
    "n_estimators": [200, 400, 600, 800, 1200],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_child_weight": [1, 2, 4, 6, 8, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.01, 0.1, 0.3, 1, 3],
    "reg_alpha": [0, 1e-3, 1e-2, 0.1, 1],
    "reg_lambda": [0.5, 1, 2, 5, 10]
}