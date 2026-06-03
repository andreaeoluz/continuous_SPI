# model_classic.py - Classical ML models for drought forecasting (RF and XGBoost)

import numpy as np
import torch
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from config import RF_SPACE, XGB_SPACE, RANDOM_SEED, CLASSIC_PARAMS
from metrics import wi, rmse, mae

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='A column-vector y was passed')


def randomized_search(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                      n_iter: int = None, cv: int = None):
    """
    Perform randomized hyperparameter search with time series cross-validation.

    Args:
        model_name: "RF" or "XGBoost"
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples, n_outputs)
        n_iter: Number of parameter combinations to try
        cv: Number of cross-validation folds

    Returns:
        tuple: (best_estimator, best_params, cv_results)
    """
    n_iter = n_iter or CLASSIC_PARAMS["n_iter"]
    cv = cv or CLASSIC_PARAMS["cv"]

    if len(X_train) == 0:
        raise ValueError("X_train is empty. Cannot run randomized_search.")

    # Adjust CV splits when data is scarce
    n_splits = max(2, min(cv, len(X_train) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if model_name == "RF":
        model = RandomForestRegressor(
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        param_dist = RF_SPACE

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=min(n_iter, 50),
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            random_state=RANDOM_SEED,
            verbose=0,
            error_score="raise",
            refit=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_, search.cv_results_

    elif model_name == "XGBoost":
        base_model = XGBRegressor(
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=0
        )
        model = MultiOutputRegressor(base_model, n_jobs=1)

        param_dist = {f"estimator__{k}": v for k, v in XGB_SPACE.items()}

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=min(n_iter, 50),
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            random_state=RANDOM_SEED,
            verbose=0,
            error_score="raise",
            refit=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_, search.cv_results_

    else:
        raise ValueError(f"Invalid model_name: {model_name}. Use 'RF' or 'XGBoost'.")


def evaluate(model, X: np.ndarray, Y_seq: np.ndarray, Q: int) -> dict:
    """
    Evaluate model predictions using multiple metrics.

    Args:
        model: Trained sklearn model
        X: Input features
        Y_seq: Ground truth targets (n_samples, Q) or (n_samples,)
        Q: Number of forecast horizons

    Returns:
        dict: Metrics including WI, RMSE, MAE, and per-horizon WI
    """
    if len(X) == 0:
        return {"wi": np.nan, "rmse": np.nan, "mae": np.nan, "wi_by_h": [np.nan] * Q}

    Y_pred = model.predict(X)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    if Y_seq.ndim == 1:
        Y_seq = Y_seq.reshape(-1, 1)

    # Flatten for global metrics
    yt_all, yp_all = Y_seq.reshape(-1), Y_pred.reshape(-1)
    mask = ~np.isnan(yt_all) & ~np.isnan(yp_all)
    yt_t = torch.tensor(yt_all[mask], dtype=torch.float32)
    yp_t = torch.tensor(yp_all[mask], dtype=torch.float32)

    metrics = {
        "wi": float(wi(yt_t, yp_t)),
        "rmse": float(rmse(yt_t, yp_t)),
        "mae": float(mae(yt_t, yp_t)),
        "wi_by_h": []
    }

    # Per-horizon metrics
    for h in range(Q):
        if h < Y_seq.shape[1] and h < Y_pred.shape[1]:
            yt = Y_seq[:, h] if Y_seq.ndim > 1 else Y_seq
            yp = Y_pred[:, h] if Y_pred.ndim > 1 else Y_pred
            mask = ~np.isnan(yt) & ~np.isnan(yp)
            if np.sum(mask) > 10:
                yt_t = torch.tensor(yt[mask], dtype=torch.float32)
                yp_t = torch.tensor(yp[mask], dtype=torch.float32)
                metrics["wi_by_h"].append(float(wi(yt_t, yp_t)))
            else:
                metrics["wi_by_h"].append(np.nan)
        else:
            metrics["wi_by_h"].append(np.nan)

    return metrics


def evaluate_with_fallback(model, X_test: np.ndarray, Y_test: np.ndarray,
                           X_val: np.ndarray, Y_val: np.ndarray,
                           Q: int, min_samples: int = 1, test_source: str = "test") -> dict:
    """
    Evaluate model using test data if available, otherwise fall back to validation.

    Args:
        model: Trained model
        X_test, Y_test: Test data
        X_val, Y_val: Validation data
        Q: Number of forecast horizons
        min_samples: Minimum required samples for evaluation
        test_source: Source identifier for logging

    Returns:
        dict: Evaluation metrics with metadata
    """
    if len(X_test) >= min_samples and test_source == "test":
        metrics = evaluate(model, X_test, Y_test, Q)
        metrics["test_source"] = test_source
        metrics["n_test_samples"] = len(X_test)
        return metrics
    elif len(X_val) >= min_samples:
        metrics = evaluate(model, X_val, Y_val, Q)
        metrics["test_source"] = "validation_as_test"
        metrics["n_test_samples"] = len(X_val)
        return metrics
    else:
        return {
            "wi": np.nan, "rmse": np.nan, "mae": np.nan,
            "wi_by_h": [np.nan] * Q, "test_source": "none", "n_test_samples": 0
        }


def run_classic(model_name: str, X_train: np.ndarray, Y_train_seq: np.ndarray,
                X_val: np.ndarray, Y_val_seq: np.ndarray,
                P: int, Q: int, n_iter: int = None, cv: int = None) -> dict:
    """
    Train and evaluate a classical ML model.

    Args:
        model_name: "RF" or "XGBoost"
        X_train, Y_train_seq: Training data
        X_val, Y_val_seq: Validation data
        P: Input sequence length (for metadata)
        Q: Output sequence length
        n_iter: Hyperparameter search iterations
        cv: Cross-validation folds

    Returns:
        dict: Results containing model, metrics, and best parameters
    """
    if len(X_train) == 0:
        return {"model_name": model_name, "P": P, "Q": Q, "val_metrics": {}, "model": None}

    model, best_params, cv_results = randomized_search(model_name, X_train, Y_train_seq, n_iter, cv)

    return {
        "model_name": model_name,
        "P": P,
        "Q": Q,
        "val_metrics": evaluate(model, X_val, Y_val_seq, Q),
        "model": model,
        "best_params": best_params,
        "cv_results": cv_results
    }


def predict_multioutput(model, X: np.ndarray, Q: int) -> np.ndarray:
    """
    Generate multi-step predictions using a trained model.

    Args:
        model: Trained sklearn model
        X: Input features
        Q: Number of output steps (for shape consistency)

    Returns:
        np.ndarray: Predictions with shape (n_samples, Q)
    """
    predictions = model.predict(X)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    return predictions.astype(np.float32)