"""
Classical models (Random Forest and XGBoost) for SPI forecasting.
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from train_model import wi, rmse, mae, nse, bias


def create_model_optimized(model_name, dataset_size=None):
    """
    Creates model with parameters adjusted to dataset size.
    """
    
    if model_name == "RF":
        if dataset_size and dataset_size > 50000:
            return RandomForestRegressor(
                n_estimators=80, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, max_samples=0.5, n_jobs=-12,
                random_state=123, verbose=0,
            )
        elif dataset_size and dataset_size > 10000:
            return RandomForestRegressor(
                n_estimators=120, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_samples=0.7, n_jobs=-12,
                random_state=123, verbose=0,
            )
        else:
            return RandomForestRegressor(
                n_estimators=150, max_depth=25, min_samples_split=2,
                min_samples_leaf=1, n_jobs=-1, random_state=123, verbose=0,
            )

    elif model_name == "XGBoost":
        if dataset_size and dataset_size > 50000:
            return XGBRegressor(
                n_estimators=600, learning_rate=0.03, max_depth=5,
                subsample=0.6, colsample_bytree=0.6, reg_lambda=2.0,
                tree_method="hist", n_jobs=-1, random_state=123, verbosity=0,
            )
        else:
            return XGBRegressor(
                n_estimators=800, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                tree_method="hist", n_jobs=-1, random_state=123, verbosity=0,
            )

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def train_classic_onestep_optimized(model_name, X_train, Y_train_seq):
    """
    Trains model for one-step forecasting.
    """
    if len(X_train) == 0:
        return None

    model = create_model_optimized(model_name, dataset_size=len(X_train))
    y_train = Y_train_seq[:, 0]
    
    model.fit(X_train, y_train)
    return model


def update_window(current_window, spi_pred):
    """
    Updates temporal window for autoregressive forecasting.
    
    Args:
        current_window: [P, 3] (PR, SPI, ΔSPI)
        spi_pred: Predicted SPI value
    
    Returns:
        Updated window
    """
    P = current_window.shape[0]
    new_row = np.zeros((1, 3), dtype=np.float32)

    if P >= 2:
        pr_last = current_window[-1, 0]
        pr_prev = current_window[-2, 0]
        new_row[0, 0] = 0.7 * pr_last + 0.3 * pr_prev
    else:
        new_row[0, 0] = current_window[-1, 0]

    new_row[0, 1] = spi_pred
    new_row[0, 2] = spi_pred - current_window[-1, 1]

    return np.vstack([current_window[1:], new_row])


def forecast_autoregressive(model, X_init, P, Q):
    """
    Generates multi-horizon forecasts via rolling.
    
    Args:
        X_init: [N, 3P]
        P: Historical steps
        Q: Horizon
    
    Returns:
        preds_all: [N, Q]
    """
    N = X_init.shape[0]
    preds_all = np.zeros((N, Q), dtype=np.float32)

    for i in range(N):
        window = X_init[i].reshape(P, 3).copy()

        for q in range(Q):
            features = window.reshape(1, -1)
            spi_pred = model.predict(features)[0]
            preds_all[i, q] = spi_pred
            window = update_window(window, spi_pred)

    return preds_all


def evaluate_autoregressive(model, X_val, Y_val_seq, P, Q):
    """
    Evaluates model with autoregressive forecasting.
    """
    if len(X_val) == 0:
        return {
            "wi": np.nan, "rmse": np.nan, "mae": np.nan,
            "nse": np.nan, "bias": np.nan, "wi_by_h": [np.nan] * Q,
        }

    Y_pred_seq = forecast_autoregressive(model, X_val, P, Q)

    # Global metrics
    yt_all = Y_val_seq.reshape(-1)
    yp_all = Y_pred_seq.reshape(-1)

    mask = ~np.isnan(yt_all) & ~np.isnan(yp_all)
    yt_t = torch.tensor(yt_all[mask], dtype=torch.float32)
    yp_t = torch.tensor(yp_all[mask], dtype=torch.float32)

    metrics = {
        "wi": float(wi(yt_t, yp_t)),
        "rmse": float(rmse(yt_t, yp_t)),
        "mae": float(mae(yt_t, yp_t)),
        "nse": float(nse(yt_t, yp_t)),
        "bias": float(bias(yt_t, yp_t)),
    }

    # WI by horizon
    wi_by_h = []
    for h in range(Q):
        yt = Y_val_seq[:, h]
        yp = Y_pred_seq[:, h]
        mask = ~np.isnan(yt) & ~np.isnan(yp)

        if np.sum(mask) > 10:
            yt_t = torch.tensor(yt[mask], dtype=torch.float32)
            yp_t = torch.tensor(yp[mask], dtype=torch.float32)
            wi_by_h.append(float(wi(yt_t, yp_t)))
        else:
            wi_by_h.append(np.nan)

    metrics["wi_by_h"] = wi_by_h
    return metrics


def run_classic(model_name, X_train, Y_train_seq, X_val, Y_val_seq, P, Q):
    """
    Complete pipeline for classical models.
    """
    model = train_classic_onestep_optimized(model_name, X_train, Y_train_seq)

    if model is None:
        return {"model_name": model_name, "P": P, "Q": Q, "metrics": {}, "model": None}

    metrics = evaluate_autoregressive(model, X_val, Y_val_seq, P, Q)

    return {"model_name": model_name, "P": P, "Q": Q, "metrics": metrics, "model": model}