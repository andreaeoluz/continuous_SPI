# metrics.py - Evaluation metrics for drought forecasting

import torch
import numpy as np


def wi(yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """
    Willmott's Index of Agreement (WI).

    Measures how well predictions match observations, ranging from 0 to 1,
    where 1 indicates perfect agreement.

    Args:
        yt: Ground truth values
        yp: Predicted values

    Returns:
        WI value as a torch.Tensor (scalar)
    """
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if yt.numel() == 0:
        return torch.tensor(float("nan"))

    ybar = yt.mean()
    sse = ((yt - yp) ** 2).sum()
    denom = ((yp - ybar).abs() + (yt - ybar).abs()).pow(2).sum()

    return 1 - sse / denom


def rmse(yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Square Error (RMSE).

    Args:
        yt: Ground truth values
        yp: Predicted values

    Returns:
        RMSE value as a torch.Tensor (scalar)
    """
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if yt.numel() == 0:
        return torch.tensor(float("nan"))

    return torch.sqrt(torch.mean((yt - yp) ** 2))


def mae(yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (MAE).

    Args:
        yt: Ground truth values
        yp: Predicted values

    Returns:
        MAE value as a torch.Tensor (scalar)
    """
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if yt.numel() == 0:
        return torch.tensor(float("nan"))

    return torch.mean(torch.abs(yt - yp))


def compute_all_metrics(yt: torch.Tensor, yp: torch.Tensor) -> dict:
    """
    Compute all three metrics (WI, RMSE, MAE) at once.

    Args:
        yt: Ground truth values
        yp: Predicted values

    Returns:
        Dictionary with keys: "wi", "rmse", "mae"
    """
    return {
        "wi": float(wi(yt, yp)),
        "rmse": float(rmse(yt, yp)),
        "mae": float(mae(yt, yp))
    }


def select_eval_mode(metric_h: list, mode: str = "last") -> float:
    """
    Aggregate per-horizon metrics according to evaluation mode.

    Args:
        metric_h: List of metric values per horizon
        mode: Aggregation mode - "last", "best_of_h", or "mean"

    Returns:
        Aggregated metric value
    """
    arr = np.array(metric_h)

    if len(arr) == 0:
        return np.nan

    if mode == "last":
        return arr[-1]
    if mode == "best_of_h":
        return np.nanmax(arr)

    return np.nanmean(arr)  # "mean" mode (default fallback)