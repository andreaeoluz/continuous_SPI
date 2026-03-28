"""
Training and evaluation of models for multi-horizon forecasting.
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader


# =====================================================================
# Metrics
# =====================================================================

def wi(yt, yp):
    """Willmott Index."""
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.numel() == 0:
        return torch.tensor(float("nan"))
    ybar = yt.mean()
    sse = ((yt - yp) ** 2).sum()
    denom = ((yp - ybar).abs() + (yt - ybar).abs()).pow(2).sum()
    return 1 - sse / denom


def rmse(yt, yp):
    """Root Mean Square Error."""
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.numel() == 0:
        return torch.tensor(float("nan"))
    return torch.sqrt(torch.mean((yt - yp) ** 2))


def mae(yt, yp):
    """Mean Absolute Error."""
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.numel() == 0:
        return torch.tensor(float("nan"))
    return torch.mean(torch.abs(yt - yp))


def nse(yt, yp):
    """Nash-Sutcliffe Efficiency coefficient."""
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.numel() == 0:
        return torch.tensor(float("nan"))
    num = torch.sum((yt - yp) ** 2)
    den = torch.sum((yt - yt.mean()) ** 2)
    return 1 - num / den


def bias(yt, yp):
    """Mean bias."""
    mask = torch.isfinite(yt) & torch.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.numel() == 0:
        return torch.tensor(float("nan"))
    return torch.mean(yp - yt)


# =====================================================================
# Evaluation by horizon
# =====================================================================

def wi_by_horizon(model, loader, Q, device):
    """Calculates WI for each forecast horizon."""
    model.eval()

    preds_h = [[] for _ in range(Q)]
    trues_h = [[] for _ in range(Q)]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model.forecast(x, Q)

            for h in range(Q):
                preds_h[h].append(y_pred[:, h].flatten())
                trues_h[h].append(y[:, h].flatten())

    wi_h = []
    for h in range(Q):
        if len(preds_h[h]) == 0:
            wi_h.append(np.nan)
            continue

        yp = torch.cat(preds_h[h])
        yt = torch.cat(trues_h[h])

        if yp.numel() == 0:
            wi_h.append(np.nan)
        else:
            wi_val = wi(yt, yp)
            wi_h.append(float(wi_val.detach().cpu()))

    return wi_h


def select_eval_mode(wi_h, mode="last"):
    """Selects validation metric according to mode."""
    arr = np.array(wi_h)
    if mode == "last":
        return arr[-1]
    if mode == "best_of_h":
        return np.nanmax(arr)
    return np.nanmean(arr)


# =====================================================================
# Training
# =====================================================================

def train_model(
    model, dataset_train, dataset_val, P, Q,
    epochs, lr, batch_size, device,
    patience=10, min_delta=1e-3, eval_mode="last"
):
    """
    Treina modelo com estratégia ONE-STEP (teacher forcing).
    
    IMPORTANTE: O modelo é treinado para prever APENAS o primeiro horizonte
    (t+1). Durante a inferência, o modelo opera de forma AUTOREGRESSIVA
    para gerar múltiplos passos (t+1 a t+Q).
    
    Esta divergência entre treinamento e inferência é intencional e segue
    a prática comum em previsão de séries temporais espaciais (Rolling Forecast).
    
    Args:
        model: Modelo a ser treinado
        dataset_train: Dataset de treinamento (one-step supervision)
        dataset_val: Dataset de validação (avaliado em multi-step)
        Q: Horizonte de previsão (usado apenas para validação)
        eval_mode: Modo de seleção do melhor modelo
            - "last": usa WI do último horizonte
            - "best_of_h": usa melhor WI entre horizontes
            - "mean": usa média do WI entre horizontes
    """
    
    print(f"\nTraining P={P}, Q={Q}")

    if len(dataset_train) == 0:
        print("Empty training dataset.")
        return model

    ld_train = DataLoader(dataset_train, batch_size=batch_size, 
                          shuffle=True, num_workers=2, pin_memory=True)
    ld_val = DataLoader(dataset_val, batch_size=batch_size, 
                        shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    best_wi = -float("inf")
    best_state = None
    best_wi_by_h = [np.nan] * Q
    patience_counter = 0

    history_loss = []
    history_wi = []

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0.0

        for x, y_seq in ld_train:
            x = x.to(device, non_blocking=True)
            y_seq = y_seq.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model.forward_one_step(x)
            loss = loss_fn(y_pred, y_seq[:, 0])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(ld_train), 1)
        history_loss.append(avg_loss)

        if len(dataset_val) == 0:
            val_wi = np.nan
            wi_h = [np.nan] * Q
        else:
            wi_h = wi_by_horizon(model, ld_val, Q, device)
            val_wi = select_eval_mode(wi_h, mode=eval_mode)

        history_wi.append(val_wi)

        print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, val_WI={val_wi:.4f}")

        if np.isfinite(val_wi) and (val_wi - best_wi) > min_delta:
            best_wi = val_wi
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_wi_by_h = wi_h.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.best_wi = best_wi
    model.best_wi_by_h = best_wi_by_h
    model.eval_mode = eval_mode
    model.epochs_trained = len(history_loss)

    _save_training_curve(model, P, Q, history_loss, history_wi)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return model


def _save_training_curve(model, P, Q, history_loss, history_wi):
    """Saves training curve."""
    out_dir = Path("EXPERIMENTS/curves")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(history_loss)
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(history_wi)
    plt.ylabel("Validation WI")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    filename = f"training_curve_P{P}_Q{Q}.pdf"
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def evaluate_model(model, dataset_val, Q, device):
    """
    Evaluates model on validation set.
    """
    if len(dataset_val) == 0:
        return {
            "wi": np.nan, "rmse": np.nan, "mae": np.nan,
            "nse": np.nan, "bias": np.nan, "wi_by_h": [np.nan] * Q,
        }

    ld_val = DataLoader(dataset_val, batch_size=4, shuffle=False)
    model.eval()

    yt_all = []
    yp_all = []
    preds_h = [[] for _ in range(Q)]
    trues_h = [[] for _ in range(Q)]

    with torch.no_grad():
        for x, y_seq in ld_val:
            x = x.to(device)
            y_seq = y_seq.to(device)

            y_pred = model.forecast(x, Q)

            yt_all.append(y_seq.reshape(y_seq.size(0), -1))
            yp_all.append(y_pred.reshape(y_pred.size(0), -1))

            for h in range(Q):
                preds_h[h].append(y_pred[:, h].reshape(-1))
                trues_h[h].append(y_seq[:, h].reshape(-1))

    yt_all = torch.cat(yt_all, dim=0)
    yp_all = torch.cat(yp_all, dim=0)

    metrics = {
        "wi": float(wi(yt_all, yp_all)),
        "rmse": float(rmse(yt_all, yp_all)),
        "mae": float(mae(yt_all, yp_all)),
        "nse": float(nse(yt_all, yp_all)),
        "bias": float(bias(yt_all, yp_all)),
    }

    wi_h = []
    for h in range(Q):
        yp = torch.cat(preds_h[h])
        yt = torch.cat(trues_h[h])
        wi_h.append(float(wi(yt, yp)))

    metrics["wi_by_h"] = wi_h
    return metrics