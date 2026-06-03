# train_model.py - Training loop for ConvLSTM3D model

import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONVLSTM3D_PARAMS, EVAL_MODE, DEVICE
from metrics import wi, rmse
from plots import plot_training_curves


def train_model(model, dataset_train, dataset_val, P, Q,
                epochs=None, lr=None, batch_size=None, device=None,
                patience=None, eval_mode=None):
    """
    Train ConvLSTM3D model using teacher forcing.

    Args:
        model: ConvLSTM3D instance
        dataset_train: Training dataset
        dataset_val: Validation dataset
        P: Input sequence length
        Q: Output sequence length
        epochs: Maximum number of epochs
        lr: Learning rate
        batch_size: Batch size
        device: torch device
        patience: Early stopping patience
        eval_mode: "last", "best_of_h", or "all"

    Returns:
        Trained model (loaded with best weights)
    """
    epochs = epochs or CONVLSTM3D_PARAMS["epochs"]
    lr = lr or CONVLSTM3D_PARAMS["lr"]
    batch_size = batch_size or CONVLSTM3D_PARAMS["batch_size"]
    device = device or DEVICE
    patience = patience or CONVLSTM3D_PARAMS["patience"]
    eval_mode = eval_mode or EVAL_MODE

    if len(dataset_train) == 0:
        print(f"  ⚠ No training data for P={P}, Q={Q}")
        return model

    print(f"\n  📊 Training: P={P}, Q={Q} | epochs={epochs} | lr={lr} | batch={batch_size}")
    print(f"     Train samples: {len(dataset_train)} | Val samples: {len(dataset_val)}")

    # Inspect spatial dimensions
    sample_x, sample_y = dataset_train[0]
    P_seq, C, H, W = sample_x.shape
    print(f"     Input shape: [P={P_seq}, C={C}, H={H}, W={W}]")
    print(f"     Target shape: [Q={Q}, H={H}, W={W}]")

    # Windows compatibility: num_workers must be 0
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    ) if len(dataset_val) > 0 else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-6
    )

    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_wi = -float("inf")
    best_state = None
    patience_counter = 0
    history_loss, history_wi, history_rmse = [], [], []

    print(f"     Epoch | Loss    | Val WI  | Val RMSE | LR")
    print(f"     {'-' * 48}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # === TRAINING WITH TEACHER FORCING ===
        for x, y_seq in loader_train:
            x = x.to(device, non_blocking=True)
            y_seq = y_seq.to(device, non_blocking=True)

            B = x.shape[0]
            H, W = x.shape[3], x.shape[4]

            optimizer.zero_grad(set_to_none=True)

            # Teacher forcing: use ground truth as input for next step
            current = x.clone()
            predictions = []

            for step in range(Q):
                pred = model.forward_one_step(current)
                predictions.append(pred.unsqueeze(1))

                if step < Q - 1:
                    new_input = torch.zeros(B, 1, 3, H, W, dtype=x.dtype, device=device)
                    new_input[:, 0, 0] = current[:, -1, 0]           # precipitation
                    new_input[:, 0, 1] = y_seq[:, step + 1]          # true SPI
                    new_input[:, 0, 2] = y_seq[:, step + 1] - current[:, -1, 1]  # true delta

                    current = torch.cat([current[:, 1:], new_input], dim=1)

            y_pred = torch.cat(predictions, dim=1)
            loss = loss_fn(y_pred, y_seq)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        history_loss.append(avg_loss)

        # Periodic cache cleanup
        if device.type == "cuda" and epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # === VALIDATION (autoregressive forecast) ===
        if loader_val is not None and len(dataset_val) > 0:
            model.eval()
            wi_h = []
            rmse_h = []

            with torch.no_grad():
                for x, y_seq in loader_val:
                    x = x.to(device, non_blocking=True)
                    y_seq = y_seq.to(device, non_blocking=True)

                    pred = model.forecast(x, Q)

                    for h in range(Q):
                        yt_h = y_seq[:, h].flatten()
                        yp_h = pred[:, h].flatten()

                        wi_h.append(float(wi(yt_h, yp_h).cpu()))
                        rmse_h.append(float(rmse(yt_h, yp_h).cpu()))

            # Aggregate validation metric based on eval_mode
            if eval_mode == "last":
                val_wi = sum(wi_h[-Q:]) / Q if len(wi_h) >= Q else sum(wi_h) / len(wi_h)
            elif eval_mode == "best_of_h":
                val_wi = max(wi_h[-Q:]) if len(wi_h) >= Q else max(wi_h)
            else:
                val_wi = sum(wi_h) / len(wi_h)

            val_rmse = sum(rmse_h) / len(rmse_h)

            history_wi.append(val_wi)
            history_rmse.append(val_rmse)

            scheduler.step(val_wi)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"     {epoch:4d} | {avg_loss:6.4f} | {val_wi:7.4f} | {val_rmse:7.4f} | {current_lr:.6f}")

            # Early stopping logic
            if val_wi - best_wi > 1e-6:
                best_wi = val_wi
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"     ⏹ Early stopping at epoch {epoch}")
                break
        else:
            print(f"     {epoch:4d} | {avg_loss:6.4f} | {'N/A':7} | {'N/A':7} | {optimizer.param_groups[0]['lr']:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_wi = best_wi

    print(f"     ✅ Best WI: {best_wi:.4f}")

    plot_training_curves(history_loss, history_wi, history_rmse, P, Q)

    # Clean up
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return model