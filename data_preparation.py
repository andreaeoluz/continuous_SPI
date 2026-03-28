"""
Data preparation for deep learning and classical models.
"""

import numpy as np
import random
import pandas as pd  # <-- ADICIONAR ESTA LINHA
from dataset import SPIDataset
from dataset_unified import UnifiedSPIDataset  # <-- ADICIONAR ESTA LINHA


def create_datasets(df_pr, df_spi, P, Q, split_date):
    """Creates training and validation datasets for DL models."""
    ds_train = SPIDataset(df_pr, df_spi, P, Q, train=True, split_date=split_date)
    ds_val = SPIDataset(df_pr, df_spi, P, Q, train=False, split_date=split_date)
    return ds_train, ds_val


def create_datasets_unified(df_pr, df_spi, P, Q, split_date, mode="full"):
    """
    Creates datasets with unified mode between DL and classical models.
    
    Args:
        mode: "full" (DL) or "sampled" (classical)
    """
    ds_train = UnifiedSPIDataset(
        df_pr, df_spi, P, Q, 
        train=True, 
        split_date=split_date,
        sampling_mode=mode,
        sampling_rate=0.1 if mode == "sampled" else 1.0,
        max_samples=50000 if mode == "sampled" else None
    )
    
    ds_val = UnifiedSPIDataset(
        df_pr, df_spi, P, Q, 
        train=False, 
        split_date=split_date,
        sampling_mode=mode,
        sampling_rate=0.1 if mode == "sampled" else 1.0,
        max_samples=12500 if mode == "sampled" else None
    )
    
    return ds_train, ds_val


def prepare_classic_data_unified(df_pr, df_spi, P, Q, split_date):
    """
    Unified version using the same dataset base.
    """
    ds_train, ds_val = create_datasets_unified(
        df_pr, df_spi, P, Q, split_date, mode="sampled"
    )
    
    # Extract tabular data
    X_train = []
    Y_train = []
    for i in range(len(ds_train)):
        x, y = ds_train[i]
        X_train.append(x)
        Y_train.append(y)
    
    X_val = []
    Y_val = []
    for i in range(len(ds_val)):
        x, y = ds_val[i]
        X_val.append(x)
        Y_val.append(y)
    
    if len(X_train) > 0:
        X_train = np.array(X_train, dtype=np.float32)
        Y_train = np.array(Y_train, dtype=np.float32)
    else:
        X_train = np.empty((0, 3 * P), dtype=np.float32)
        Y_train = np.empty((0, Q), dtype=np.float32)
    
    if len(X_val) > 0:
        X_val = np.array(X_val, dtype=np.float32)
        Y_val = np.array(Y_val, dtype=np.float32)
    else:
        X_val = np.empty((0, 3 * P), dtype=np.float32)
        Y_val = np.empty((0, Q), dtype=np.float32)
    
    H = ds_train.H if hasattr(ds_train, 'H') else 1
    W = ds_train.W if hasattr(ds_train, 'W') else 1
    
    return X_train, Y_train, X_val, Y_val, H, W


def prepare_classic_data(
    df_pr, 
    df_spi, 
    P, 
    Q, 
    split_date, 
    sampling_rate=0.1,
    max_samples=50000,
    random_seed=123
):
    """Prepares tabular data for classical models with spatial sampling."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    ds_train = SPIDataset(df_pr, df_spi, P, Q, train=True, split_date=split_date)
    ds_val = SPIDataset(df_pr, df_spi, P, Q, train=False, split_date=split_date)
    
    H = ds_train.pr.shape[1]
    W = ds_train.pr.shape[2]
    
    X_train, Y_train_seq = _extract_tabular_sampled(
        ds_train, P, Q, sampling_rate, max_samples
    )
    
    X_val, Y_val_seq = _extract_tabular_sampled(
        ds_val, P, Q, sampling_rate, max_samples // 4
    )
    
    return X_train, Y_train_seq, X_val, Y_val_seq, H, W


def _extract_tabular_sampled(dataset, P, Q, sampling_rate=0.1, max_samples=50000):
    """Extracts tabular data with random pixel sampling."""
    if len(dataset) == 0:
        return (
            np.empty((0, 3 * P), dtype=np.float32),
            np.empty((0, Q), dtype=np.float32),
        )
    
    H, W = dataset.pr.shape[1:]
    total_timesteps = len(dataset)
    total_pixels = H * W
    
    pixels_per_step = int(total_pixels * sampling_rate)
    pixels_per_step = max(1, min(pixels_per_step, total_pixels))
    
    max_pixels_per_step = max_samples // total_timesteps if total_timesteps > 0 else max_samples
    pixels_per_step = min(pixels_per_step, max_pixels_per_step)
    
    X_list = []
    Y_list = []
    
    all_pixels = [(i, j) for i in range(H) for j in range(W)]
    
    for t_idx in range(total_timesteps):
        x, y_seq = dataset[t_idx]
        sampled_pixels = random.sample(all_pixels, min(pixels_per_step, len(all_pixels)))
        
        for i, j in sampled_pixels:
            window = x[:, :, i, j].numpy()
            features = window.reshape(-1)
            target_seq = y_seq[:, i, j].numpy()
            
            if np.isnan(features).sum() > 0.5 * len(features):
                continue
                
            if np.isnan(target_seq).sum() > 0.5 * Q:
                continue
            
            X_list.append(features)
            Y_list.append(target_seq)
            
            if len(X_list) >= max_samples:
                break
        
        if len(X_list) >= max_samples:
            break
    
    if len(X_list) == 0:
        return (
            np.empty((0, 3 * P), dtype=np.float32),
            np.empty((0, Q), dtype=np.float32),
        )
    
    X = np.asarray(X_list, dtype=np.float32)
    Y_seq = np.asarray(Y_list, dtype=np.float32)
    
    X = np.nan_to_num(X, nan=0.0)
    Y_seq = np.nan_to_num(Y_seq, nan=0.0)
    
    return X, Y_seq


def prepare_classic_data_multi(df_pr, df_spi, P, Q, split_date):
    """Version without sampling (all pixels). Kept for compatibility."""
    ds_train = SPIDataset(df_pr, df_spi, P, Q, train=True, split_date=split_date)
    ds_val = SPIDataset(df_pr, df_spi, P, Q, train=False, split_date=split_date)
    
    X_train, Y_train_seq = _extract_tabular_full(ds_train, P, Q)
    X_val, Y_val_seq = _extract_tabular_full(ds_val, P, Q)
    
    H = ds_train.pr.shape[1]
    W = ds_train.pr.shape[2]
    
    return X_train, Y_train_seq, X_val, Y_val_seq, H, W


def _extract_tabular_full(dataset, P, Q):
    """Extracts all pixels without sampling."""
    X_list = []
    Y_list = []
    
    for idx in range(len(dataset)):
        x, y_seq = dataset[idx]
        x = x.numpy()
        y_seq = y_seq.numpy()
        
        _, _, H, W = x.shape
        
        for i in range(H):
            for j in range(W):
                window = x[:, :, i, j]
                features = window.reshape(-1)
                target_seq = y_seq[:, i, j]
                
                X_list.append(features)
                Y_list.append(target_seq)
    
    if len(X_list) == 0:
        return (
            np.empty((0, 3 * P), dtype=np.float32),
            np.empty((0, Q), dtype=np.float32),
        )
    
    X = np.asarray(X_list, dtype=np.float32)
    Y_seq = np.asarray(Y_list, dtype=np.float32)
    
    return X, Y_seq