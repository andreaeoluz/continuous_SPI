"""
Data preparation for deep learning and classical models.
"""

import numpy as np
import random
from dataset import SPIDataset


def create_datasets(df_pr, df_spi, P, Q, split_date):
    """
    Creates training and validation datasets for DL models.
    
    Args:
        df_pr: Precipitation DataFrame
        df_spi: SPI DataFrame
        P: Historical steps
        Q: Forecast horizon
        split_date: Training/validation cutoff date
        
    Returns:
        ds_train, ds_val: Training and validation datasets
    """
    ds_train = SPIDataset(
        df_pr, df_spi, P, Q, train=True, split_date=split_date
    )

    ds_val = SPIDataset(
        df_pr, df_spi, P, Q, train=False, split_date=split_date
    )

    return ds_train, ds_val


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
    """
    Prepares tabular data for classical models with spatial sampling.
    
    Args:
        sampling_rate: Fraction of pixels to sample per timestep
        max_samples: Maximum total number of samples
        random_seed: Seed for reproducibility
    
    Returns:
        X_train, Y_train_seq, X_val, Y_val_seq, H, W
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    ds_train = SPIDataset(
        df_pr, df_spi, P, Q, train=True, split_date=split_date
    )
    
    ds_val = SPIDataset(
        df_pr, df_spi, P, Q, train=False, split_date=split_date
    )
    
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
    """
    Extracts tabular data with random pixel sampling.
    """
    
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
    
    max_pixels_per_step = max_samples // total_timesteps
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
    """
    Version without sampling (all pixels).
    Kept for compatibility.
    """
    ds_train = SPIDataset(
        df_pr, df_spi, P, Q, train=True, split_date=split_date
    )
    
    ds_val = SPIDataset(
        df_pr, df_spi, P, Q, train=False, split_date=split_date
    )
    
    X_train, Y_train_seq = _extract_tabular_full(ds_train, P, Q)
    X_val, Y_val_seq = _extract_tabular_full(ds_val, P, Q)
    
    H = ds_train.pr.shape[1]
    W = ds_train.pr.shape[2]
    
    return X_train, Y_train_seq, X_val, Y_val_seq, H, W


def _extract_tabular_full(dataset, P, Q):
    """
    Extracts all pixels without sampling.
    """
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