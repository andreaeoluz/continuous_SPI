# data_preparation.py - Data loading utilities for classical ML models

import random
import numpy as np
from dataset import SPIDataset


def create_datasets(df_pr, df_spi, P, Q, indices):
    """
    Create train, validation, and test SPIDataset instances.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    return (
        SPIDataset(df_pr, df_spi, P, Q, "train", indices),
        SPIDataset(df_pr, df_spi, P, Q, "val", indices),
        SPIDataset(df_pr, df_spi, P, Q, "test", indices),
    )


def prepare_classic_data(df_pr, df_spi, P, Q, indices,
                         sampling_rate=0.1, max_samples=10000,
                         random_seed=123):
    """
    Convert spatiotemporal datasets to flat feature matrices for classical ML.

    Spatial sampling is applied to keep memory usage feasible.
    Each sample is a (pixel, time) pair flattened into a feature vector.

    Returns:
        tuple: (X_train, Y_train, X_val, Y_val, X_test, Y_test, H, W)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    ds_train = SPIDataset(df_pr, df_spi, P, Q, "train", indices)
    ds_val = SPIDataset(df_pr, df_spi, P, Q, "val", indices)
    ds_test = SPIDataset(df_pr, df_spi, P, Q, "test", indices)

    # Infer spatial dimensions (H, W) from the first available dataset
    H, W = None, None
    for ds in [ds_train, ds_val, ds_test]:
        if len(ds) > 0:
            sample_x, _ = ds[0]
            _, _, H, W = sample_x.shape  # (P, C, H, W)
            break

    if H is None or W is None:
        # Fallback: infer from dataframe indices
        lats = sorted(df_spi.index.get_level_values(0).unique(), reverse=True)
        lons = sorted(df_spi.index.get_level_values(1).unique())
        H, W = len(lats), len(lons)

    def extract_fast(ds, max_samp, sampling_rate):
        """Extract feature matrix from dataset with pixel-level sampling."""
        if len(ds) == 0:
            return np.empty((0, 3 * P), dtype=np.float32), np.empty((0, Q), dtype=np.float32)

        total_pixels = H * W
        n_pixels_sample = int(total_pixels * sampling_rate)
        n_pixels_sample = max(1, min(n_pixels_sample, max_samp // max(1, len(ds))))

        # Select fixed pixels for consistency across all time steps
        all_pixels = [(i, j) for i in range(H) for j in range(W)]
        fixed_pixels = random.sample(all_pixels, min(n_pixels_sample, len(all_pixels)))

        max_possible = len(ds) * len(fixed_pixels)
        X = np.zeros((max_possible, 3 * P), dtype=np.float32)
        Y = np.zeros((max_possible, Q), dtype=np.float32)

        idx = 0
        for t in range(len(ds)):
            x, y_seq = ds[t]
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            y_np = y_seq.numpy() if hasattr(y_seq, 'numpy') else y_seq

            for i, j in fixed_pixels:
                features = x_np[:, :, i, j].reshape(-1)  # (P * C)
                targets = y_np[:, i, j]                  # (Q,)

                # Accept sample if at least half of features/targets are valid
                if (np.isnan(features).sum() <= (3 * P) // 2 and
                        np.isnan(targets).sum() <= Q // 2):
                    X[idx] = np.nan_to_num(features, nan=0.0)
                    Y[idx] = np.nan_to_num(targets, nan=0.0)
                    idx += 1

                if idx >= max_samp:
                    break
            if idx >= max_samp:
                break

        return X[:idx], Y[:idx]

    X_train, Y_train = extract_fast(ds_train, max_samples, sampling_rate)
    X_val, Y_val = extract_fast(ds_val, max_samples, sampling_rate)
    X_test, Y_test = extract_fast(ds_test, max_samples, sampling_rate)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, H, W