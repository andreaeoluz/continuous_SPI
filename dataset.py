# dataset.py - PyTorch Dataset for SPI forecasting with spatiotemporal inputs

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SPIDataset(Dataset):
    """
    Dataset for SPI drought forecasting.

    Input features for each sample:
        - Precipitation (pr)
        - Current SPI value (spi)
        - SPI temporal delta (dspi)

    Output:
        - Future SPI sequence over Q time steps
    """

    def __init__(self, df_pr: pd.DataFrame, df_spi: pd.DataFrame,
                 P: int, Q: int, period: str, indices: tuple):
        """
        Args:
            df_pr: Precipitation dataframe with (lat, lon) index and date columns
            df_spi: SPI dataframe with (lat, lon) index and date columns
            P: Number of past time steps to use as input
            Q: Number of future time steps to predict
            period: "train", "val", or "test"
            indices: Tuple of (train_idx, val_idx, test_idx) column indices
        """
        dates = pd.to_datetime(df_pr.columns)
        train_idx, val_idx, test_idx = indices

        if period == "train":
            pr_df = df_pr.iloc[:, train_idx]
            spi_df = df_spi.iloc[:, train_idx]
        elif period == "val":
            pr_df = df_pr.iloc[:, val_idx]
            spi_df = df_spi.iloc[:, val_idx]
        else:  # test
            pr_df = df_pr.iloc[:, test_idx]
            spi_df = df_spi.iloc[:, test_idx]

        self.pr = self._df_to_cube(pr_df)
        self.spi = self._df_to_cube(spi_df)
        self.P = P
        self.Q = Q

        T = self.pr.shape[0]  # number of time steps
        self.indices = [t for t in range(0, T - P - Q + 1)]

    def _df_to_cube(self, df: pd.DataFrame) -> np.ndarray:
        """Convert (lat, lon) indexed dataframe to 3D cube (T, H, W)."""
        lats = sorted(df.index.get_level_values(0).unique(), reverse=True)
        lons = sorted(df.index.get_level_values(1).unique())
        T, H, W = len(df.columns), len(lats), len(lons)

        lat_pos = {v: i for i, v in enumerate(lats)}
        lon_pos = {v: j for j, v in enumerate(lons)}

        cube = np.full((T, H, W), np.nan, dtype=np.float32)
        for t, col in enumerate(df.columns):
            for (lat, lon), val in df[col].items():
                cube[t, lat_pos[lat], lon_pos[lon]] = val
        return cube

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """Return (input_tensor, target_tensor)."""
        t0 = self.indices[idx]

        # Past precipitation
        x_pr = self.pr[t0:t0 + self.P]

        # Past SPI
        x_spi = self.spi[t0:t0 + self.P]

        # SPI delta (temporal difference)
        x_dspi = np.zeros_like(x_spi)
        if self.P > 1:
            x_dspi[1:] = x_spi[1:] - x_spi[:-1]

        # Stack channels: (P, C=3, H, W)
        x = np.stack([x_pr, x_spi, x_dspi], axis=1)

        # Target: future SPI sequence (Q, H, W)
        y_seq = self.spi[t0 + self.P:t0 + self.P + self.Q]

        # Replace NaN with zero
        x_tensor = torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32)
        y_tensor = torch.tensor(np.nan_to_num(y_seq, nan=0.0), dtype=torch.float32)

        return x_tensor, y_tensor