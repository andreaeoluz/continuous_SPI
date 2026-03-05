# =====================================================================
# dataset.py — Autoregressive dataset for SPI forecasting (1 step)
# =====================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =====================================================================
# SPI → Class Conversion
# =====================================================================

def spi_to_class(spi):
    """
    Converts continuous SPI values into 7 WMO standard classes.
    """
    out = np.zeros_like(spi, dtype=np.int64)

    out[spi <= -2.0] = 0
    out[(spi > -2.0) & (spi <= -1.5)] = 1
    out[(spi > -1.5) & (spi <= -1.0)] = 2
    out[(spi > -1.0) & (spi <= 0.0)] = 3
    out[(spi > 0.0) & (spi <= 1.0)] = 4
    out[(spi > 1.0) & (spi <= 1.5)] = 5
    out[spi > 1.5] = 6

    return out


# =====================================================================
# Autoregressive Multi-Output Dataset (P,Q)
# =====================================================================

class SPIDataset(Dataset):

    def __init__(
        self,
        df_pr,
        df_spi,
        P,
        Q,
        train=True,
        split_date=None,
    ):

        dates = pd.to_datetime(df_pr.columns)
        T_total = len(dates)

        if split_date:
            sd = pd.to_datetime(split_date)
            idx_split = next((i for i, d in enumerate(dates) if d >= sd), T_total)
        else:
            idx_split = int(T_total * 0.8)

        if train:
            pr_df = df_pr.iloc[:, :idx_split]
            spi_df = df_spi.iloc[:, :idx_split]
        else:
            pr_df = df_pr.iloc[:, idx_split:]
            spi_df = df_spi.iloc[:, idx_split:]

        self.pr = self._df_to_cube(pr_df)
        self.spi = self._df_to_cube(spi_df)

        self.P = P
        self.Q = Q

        T = self.pr.shape[0]

        # valid windows require P historical + Q target
        self.indices = [
            t for t in range(0, T - P - Q + 1)
        ]

    def _df_to_cube(self, df):

        lats = sorted(df.index.get_level_values(0).unique(), reverse=True)
        lons = sorted(df.index.get_level_values(1).unique())

        T = len(df.columns)
        H, W = len(lats), len(lons)

        lat_pos = {v: i for i, v in enumerate(lats)}
        lon_pos = {v: j for j, v in enumerate(lons)}

        cube = np.full((T, H, W), np.nan, dtype=np.float32)

        for t, col in enumerate(df.columns):
            for (lat, lon), val in df[col].items():
                cube[t, lat_pos[lat], lon_pos[lon]] = val

        return cube

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        t0 = self.indices[idx]

        x_pr = self.pr[t0 : t0 + self.P]
        x_spi = self.spi[t0 : t0 + self.P]

        x_dspi = np.zeros_like(x_spi)
        if self.P > 1:
            x_dspi[1:] = x_spi[1:] - x_spi[:-1]

        x = np.stack([x_pr, x_spi, x_dspi], axis=1)

        # MULTI-HORIZON
        y_seq = self.spi[t0 + self.P : t0 + self.P + self.Q]

        x = np.nan_to_num(x, nan=0.0)
        y_seq = np.nan_to_num(y_seq, nan=0.0)

        return (
            torch.tensor(x, dtype=torch.float32),   # [P,3,H,W]
            torch.tensor(y_seq, dtype=torch.float32),  # [Q,H,W]
        )