"""
Dataset unificado para modelos clássicos e deep learning.
Permite amostragem espacial consistente entre abordagens.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random


class UnifiedSPIDataset(Dataset):
    """
    Dataset que pode operar em modo full-grid ou pixel-sampled.
    Garante consistência entre modelos clássicos e DL.
    """
    
    def __init__(
        self,
        df_pr,
        df_spi,
        P,
        Q,
        train=True,
        split_date=None,
        sampling_mode="full",  # "full" ou "sampled"
        sampling_rate=0.1,
        max_samples=50000,
        random_seed=123
    ):
        # Inicializa dataset base (igual ao original)
        self.sampling_mode = sampling_mode
        self.sampling_rate = sampling_rate
        self.max_samples = max_samples
        self.random_seed = random_seed
        
        # Cria dataset completo primeiro
        self._init_base_dataset(df_pr, df_spi, P, Q, train, split_date)
        
        if sampling_mode == "sampled":
            self._prepare_sampled_indices()
    
    def _init_base_dataset(self, df_pr, df_spi, P, Q, train, split_date):
        """Inicializa dataset completo (igual ao SPIDataset original)."""
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
        self.time_indices = [t for t in range(0, T - P - Q + 1)]
        
        self.H = self.pr.shape[1]
        self.W = self.pr.shape[2]
    
    def _prepare_sampled_indices(self):
        """Prepara lista de (time_idx, i, j) amostrados."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        total_timesteps = len(self.time_indices)
        total_pixels = self.H * self.W
        
        pixels_per_step = int(total_pixels * self.sampling_rate)
        pixels_per_step = max(1, min(pixels_per_step, total_pixels))
        
        max_pixels_per_step = self.max_samples // total_timesteps
        pixels_per_step = min(pixels_per_step, max_pixels_per_step)
        
        all_pixels = [(i, j) for i in range(self.H) for j in range(self.W)]
        
        self.sampled_pairs = []
        
        for t_idx in range(total_timesteps):
            sampled_pixels = random.sample(all_pixels, pixels_per_step)
            for i, j in sampled_pixels:
                self.sampled_pairs.append((t_idx, i, j))
                
                if len(self.sampled_pairs) >= self.max_samples:
                    break
            if len(self.sampled_pairs) >= self.max_samples:
                break
    
    def _df_to_cube(self, df):
        """Converte DataFrame para cubo 3D (T, H, W)."""
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
        if self.sampling_mode == "sampled":
            return len(self.sampled_pairs)
        return len(self.time_indices)
    
    def __getitem__(self, idx):
        if self.sampling_mode == "sampled":
            t_idx, i, j = self.sampled_pairs[idx]
            return self._get_item_at_time_and_pixel(t_idx, i, j)
        else:
            return self._get_full_grid_item(idx)
    
    def _get_full_grid_item(self, idx):
        """Retorna tensor completo [P, 3, H, W]."""
        t0 = self.time_indices[idx]
        
        x_pr = self.pr[t0:t0 + self.P]
        x_spi = self.spi[t0:t0 + self.P]
        
        x_dspi = np.zeros_like(x_spi)
        if self.P > 1:
            x_dspi[1:] = x_spi[1:] - x_spi[:-1]
        
        x = np.stack([x_pr, x_spi, x_dspi], axis=1)
        
        y_seq = self.spi[t0 + self.P:t0 + self.P + self.Q]
        
        x = np.nan_to_num(x, nan=0.0)
        y_seq = np.nan_to_num(y_seq, nan=0.0)
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32)
        )
    
    def _get_item_at_time_and_pixel(self, t_idx, i, j):
        """Retorna amostra pixel-wise [3P] e [Q]."""
        t0 = self.time_indices[t_idx]
        
        # Extrair série temporal para um pixel específico
        x_pr = self.pr[t0:t0 + self.P, i, j]
        x_spi = self.spi[t0:t0 + self.P, i, j]
        
        x_dspi = np.zeros_like(x_spi)
        if self.P > 1:
            x_dspi[1:] = x_spi[1:] - x_spi[:-1]
        
        # Features: [3P]
        features = np.concatenate([x_pr, x_spi, x_dspi])
        
        # Target: [Q]
        target = self.spi[t0 + self.P:t0 + self.P + self.Q, i, j]
        
        features = np.nan_to_num(features, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)
        
        return features, target