# =====================================================================
# model_convlstm3d.py — ConvLSTM3D + Temporal Attention (BN + Large Kernel)
# =====================================================================

import torch
import torch.nn as nn


# =====================================================================
# ConvLSTM3D Cell (with internal BatchNorm + larger kernel)
# =====================================================================

class ConvLSTM3DCell(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv3d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False
        )

        # 🔥 BatchNorm inside cell (stabilizes gates)
        self.bn = nn.BatchNorm3d(4 * hidden_channels)

    def forward(self, x, h_prev, c_prev):

        x3 = x.unsqueeze(2)
        h3 = h_prev.unsqueeze(2)

        combined = torch.cat([x3, h3], dim=1)

        gates = self.bn(self.conv(combined)).squeeze(2)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# =====================================================================
# Channel Attention
# =====================================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=(2, 3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


# =====================================================================
# Spatial Attention
# =====================================================================

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=5, padding=2)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


# =====================================================================
# ConvLSTM Encoder + Temporal Attention
# =====================================================================

class ConvLSTM3DEncoder(nn.Module):

    def __init__(self, input_dim=3, hidden_dims=(64, 48, 32)):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for i, hidden_dim in enumerate(hidden_dims):
            in_channels = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(
                ConvLSTM3DCell(in_channels, hidden_dim, kernel_size=3)
            )

        # 🔥 More robust temporal attention
        self.temp_fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.temp_fc2 = nn.Linear(hidden_dims[-1] // 2, 1)

    def forward(self, x):

        B, P, C, H, W = x.shape

        h_states = [
            torch.zeros(B, h, H, W, device=x.device)
            for h in self.hidden_dims
        ]

        c_states = [
            torch.zeros(B, h, H, W, device=x.device)
            for h in self.hidden_dims
        ]

        temporal_outputs = []

        for t in range(P):

            input_t = x[:, t]

            for layer_idx, cell in enumerate(self.layers):

                h_states[layer_idx], c_states[layer_idx] = cell(
                    input_t,
                    h_states[layer_idx],
                    c_states[layer_idx],
                )

                input_t = h_states[layer_idx]

            temporal_outputs.append(h_states[-1])

        temporal_stack = torch.stack(temporal_outputs, dim=1)

        pooled = temporal_stack.mean(dim=(3, 4))  # [B,P,C]

        scores = torch.relu(self.temp_fc1(pooled))
        scores = self.temp_fc2(scores)

        weights = torch.softmax(scores, dim=1)

        weights = weights.unsqueeze(-1).unsqueeze(-1)

        weighted_sum = (temporal_stack * weights).sum(dim=1)

        return weighted_sum


# =====================================================================
# Complete Model
# =====================================================================

class ConvLSTM3D(nn.Module):

    def __init__(self, hidden=(64, 48, 32), dropout_p=0.2):
        super().__init__()

        self.encoder = ConvLSTM3DEncoder(
            input_dim=3,
            hidden_dims=hidden
        )

        self.channel_att = SEBlock(hidden[-1])
        self.spatial_att = SpatialAttention()

        self.refine = nn.Conv2d(
            hidden[-1],
            hidden[-1],
            kernel_size=3,
            padding=1
        )

        self.dropout = nn.Dropout2d(p=dropout_p)
        self.batch_norm = nn.BatchNorm2d(hidden[-1])

        # 🔥 Deeper head
        self.head = nn.Sequential(
            nn.Conv2d(hidden[-1], 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    # -----------------------------------------------------------------

    def forward_one_step(self, x):

        h = self.encoder(x)

        h = self.channel_att(h)
        h = self.spatial_att(h)

        h_ref = torch.relu(self.refine(h))
        h_ref = h_ref + h

        h_ref = self.dropout(h_ref)
        h_norm = self.batch_norm(h_ref)

        spi_pred = self.head(h_norm).squeeze(1)

        # Global residual skip
        last_spi = x[:, -1, 1]
        spi_pred = spi_pred + last_spi

        return spi_pred

    # -----------------------------------------------------------------

    def forecast(self, x_init, Q):
        """
        Gera previsões multi-horizonte SEM estimar precipitação.
        Apenas SPI é previsto; PR é mantida constante (último valor observado).
        """
        B, P, C, H, W = x_init.shape
        predictions = []
        current_window = x_init.clone()

        for step in range(Q):
            spi_pred = self.forward_one_step(current_window)
            predictions.append(spi_pred.unsqueeze(1))

            # Criar nova entrada apenas com SPI previsto
            new_in = torch.zeros(B, 1, C, H, W, device=x_init.device)
            
            # PR: mantém o último valor observado
            new_in[:, 0, 0] = current_window[:, -1, 0]  # PR constante
            
            # SPI: valor previsto
            new_in[:, 0, 1] = spi_pred
            
            # ΔSPI: diferença entre SPI previsto e último observado
            last_spi = current_window[:, -1, 1]
            new_in[:, 0, 2] = spi_pred - last_spi

            current_window = torch.cat([current_window[:, 1:], new_in], dim=1)

        return torch.cat(predictions, dim=1)