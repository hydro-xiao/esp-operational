"""Encoder-decoder TCN post-processor architecture. Verbatim copy of
model_encdec_tcn.EncDecTCNForecaster (lstm.new.dev.jul.2026/scr) -- model
definition only, no training code, so this file plus the saved checkpoints
(runs/tcn_{jan,feb,mar,apr,may}/{config.json,norm_stats.npz,model.pt}) are
everything the server needs to run TCN inference."""
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove right padding added by Conv1d, preserving causality and length."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout1d(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout1d(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.out_act = nn.GELU()

    def forward(self, x):
        residual = self.residual(x)
        y = self.conv1(x)
        y = self.drop1(self.act1(self.chomp1(y)))
        y = self.conv2(y)
        y = self.drop2(self.act2(self.chomp2(y)))
        return self.out_act(y + residual)


class TCNStack(nn.Module):
    def __init__(self, in_channels, hidden, n_levels, kernel_size, dropout):
        super().__init__()
        blocks = []
        ch_in = in_channels
        for level in range(n_levels):
            blocks.append(TemporalBlock(ch_in, hidden, kernel_size, dilation=2 ** level, dropout=dropout))
            ch_in = hidden
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class EncDecTCNForecaster(nn.Module):
    def __init__(self, n_features, n_basins, embed_dim=8, hidden_size=64, n_levels=3, kernel_size=3, dropout=0.4):
        super().__init__()
        self.basin_embed = nn.Embedding(n_basins, embed_dim)
        self.encoder = TCNStack(n_features + embed_dim, hidden_size, n_levels, kernel_size, dropout)
        self.decoder = TCNStack(n_features + embed_dim + hidden_size, hidden_size, n_levels, kernel_size, dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_lookback, x_horizon, basin_idx):
        b_embed_lb = self.basin_embed(basin_idx).unsqueeze(1).expand(-1, x_lookback.size(1), -1)
        enc_in = torch.cat([x_lookback, b_embed_lb], dim=-1)
        enc_out = self.encoder(enc_in)
        context = enc_out[:, -1, :]

        b_embed_h = self.basin_embed(basin_idx).unsqueeze(1).expand(-1, x_horizon.size(1), -1)
        context_h = context.unsqueeze(1).expand(-1, x_horizon.size(1), -1)
        dec_in = torch.cat([x_horizon, b_embed_h, context_h], dim=-1)
        dec_out = self.decoder(dec_in)
        return self.head(dec_out).squeeze(-1)
