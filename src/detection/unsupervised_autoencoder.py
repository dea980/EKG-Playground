"""Simple convolutional autoencoder for unsupervised anomaly scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class ConvAutoEncoder(nn.Module):
    def __init__(self, n_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embedding_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(16),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


@dataclass
class AutoEncoderConfig:
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cpu"


def train_autoencoder(
    model: ConvAutoEncoder,
    loader: DataLoader,
    *,
    config: AutoEncoderConfig = AutoEncoderConfig(),
) -> Dict[str, float]:
    device = config.device
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    history: Dict[str, float] = {}

    for epoch in range(config.epochs):
        model.train()
        loss_total = 0.0
        for batch in loader:
            if isinstance(batch, dict):
                signal = batch["signal"].to(device)
            else:
                signal = batch[0].to(device)
            optimizer.zero_grad()
            output = model(signal)
            loss = criterion(output, signal)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        history[f"train_loss_{epoch}"] = loss_total / max(len(loader), 1)
    return history


def anomaly_score(model: ConvAutoEncoder, beats: Tensor) -> Tensor:
    """Return per-beat reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon = model(beats)
    return ((recon - beats) ** 2).mean(dim=(1, 2))
