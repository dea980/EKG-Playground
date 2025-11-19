"""1D CNN baseline for supervised arrhythmia detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class BeatClassifier(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.feature(x))


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cpu"


def _step(batch, model: nn.Module, criterion: nn.Module, device: str) -> Tensor:
    if isinstance(batch, dict):
        inputs = batch["signal"].to(device)
        labels = batch["label"].to(device)
    else:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
    logits = model(inputs)
    return criterion(logits, labels)


def train_supervised(
    model: BeatClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: TrainConfig = TrainConfig(),
) -> Dict[str, float]:
    device = config.device
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    history: Dict[str, float] = {}
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = _step(batch, model, criterion, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_loader), 1)
        history[f"train_loss_{epoch}"] = epoch_loss

        if val_loader is None:
            continue
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += _step(batch, model, criterion, device).item()
        val_loss /= max(len(val_loader), 1)
        history[f"val_loss_{epoch}"] = val_loss
        if val_loss < best_val:
            best_val = val_loss
            history["best_epoch"] = epoch
    return history
