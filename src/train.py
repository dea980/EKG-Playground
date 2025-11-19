"""CLI entry point for training detectors."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from detection.supervised_cnn import BeatClassifier, TrainConfig, train_supervised
from detection.unsupervised_autoencoder import AutoEncoderConfig, ConvAutoEncoder, train_autoencoder

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_split(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = PROCESSED_DIR / f"beats_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split {split} not found at {path}. Generate it via preprocessing notebook first."
        )
    data = np.load(path)
    beats = torch.from_numpy(data["beats"]).float()
    labels = torch.from_numpy(data["labels"]).long()
    beats = beats.unsqueeze(1) if beats.ndim == 2 else beats
    return beats, labels


def _make_loader(split: str, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    beats, labels = _load_split(split)
    dataset = TensorDataset(beats, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG anomaly detectors")
    parser.add_argument("mode", choices=["supervised", "autoencoder"], help="Training objective")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "supervised":
        train_loader = _make_loader("train", batch_size=args.batch_size)
        val_loader = _make_loader("val", batch_size=args.batch_size, shuffle=False)
        model = BeatClassifier(n_channels=1, n_classes=2)
        config = TrainConfig(epochs=args.epochs, lr=args.lr, device=args.device)
        history = train_supervised(model, train_loader, val_loader, config=config)
        print(history)
    else:
        train_loader = _make_loader("train", batch_size=args.batch_size)
        model = ConvAutoEncoder(n_channels=1)
        config = AutoEncoderConfig(epochs=args.epochs, lr=args.lr, device=args.device)
        history = train_autoencoder(model, train_loader, config=config)
        print(history)


if __name__ == "__main__":
    main()
