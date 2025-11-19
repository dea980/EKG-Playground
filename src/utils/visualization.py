"""Plotting helpers for notebooks and scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_beats(beats: np.ndarray, title: str = "Beats", max_plots: int = 10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    subset = beats[:max_plots]
    for beat in subset:
        ax.plot(beat, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    return fig


def plot_reconstruction(original: np.ndarray, reconstructed: np.ndarray, title: str = "AE Reconstruction") -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(original, label="original")
    axes[0].legend()
    axes[1].plot(reconstructed, label="reconstructed")
    axes[1].legend()
    axes[1].set_xlabel("Sample")
    fig.suptitle(title)
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
