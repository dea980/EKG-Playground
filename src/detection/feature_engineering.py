"""Hand-crafted feature helpers for classic detectors."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _qrs_width(beat: np.ndarray, fs: float) -> float:
    peak_idx = np.argmax(np.abs(beat))
    amplitude = np.abs(beat[peak_idx])
    threshold = 0.5 * amplitude
    left = peak_idx
    while left > 0 and np.abs(beat[left]) > threshold:
        left -= 1
    right = peak_idx
    while right < beat.size - 1 and np.abs(beat[right]) > threshold:
        right += 1
    return (right - left) / fs


def _area_under_curve(beat: np.ndarray, fs: float) -> float:
    return float(np.trapz(np.abs(beat), dx=1.0 / fs))


def extract_features(beats: np.ndarray, r_peaks: np.ndarray, fs: float = 360.0) -> pd.DataFrame:
    """Return a pandas DataFrame with morphological + timing descriptors."""
    rr = np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([], dtype=np.float32)
    rr = np.concatenate([rr[:1], rr])  # align lengths

    features = []
    for idx, beat in enumerate(beats):
        feat: Dict[str, float] = {
            "beat_index": float(idx),
            "max_amp": float(np.max(beat)),
            "min_amp": float(np.min(beat)),
            "qrs_width": _qrs_width(beat, fs),
            "auc": _area_under_curve(beat, fs),
            "rr_interval": float(rr[min(idx, rr.shape[0] - 1)]),
        }
        features.append(feat)
    return pd.DataFrame(features)
