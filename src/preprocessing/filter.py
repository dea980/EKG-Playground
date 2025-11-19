"""Filtering helpers for ECG preprocessing."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")


def bandpass_filter(signal: np.ndarray, fs: float = 360, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth band-pass filtering to each channel."""
    b, a = _butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, fs: float = 360, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Remove mains hum using an IIR notch filter."""
    b, a = iirnotch(w0=freq / (fs / 2.0), Q=q)
    return filtfilt(b, a, signal, axis=0)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std
