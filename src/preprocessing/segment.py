"""Beat segmentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from wfdb import processing  # type: ignore
except ImportError:  # pragma: no cover
    processing = None


@dataclass
class SegmentationConfig:
    fs: float = 360.0
    pre_window: float = 0.2  # seconds before R-peak
    post_window: float = 0.4  # seconds after R-peak


def detect_r_peaks(signal: np.ndarray, fs: float = 360.0) -> np.ndarray:
    """Detect R-peaks using WFDB gqrs, fallback to scipy peak finding."""
    if processing is None:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(signal[:, 0], distance=int(0.2 * fs))
        return peaks
    return processing.xqrs_detect(sig=signal[:, 0], fs=fs)


def segment_beats(signal: np.ndarray, r_peaks: np.ndarray, config: SegmentationConfig | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    config = config or SegmentationConfig()
    pre = int(config.pre_window * config.fs)
    post = int(config.post_window * config.fs)
    window = pre + post

    beats = []
    keep_idx = []
    for idx, peak in enumerate(r_peaks):
        start = peak - pre
        stop = peak + post
        if start < 0 or stop > signal.shape[0]:
            continue
        beats.append(signal[start:stop, 0])
        keep_idx.append(idx)

    segments = np.stack(beats, axis=0) if beats else np.empty((0, window))
    meta = {
        "r_peaks": np.asarray(r_peaks[keep_idx], dtype=np.int32),
        "window": np.asarray([pre, post], dtype=np.int32),
    }
    return segments, meta
