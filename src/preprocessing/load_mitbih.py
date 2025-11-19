"""Convenience utilities for loading MIT-BIH Arrhythmia records."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import wfdb  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    wfdb = None


def _default_data_root() -> Path:
    """Try multiple locations so notebooks/scripts can find the dataset."""
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parents[2])
    except NameError:  # pragma: no cover
        pass
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    for base in candidates:
        candidate = base / "data" / "mitbih"
        if candidate.exists():
            return candidate
    # Fallback to cwd/data/mitbih even if missing so the error message is informative.
    return cwd / "data" / "mitbih"


DATA_ROOT = _default_data_root()


def _require_wfdb() -> None:
    if wfdb is None:
        raise ImportError(
            "wfdb is required to read MIT-BIH records. Install it via pip install wfdb"
        )


def list_records(data_root: Path = DATA_ROOT) -> List[str]:
    """Return all record identifiers shipped with the PhysioNet release."""
    records_file = data_root / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(
            f"Cannot find RECORDS file under {data_root}. Did you download MIT-BIH?"
        )
    return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]


def load_record(
    record_id: str,
    *,
    data_root: Path = DATA_ROOT,
    channels: Optional[Iterable[str]] = None,
    physical: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load a full-length record along with annotations."""
    _require_wfdb()
    record_path = str(data_root / record_id)
    record = wfdb.rdrecord(record_path, channel_names=channels, physical=physical)
    annotation = wfdb.rdann(record_path, "atr")
    signal = record.p_signal if physical else record.d_signal
    meta: Dict[str, np.ndarray] = {
        "sample": np.asarray(annotation.sample, dtype=np.int32),
        "symbol": np.asarray(annotation.symbol),
        "aux_note": np.asarray(annotation.aux_note),
    }
    return np.asarray(signal, dtype=np.float32), meta


def load_window(
    record_id: str,
    start_sample: int,
    stop_sample: int,
    *,
    data_root: Path = DATA_ROOT,
    channels: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Load a slice of a record (useful for notebooks)."""
    _require_wfdb()
    record_path = str(data_root / record_id)
    record = wfdb.rdrecord(
        record_path,
        sampfrom=start_sample,
        sampto=stop_sample,
        channel_names=channels,
        physical=True,
    )
    return np.asarray(record.p_signal, dtype=np.float32)
