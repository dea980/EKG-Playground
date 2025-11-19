# Data Assets

- `mitbih/`: Original MIT-BIH Arrhythmia Database downloaded from PhysioNet. Contains waveform files (`.dat`), headers (`.hea`), annotations (`.atr`/`.xws`), and helper metadata from the PhysioNet release.
- `processed/`: Workspace for intermediate NumPy/Parquet tensors after segmentation, filtering, and feature extraction.

## Usage Notes
1. Always read records via `wfdb` or the helpers under `src/preprocessing/` so sampling rates and annotations remain synchronized.
2. Keep the processed folder lightweight—store only derived arrays that can be regenerated from scripts so the repo stays manageable.
3. If you download an updated release, drop it inside `mitbih/` and rerun preprocessing to refresh artifacts.
