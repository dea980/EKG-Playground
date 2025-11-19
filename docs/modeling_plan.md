# Modeling Plan

## Objectives
1. **Baseline**: Supervised PVC vs. non-PVC detection using CNN/ResNet on segmented beats.
2. **Exploration**: Unsupervised/ self-supervised anomaly scores using autoencoders and contrastive learning.
3. **Extension**: Policy-gradient (GRPO-style) calibrator that penalizes clinically unsafe miss/false-alarm trade-offs.

## Preprocessing Pipeline
- Band-pass filter 0.5–40 Hz, optional notch at mains frequency.
- R-peak detection (Pan–Tompkins or `wfdb.processing.xqrs_detect`).
- Beat segmentation (≈300–400 ms window centered on R-peak) and z-score normalization.
- Metadata merge: attach AAMI heartbeat label, patient ID, rhythm context.

Scripts live in `src/preprocessing/`. `load_mitbih.py` wraps WFDB IO, `filter.py` handles filtering helpers, and `segment.py` produces beat tensors persisted under `data/processed`.

## Modeling Tracks
### Supervised
- 1D CNN/ResNet or InceptionTime for beat classification.
- CNN+BiLSTM hybrid to capture temporal dependencies inside beat sequences.
- Metrics: beat-level accuracy, sensitivity/specificity, ROC AUC, PR AUC (PVC focus), and per-class confusion matrices.

### Unsupervised / Self-supervised
- Plain Autoencoder (AE) and Variational AE trained on normal beats.
- Isolation Forest / One-Class SVM / LOF for feature-space density estimation.
- Representation learning with contrastive forecasting; use reconstruction error histograms to set anomaly thresholds.

### Reinforcement-style Calibration
- Treat detector outputs as actions; GRPO reward boosts clinically valid detections, penalizes PVC misses more than benign false alarms.
- Adaptive sampling to handle class imbalance (normal vs rare events) during policy updates.

## Experiment Tracking
- Use notebooks under `notebooks/` for rapid exploration.
- Stage runnable training entry points in `src/train.py` (accepts config YAML/CLI flags) and log metrics to `results/logs`.
- Store qualitative plots (confusion matrices, beat reconstructions) in `results/figures`.
