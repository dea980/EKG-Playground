# Paper / Research Plan

## Week 1 – Foundation
- Finalize repo structure, document dataset, finish preprocessing prototype (filtering + R-peak detection).
- Train initial autoencoder baseline and visualize anomalous beats.

## Week 2 – Unsupervised Deep Dive
- Compare AE vs. VAE reconstructions and latent clustering.
- Improve artifact/noise detection and derive beat-level anomaly score calibration.

## Week 3 – Supervised Benchmarks
- Add CNN/LSTM/ResNet classifiers, perform patient-level splits, and benchmark against unsupervised models.
- Curate notebooks with experiments, confusion matrices, and PR curves.

## Week 4 – Reinforcement & Write-up
- Investigate GRPO-style reward shaping for clinical prioritization (PVC recall first).
- Draft research note + paper skeleton that explains pipeline, evaluation, and personalization outlook.

## Deliverables
1. Clean preprocessing scripts and reproducible training entry point.
2. Structured notebooks for dataset exploration, preprocessing validation, and baseline models.
3. Figures + logs stored under `results/` to support the write-up.
4. Draft manuscript section: introduction, methods, experiments, discussion.
