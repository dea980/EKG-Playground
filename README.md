# ECG Anomaly Detection Playground

Pipeline for MIT-BIH Arrhythmia anomaly detection. The goal is to explore preprocessing, supervised CNN baselines, unsupervised autoencoders, and reinforcement-style calibration (GRPO) for clinically-aware alerting.

## Repository Layout
```
ECG-Anomaly-Detection/
├── data/
│   ├── mitbih/              # raw PhysioNet files (.dat/.hea/.atr)
│   ├── processed/           # filtered + segmented tensors (npz/parquet)
│   └── README.md
├── docs/
│   ├── dataset_overview.md
│   ├── anomaly_definition.md
│   ├── modeling_plan.md
│   └── paper_plan.md
├── src/
│   ├── preprocessing/
│   │   ├── load_mitbih.py
│   │   ├── filter.py
│   │   └── segment.py
│   ├── detection/
│   │   ├── supervised_cnn.py
│   │   ├── unsupervised_autoencoder.py
│   │   └── feature_engineering.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── train.py
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   └── 03_model_baseline.ipynb
├── results/
│   ├── figures/
│   └── logs/
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`.
2. Explore data using `notebooks/01_explore_dataset.ipynb`.
3. Generate processed beats and export `.npz` splits into `data/processed` via notebook 02.
4. Train baselines: `python src/train.py supervised --device cpu` or `python src/train.py autoencoder`.
5. Track outputs under `results/` and document findings inside `docs/`.
