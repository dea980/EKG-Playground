# ECG Anomaly Definition

In the context of MIT-BIH arrhythmia detection we treat the following cases as anomalies:

- **Morphology deviations**: QRS complexes with unusual width/asymmetry, ST elevation/depression, or missing P/T waves.
- **Temporal irregularities**: Abnormal RR intervals, premature beats (PVC/APB), dropped beats, or compensatory pauses.
- **Ectopic activity**: Ventricular or atrial premature contractions that break the dominant rhythm class.
- **Artifacts / noise spikes**: High-amplitude impulses or baseline wander that mimic beats.
- **Sensor issues**: Flat-line segments, saturation, or lead swaps.

When experimenting with unsupervised detectors we will label a beat as anomalous if any of the criteria above hold or if the reconstruction/forecasting error exceeds the dynamic threshold derived from clean normal beats.
