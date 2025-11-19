

 1. 데이터 레벨

MIT-BIH ECG 데이터 로드

360Hz waveform → preprocessing

baseline wandering 제거

noise filtering

beat segmentation

 2. Feature 레벨

RR interval, QRS duration, amplitude

morphology 기반 특징, frequency 특징

 3. 모델링 레벨

Classical Anomaly Detection (Isolation Forest, Elliptic Envelope)

Autoencoder 기반 reconstruction anomaly

1D CNN feature extractor

Lightweight transformer 적용 가능

 4. 2차 목표: GRPO 활용 가능성

anomaly labeling 없이도 self-supervised reward shaping 가능
“class imbalance + limited labeling” 문제를 self-supervised 방식으로 해결하려는 의도