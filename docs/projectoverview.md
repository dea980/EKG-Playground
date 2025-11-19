ECG는 심장질환 조기 발견의 핵심 신호지만, 실제 병원 데이터는 노이즈·오류·이상치가 매우 흔하다.

이상치(artifact / abnormal beats)를 정확하게 탐지하지 못하면, 모델은 잘못된 패턴을 학습하고 진단 오류 → 생명 위험으로 이어질 수 있다.

웨어러블 심전도(Apple Watch, Galaxy Watch)의 보급으로 대규모 ECG 데이터가 폭증하고 있다, 그러나 품질 관리 알고리즘은 여전히 취약하다.

MIT-BIH 등 기존 데이터셋은 라벨링은 잘 되어 있지만, 실제 웨어러블 환경을 반영하지 못한다.

기존 이상치 탐지 모델들은 도메인 일반화 능력이 부족해 특정 기관/장비 기반 데이터에만 최적화된다.

대부분의 모델들은 전처리(denoising, artifact removal)에 과도하게 의존하며, 이상치 자체를 학습 대상으로 간주하지 않는다.

임상 데이터는 imbalanced, rare anomaly, high-noise라서 일반적 딥러닝 모델은 overfit되기 쉽다.

Phase 1 — Classical Anomaly Detection

MIT-BIH 데이터 로드 및 전처리 (baseline drift 제거, filtering)

비정상 비트(annotation) 기반 binary anomaly label 구축

Classical methods 적용:

Z-score thresholding

Statistical outlier detection (IQR, median absolute deviation)

Peak anomaly detection

Isolation Forest

One-Class SVM

Phase 2 — Deep Learning Methods

1D CNN Autoencoder 기반 Reconstruction Error 탐지

Temporal convolutional network (TCN) 기반 예측 오차 기반 감지

Phase 3 — GRPO (2차 목표)

심전도의 “정상/비정상”을 reward로 하는
Group Relative Policy Optimization 기반 representation refinement

목표: 의료 데이터 부족 환경에서 Sim-to-Real 스타일 generalization 강화

MIT-BIH ECG 데이터를 기반으로 이상치 탐지 파이프라인을 구축한다.

Rule-based + ML-based + DL-based 방법을 비교하여 현실적 성능·해석력·일반화 능력의 균형을 평가한다.

궁극적으로 임상 활용 가능성을 갖춘 lightweight anomaly detector를 목표로 한다.

