motivation for this project

1. Problem Definition

ECG에서 이상치는 심정지, 심방세동, PVC 등의 전조가 될 수 있다.

웨어러블 ECG는 데스크 톱 의료기기보다 잡음이 크다.

실제 임상에서는 라벨링 비용이 매우 높고 사람 의존도가 크다.

→ “이상치 탐지는 의료 현장의 부담을 줄이고 조기 발견률을 높인다.”

2. Current Limitations

기존 supervised classification은 라벨 많아야 한다.

도메인 shift: 병원마다, 장비마다 신호가 다름.

Sim-to-Real 문제가 존재한다. (너가 좋아하는 영역)

3. Why MIT-BIH matters

benchmark dataset

anomaly label already annotated

연구 reproducibility 확보

4. Proposed Approach

classical anomaly → deep anomaly → RL(GRPO) 기반 refinement까지 pipeline

step-by-step 자연스러운 연구 확장 가능

5. Expected Contributions

ECG 데이터 전처리 파이프라인 재현

복수의 anomaly detection baseline 비교

deep method 기반 generalizable anomaly detection

향후 웨어러블 Real-to-Sim/Sim-to-Real 연구로 확장 가능