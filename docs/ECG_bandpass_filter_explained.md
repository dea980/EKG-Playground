# Bandpass Filtering in ECG Preprocessing

## 1. Why Filtering Is Needed

ECG signals contain various forms of noise:

- **Baseline Wander** (0–0.5 Hz): breathing, movement, electrode drift  
- **Muscle Noise (EMG)** (20–100 Hz): high‑frequency random noise  
- **Powerline Noise** (50/60 Hz): electrical interference  
- **Motion Artifacts**

To detect R‑peaks and classify anomalies, we need to *preserve cardiac frequency components* (QRS complex ≈ 5–15 Hz) and suppress unwanted noise.

---

## 2. Butterworth Bandpass Filter

A bandpass filter removes frequencies outside a defined range.  
For ECG, a common choice is **5–15 Hz**.

### 2.1 Butterworth Filter Properties

- Maximally flat in the passband  
- No ripples  
- Smooth phase response  
- Stable for physiological signals

### 2.2 Mathematical Form

The *analog* Butterworth filter has transfer function:

\[
H(s) = \frac{1}{\sqrt{1 + ( \frac{s}{\omega_c} )^{2n}}}
\]

Where  
- \( n \): filter order  
- \( \omega_c \): cutoff frequency  

For a **bandpass**, we convert low/high cutoffs:

\[
\omega_{low} = 2\pi f_{low}, \quad \omega_{high} = 2\pi f_{high}
\]

Then we apply **bilinear transform** to map analog → digital filter:

\[
s = \frac{2}{T} \, \frac{1 - z^{-1}}{1 + z^{-1}}
\]

### 2.3 Why We Use It

- Simple, stable, widely used in medical signal processing  
- Good for QRS extraction  
- Low‑order filter (order=4) works well for 360 Hz sampling

---

## 3. `butter_bandpass()` Function Explained

```python
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
```

### Step-by-step meaning:

1. **Nyquist Frequency**  
   \[
   f_N = \frac{fs}{2}
   \]
   Needed to normalize frequencies to (0–1).

2. **Normalize frequency**  
   \[
   low = \frac{f_{low}}{f_N}, \quad high = \frac{f_{high}}{f_N}
   \]

3. **Butterworth filter design**  
   SciPy returns transfer coefficients:

   - \( b \): numerator  
   - \( a \): denominator  

   Transfer function:

   \[
   H(z) = \frac{b_0 + b_1 z^{-1} + \cdots}{1 + a_1 z^{-1} + a_2 z^{-2} + \cdots}
   \]

---

## 4. `bandpass_filter()` Function Explained

```python
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
```

### Why `filtfilt()`?

- Applies filter forward + backward  
- Cancels phase distortion  
- Essential for medical signals  

This produces:

\[
y[n] = H(z) \star x[n]
\]

---

## 5. Alternatives to Butterworth

| Filter | 특징 | 언제 사용? |
|-------|------|-------------|
| **Chebyshev I/II** | Ripple 존재, 더 가파른 차단 | 빠른 roll‑off 필요할 때 |
| **Elliptic** | 최강의 sharp roll‑off, ripple 있음 | 매우 좁은 대역 제한 시 |
| **Bessel** | 최고의 위상 선형성, 완만함 | PPG, HRV 같이 위상 중요할 때 |
| **Wavelet Denoising** | 비정상 신호에 강함 | ECG 비정상 패턴 탐지 |
| **Savitzky–Golay** | smoothing 계열 | 피크 보존 smoothing |

---

## 6. Why Butterworth Is Chosen for ECG (초기 단계)

- PQRST 파형을 왜곡하지 않음  
- 구현 쉽고 안정적  
- R-peak detection이 크게 향상됨  
- MIT‑BIH 연구에서 가장 흔하게 쓰임  
- 신호 초보자가 이해·확장하기 좋음

---

## 7. Summary

- ECG preprocessing의 핵심은 *QRS 대역 보존 + 노이즈 제거*  
- Butterworth bandpass 5–15Hz는 가장 많이 검증된 방식  
- 필터링을 이해해야 나중에 **특징 추출 → ML/RL → 이상치 탐지** 흐름이 확립됨  
- Wavelet, Chebyshev 등 대안은 2차 확장 단계에서 사용 가능

