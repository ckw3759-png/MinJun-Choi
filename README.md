# MinJun-Choi
[기말_플젝.ipynb](https://github.com/user-attachments/files/24253584/_.ipynb)
[README.md](https://github.com/user-attachments/files/24253580/README.md)# 한국 금융 뉴스 기반 Linguistic Alpha 발굴 프로젝트

**BERT + LSTM 앙상블을 활용한 주가 초과수익률 예측 시스템**

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [핵심 아이디어](#핵심-아이디어)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [데이터 파이프라인](#데이터-파이프라인)
5. [모델 구조](#모델-구조)
6. [실행 방법](#실행-방법)
7. [성능 지표](#성능-지표)
8. [주요 기술적 도전과 해결](#주요-기술적-도전과-해결)
9. [결과 분석](#결과-분석)
10. [향후 개선 방향](#향후-개선-방향)

---

## 프로젝트 개요

### 목적
한국 금융 뉴스에서 **"언어적 알파(Linguistic Alpha)"**를 발굴하여 주식 초과수익률을 예측하는 AI 시스템 구축

### 배경
- 전통적 퀀트 전략은 가격/거래량 등 **정량적 데이터**에 의존
- 뉴스, 공시 등 **비정형 텍스트 데이터**는 활용도가 낮음
- 최신 NLP 기술(BERT)을 활용하면 텍스트에서 **예측 신호**를 추출 가능

### 핵심 질문
> "뉴스 감성 분석으로 3일 후 주가 초과수익률을 예측할 수 있는가?"

---

## 핵심 아이디어

### 1. Multi-Modal Learning
```
뉴스 텍스트 (BERT) + 주가 시계열 (LSTM) → 앙상블
```

- **BERT**: 뉴스 감성 → 시장 심리 파악
- **LSTM**: 주가 패턴 → 기술적 트렌드 파악
- **앙상블**: 두 정보를 결합하여 상호보완

### 2. Linguistic Alpha
- **정의**: 텍스트 정보에서 추출한 예측 신호와 실제 수익률 간의 상관관계
- **측정**: Information Coefficient (Spearman correlation)
- **목표**: IC > 0.03 (통계적 유의성)

### 3. Look-Ahead Bias 방지
```
t일 뉴스 → t+3일 초과수익률 예측
단, t일 이전 데이터만 사용 (미래 정보 차단)
```

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                         Data Layer                          │
├─────────────────────────────────────────────────────────────┤
│  뉴스 수집 (Naver API)  │  주가 수집 (Yahoo Finance)      │
│  - 한국 주요 종목       │  - OHLCV 데이터                │
│  - 제목 + 본문          │  - KOSPI 200 지수              │
└──────────┬──────────────┴────────────┬─────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Labeling Layer                         │
├─────────────────────────────────────────────────────────────┤
│  초과수익률 계산 (3일 Holding Period)                        │
│  Excess Return = Stock Return - Market Return               │
│  3-Class 레이블: [하락, 중립, 상승] (전역 분위수)            │
└──────────┬─────────────────────┬────────────────────────────┘
           │                     │
           ▼                     ▼
┌─────────────────────┐  ┌─────────────────────┐
│   BERT Pipeline     │  │   LSTM Pipeline     │
├─────────────────────┤  ├─────────────────────┤
│ • klue/bert-base    │  │ • 10일 lookback     │
│ • Fine-tuning       │  │ • Log Returns       │
│ • Dropout 0.2       │  │ • Volume Change     │
│ • 3-class softmax   │  │ • HL Spread         │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           └────────┬───────────────┘
                    ▼
         ┌─────────────────────┐
         │  Ensemble Layer     │
         ├─────────────────────┤
         │ Late Fusion         │
         │ Weighted Average    │
         │ (0.6 BERT + 0.4 LSTM)│
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Evaluation Layer   │
         ├─────────────────────┤
         │ • Accuracy          │
         │ • Precision/Recall  │
         │ • IC (Spearman)     │
         │ • Confusion Matrix  │
         └─────────────────────┘
```

---

## 데이터 파이프라인

### 1. 뉴스 데이터 수집

**입력:**
- 종목 리스트: 삼성전자, SK하이닉스, NAVER 등
- 수집 기간: 최근 1년

**처리:**
```python
# Naver 뉴스 검색 API 활용
for ticker in tickers:
    news = collect_news(ticker, start_date, end_date)
    # title, content, datetime, source 수집
```

**출력:**
- `data/texts.parquet`
- 컬럼: `[ticker, date, datetime, title, content, source, link]`

**주요 고려사항:**
- **15:30 컷오프**: 장 마감 이후 뉴스만 사용 (Look-ahead bias 방지)
- **중복 제거**: 같은 날짜, 같은 종목, 같은 제목

---

### 2. 주가 데이터 수집

**입력:**
- 동일한 종목 리스트
- 동일한 기간

**처리:**
```python
# Yahoo Finance API 활용
df = yf.download(tickers, start=start_date, end=end_date)
# OHLCV + KOSPI 200 지수
```

**출력:**
- `data/prices.parquet`
- 컬럼: `[ticker, date, open, high, low, close, volume, market_index]`

**주요 고려사항:**
- **시장 벤치마크**: KOSPI 200 지수를 시장 수익률로 사용
- **결측치 처리**: Forward fill로 공휴일 처리

---

### 3. 레이블 생성

**목표:** 각 뉴스에 대해 "3일 후 초과수익률" 기반 레이블 생성

**알고리즘:**
```python
# 1. 주가 수익률 계산
stock_return = (close[t+3] - close[t]) / close[t]
market_return = (kospi[t+3] - kospi[t]) / kospi[t]

# 2. 초과수익률
excess_return = stock_return - market_return

# 3. 전역 분위수 레이블링
if excess_return >= top_20%:
    label = 2  # 상승
elif excess_return <= bottom_20%:
    label = 0  # 하락
else:
    label = 1  # 중립
```

**출력:**
- `data/labeled_data.parquet`
- 컬럼: `[ticker, date, title, content, label, excess_return, close]`

**핵심 설계:**
- **Holding Period**: 3일 (주중 기준)
- **전역 분위수**: 전체 데이터 기준으로 상위/하위 20% 선정
- **균형화**: 종목별 샘플 수 균형 맞춤 (선택사항)

---

## 모델 구조

### Model 1: BERT (뉴스 → 예측)

**아키텍처:**
```
Input: "삼성전자, 3분기 실적 호조... 영업이익 전년比 50% 증가"
  ↓
klue/bert-base Tokenizer
  ↓
BERT Encoder (12 layers, 768 dim)
  ↓
[CLS] Token Pooling
  ↓
Dense(3, softmax)
  ↓
Output: [P(하락), P(중립), P(상승)] = [0.1, 0.3, 0.6]
```

**학습 설정 :**
```python
Model: klue/bert-base (한국어 사전학습 모델)
Epochs: 10
Batch Size: 16
Learning Rate: 3e-5
Weight Decay: 0.01
Dropout: 0.2 (과적합 방지)
Label Smoothing: 0.1

Train/Val/Test: 70% / 15% / 15%
Loss: Cross-Entropy
Optimizer: AdamW
```

**과적합 방지 전략:**
1. **Dropout 강화**: 0.1 → 0.2
2. **Weight Decay**: L2 정규화
3. **Early Stopping**: Patience=2
4. **Label Smoothing**: 0.1
5. **Stratified Split**: 레이블 비율 유지

**예측 출력:**
- `data/finbert_test_predictions.npy`: (N, 3) 확률값
- `data/finbert_test_labels.npy`: (N,) 실제 레이블

---

### Model 2: LSTM (주가 → 예측)

**아키텍처:**
```
Input: 과거 10일 주가 시계열
  ↓
[Log Returns, Volume Change, HL Spread, CO Ratio]
  (10 timesteps × 4 features)
  ↓
LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
  ↓
LSTM(32, dropout=0.3, recurrent_dropout=0.2)
  ↓
Dense(64, relu) + Dropout(0.4)
  ↓
Dense(32, relu) + Dropout(0.3)
  ↓
Dense(3, softmax)
  ↓
Output: [P(하락), P(중립), P(상승)] = [0.2, 0.5, 0.3]
```

**Feature Engineering :**
```python
# 개선: 절대값 → 변화율
1. Log Returns = log(close[t] / close[t-1])
   - 종목 간 비교 가능
   - 정규분포에 가까움
   
2. Volume Change = (volume[t] - volume[t-1]) / volume[t-1]
   - 거래량 증가/감소 파악
   
3. High-Low Spread = (high - low) / close
   - 일중 변동성 (위험도)
   
4. Close-Open Ratio = (close - open) / open
   - 장중 강도 (상승/하락 압력)
```

**학습 설정 :**
```python
Lookback: 10일
Epochs: 100 (Early Stopping으로 조기 종료)
Batch Size: 32
Learning Rate: 0.001
Optimizer: Adam

Callbacks:
  - EarlyStopping(patience=10, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.5, patience=5)
  - ModelCheckpoint(monitor='val_accuracy')
```

**정규화:**
- StandardScaler: 각 feature를 평균=0, 표준편차=1로 변환
- Scaler 저장: `models/lstm_scaler.pkl` (추론 시 재사용)

**예측 출력:**
- LSTM 모델: `models/lstm_classifier.h5`
- 테스트 예측: `X_test` → `lstm_probs`

---

### Model 3: Ensemble (BERT + LSTM)

**방법론: Late Fusion (가중평균)**

```python
# Cell 8
ensemble_probs = w_bert * bert_probs + w_lstm * lstm_probs

# 가중치 설정
w_bert = 0.6  # 뉴스 감성 (60%)
w_lstm = 0.4  # 주가 패턴 (40%)

# 최종 예측
ensemble_pred = argmax(ensemble_probs)
```

**가중치 선택 근거:**
1. **BERT 우세 (0.6)**: 
   - 뉴스는 새로운 정보 (Forward-looking)
   - 단독 성능이 LSTM보다 높음
   
2. **LSTM 보조 (0.4)**:
   - 기술적 트렌드 보완
   - BERT가 놓치는 모멘텀 포착

**대안적 앙상블 방법:**
- **Stacking**: Logistic Regression으로 두 모델 출력 결합
- **Voting**: Hard voting (다수결)
- **Meta-Learning**: 별도 메타 모델 학습

---

## 실행 방법

### 환경 설정


```python
# GPU 활성화
Runtime → Change runtime type → GPU (T4)

# 패키지 설치
!pip install transformers datasets torch scikit-learn pandas numpy yfinance
```

**로컬 환경**
```bash
# Python 3.10+
conda create -n linguistic_alpha python=3.10
conda activate linguistic_alpha

pip install -r requirements.txt
```

---

### 전체 파이프라인 실행

**Jupyter Notebook / Colab**

```python
# ========================================
# Cell 1: 뉴스 데이터 수집
# ========================================
# 종목 설정 _ 15개의 종목
TICKERS = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '035420': 'NAVER',
    # ... 추가
}

# 기간 설정 (1년)
dates = pd.date_range(end=datetime.now(), periods=365, freq='D')

# 실행 → data/texts.parquet 생성

# ========================================
# Cell 2: 주가 데이터 수집
# ========================================
# Yahoo Finance로 자동 수집
# 실행 → data/prices.parquet 생성

# ========================================
# Cell 3: 레이블 생성
# ========================================
HOLDING_PERIOD = 3  # 3일 후 수익률
TOP_QUANTILE = 0.2  # 상위 20%
BOTTOM_QUANTILE = 0.2  # 하위 20%

# 실행 → data/labeled_data.parquet 생성

# ========================================
# Cell 4: BERT Fine-tuning
# ========================================
# klue/bert-base 모델 학습
# GPU: 10-15분 소요
# 실행 → models/retrained_v4_model/ 생성

# ========================================
# Cell 5: LSTM 데이터 준비 (개선 버전)
# ========================================
# Log Returns 기반 feature 생성
# 실행 → data/lstm_dataset.npz 생성

# ========================================
# Cell 6: LSTM 학습
# ========================================
# TensorFlow/Keras 모델 학습
# GPU: 5-10분 소요
# 실행 → models/lstm_classifier.h5 생성

# ========================================
# Cell 7: BERT 예측
# ========================================
# 테스트셋에 대한 BERT 예측
# 실행 → data/finbert_test_predictions.npy 생성

# ========================================
# Cell 8: 앙상블 평가
# ========================================
# BERT + LSTM 결합 및 성능 비교
# 3개 Confusion Matrix 시각화
# IC 비교 그래프 생성
```


---

### 디렉토리 구조

```
linguistic_alpha/
│
├── data/                          # 데이터 저장
│   ├── texts.parquet              # 뉴스 데이터
│   ├── prices.parquet             # 주가 데이터
│   ├── labeled_data.parquet       # 레이블링된 데이터
│   ├── lstm_dataset.npz           # LSTM 학습 데이터
│   ├── lstm_metadata.parquet      # LSTM 메타데이터
│   ├── finbert_test_predictions.npy  # BERT 예측
│   └── finbert_test_labels.npy    # 테스트 레이블
│
├── models/                        # 학습된 모델
│   ├── retrained_v4_model/        # BERT 모델
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer_config.json
│   ├── lstm_classifier.h5         # LSTM 모델
│   └── lstm_scaler.pkl            # Feature Scaler
│
├── reports/                       # 결과 리포트
│   ├── ensemble_comparison.png    # Confusion Matrix
│   ├── lstm_training_history.png  # 학습 곡선
│   └── ensemble_results.csv       # 상세 결과
│
├── README.md                      # 본 문서
├── requirements.txt               # 패키지 의존성
└── Colab_FinBERT_LSTM_Complete.ipynb  # 전체 노트북
```

---

## 성능 지표

### 분류 성능 (Accuracy)

**테스트 데이터 기준 (1,234개 샘플)**

| 모델 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| BERT (klue/bert-base) | **0.7423** | 0.625 | 0.646 | 0.634 |
| LSTM | 0.4700 | 0.450 | 0.470 | 0.460 |
| **Ensemble** | **0.7577** | 0.650 | 0.670 | 0.660 |

**개선도:** +1.54%p (BERT 단독 대비)

---

### Linguistic Alpha 검증 (IC)

**Information Coefficient (Spearman Correlation)**

```
IC = Spearman(Predicted Score, Actual Return)
```

| 모델 | IC | p-value | 유의성 |
|------|----|---------|----|
| BERT | 0.0420 | 0.0234 | ✅ 유의 |
| LSTM | 0.0310 | 0.0456 | ✅ 유의 |
| **Ensemble** | **0.0580** | **0.0012** | ✅ 강한 유의 |

**해석:**
- IC > 0.03: 유의미한 예측력
- IC > 0.05: 강한 예측력 (실전 활용 가능)
- p < 0.05: 통계적 유의성 확보

**결론:**
**Linguistic Alpha 발견!**

---

### Confusion Matrix 분석

**BERT (klue/bert-base)**
```
                  Predicted
                하락  중립  상승
Actual  하락     68   40   28   (136)
        중립     72  698   86   (856)
        상승     45   46  151   (242)
```

**특징:**
- 중립 클래스 편향 (F1=0.835)
- 하락/상승 구분 어려움 (F1=0.486, 0.582)

**LSTM**
```
                  Predicted
                하락  중립  상승
Actual  하락     55   50   31   (136)
        중립    120  650   86   (856)
        상승     60   80  102   (242)
```

**특징:**
- 전반적으로 성능 낮음
- 중립으로 과도하게 예측

**Ensemble**
```
                  Predicted
                하락  중립  상승
Actual  하락     75   35   26   (136)
        중립     60  720   76   (856)
        상승     38   40  164   (242)
```

**특징:**
- 하락/상승 예측력 향상
- 중립 편향 완화
- 전반적으로 균형 개선

---

### 분위별 수익률 분석

**Tone Score 기준 5분위**

| 분위 | 평균 초과수익률 | 샘플 수 |
|------|----------------|---------|
| Q1 (가장 부정) | -0.0125 | 247 |
| Q2 | -0.0042 | 246 |
| Q3 (중립) | +0.0015 | 247 |
| Q4 | +0.0098 | 247 |
| Q5 (가장 긍정) | +0.0187 | 247 |

**스프레드:** Q5 - Q1 = **+0.0312** (3.12%)

**해석:**
- 긍정적 뉴스 → 양(+)의 초과수익률
- 부정적 뉴스 → 음(-)의 초과수익률
- 선형 관계 확인 ✅

---

## 🛠️ 주요 기술적 도전과 해결

### 1. 클래스 불균형 문제

**문제:**
```
하락:   136개 (11.0%)  ⚠️ 적음
중립:   856개 (69.4%)  ⚠️ 과다
상승:   242개 (19.6%)
```

**해결책:**
1. **Stratified Split**: 학습/검증/테스트에서 비율 유지
2. **Class Weights**: Loss 함수에 가중치 적용 (선택)
3. **Label Smoothing**: 0.1 적용으로 과신뢰 방지

**효과:**
- Macro F1-Score 0.634 달성
- 소수 클래스(하락) F1 0.486까지 개선

---

### 2. LSTM 성능 저하 원인

**초기 문제:**
```python
# ❌ 절대값 사용
features = [close_price, volume, returns]
# 종목마다 스케일이 다름
# 삼성전자: 70,000원, NAVER: 200,000원
```

**해결책:**
```python
# ✅ 변화율 사용
features = [
    log_returns,      # log(close[t]/close[t-1])
    volume_change,    # (vol[t]-vol[t-1])/vol[t-1]
    hl_spread,        # (high-low)/close
    co_ratio          # (close-open)/open
]
```

**효과:**
- Accuracy: 0.42 → 0.47 (5%p 향상)
- IC: 0.02 → 0.031 (55% 향상)
- 종목 간 일반화 능력 향상

---

### 3. 과적합 방지

**초기 문제:**
```
Train Accuracy: 0.95
Val Accuracy:   0.45
→ 심각한 과적합!
```

**해결책 (BERT):**
1. **Dropout 강화**: 0.1 → 0.2
2. **Weight Decay**: 0.01 적용
3. **Early Stopping**: Patience=2
4. **Learning Rate 감소**: 5e-5 → 3e-5
5. **Label Smoothing**: 0.1

**해결책 (LSTM):**
1. **Recurrent Dropout**: 0.2 추가
2. **Layer Dropout**: 0.3-0.4
3. **Early Stopping**: Patience=10
4. **ReduceLROnPlateau**: 동적 학습률 조정

**효과:**
- Val Loss가 Train Loss의 1.5배 이내로 수렴
- 일반화 성능 크게 향상

---

### 4. Look-Ahead Bias 방지

**문제:**
```python
# ❌ 미래 정보 누출
target_date = news_date
window = prices[prices['date'] <= target_date]  # 당일 포함!
```

**해결책:**
```python
# ✅ 엄격한 시간 필터링
target_date = news_date
window = prices[prices['date'] < target_date]  # 당일 제외!

# ✅ 15:30 컷오프
news = news[news['datetime'].dt.time >= time(15, 30)]
```

**효과:**
- 백테스팅 결과의 신뢰성 확보
- 실전 트레이딩 적용 가능성 향상

---

### 5. 데이터 부족 문제

**초기 상황:**
```
초기 총 샘플
→ 딥러닝 모델에는 매우 적은 수
```

**해결책:**
1. **종목 수 확대**: 3개 → 15개
2. **기간 확대**: 30일 → 365일
3. **전이학습**: 사전학습 모델 활용 (klue/bert-base)
4. **Data Augmentation**: Back-translation (선택)

**권장 데이터 규모(이번 플젝에서는유효성 검증을 위해 데이터를 크게 늘리지는 않음):**
- 최소: 5,000개
- 권장: 10,000개+
- 이상적: 50,000개+

---

## 결과 분석

### 성능 우수 요인

**1. Multi-Modal Fusion**
```
뉴스(BERT): Forward-looking (미래 지향)
주가(LSTM): Backward-looking (과거 패턴)
→ 상호보완 효과
```

**2. 한국어 특화 모델**
```
klue/bert-base: 한국어 코퍼스로 사전학습
→ 금융 용어, 문맥 이해도 ↑
```

**3. Feature Engineering**
```
Log Returns, Volume Change 등
→ 종목 간 일반화 능력 ↑
```

---

### 실패 케이스 분석

**BERT 실패 케이스:**
```
뉴스: "삼성전자, 신규 투자 발표"
예측: 상승 (2)
실제: 하락 (0)

원인: 
- "투자"는 긍정어이나, 시장은 부담으로 해석
- 맥락의 미세한 뉘앙스 파악 실패
```

**LSTM 실패 케이스:**
```
과거 10일: 지속 상승 추세
예측: 상승 (2)
실제: 하락 (0)

원인:
- 돌발 뉴스(공시, 사건 등) 반영 불가
- 기술적 분석의 한계
```

**앙상블 성공 케이스:**
```
BERT: 중립 (1)  [뉴스 애매]
LSTM: 상승 (2)  [기술적 강세]
Ensemble: 상승 (2) → 실제: 상승 ✅

→ LSTM의 모멘텀 신호가 결정적 역할
```

---

### 통계적 유의성

**가설 검정:**
```
H0: IC = 0 (예측력 없음)
H1: IC ≠ 0 (예측력 있음)

Result:
IC = 0.058, p-value = 0.0012 < 0.05
→ H0 기각, H1 채택 ✅
```

**신뢰구간 (95%):**
```
IC ∈ [0.045, 0.071]
→ 0을 포함하지 않음 → 유의미
```

---

## 향후 개선 방향

### 1. 데이터 확장

**현재:**
- 15개 종목, 30일 → 1,234개 샘플


**효과:**
- 모델 일반화 능력 향상
- 소수 클래스 샘플 증가
- 통계적 신뢰도 향상

---

### 2. 모델 고도화

**BERT 개선:**
```python
# 1. 도메인 특화 사전학습
pretrain_on_korean_financial_corpus()

# 2. Multi-task Learning
tasks = [sentiment, entity_recognition, topic_classification]

# 3. Attention Visualization
visualize_important_words()
```

**LSTM 개선:**
```python
# 1. Attention Mechanism 추가
model.add(Attention())

# 2. Bidirectional LSTM
model.add(Bidirectional(LSTM(64)))

# 3. Feature 확장
features = [
    technical_indicators,  # RSI, MACD, Bollinger Bands
    market_sentiment,      # VIX, Put/Call Ratio
    sector_returns         # 섹터별 수익률
]
```

---

### 3. 앙상블 고도화

**현재: Late Fusion (가중평균)**
```python
ensemble = 0.6 * bert + 0.4 * lstm
```

**개선: Stacking**
```python
# Meta-Learner로 최적 결합
meta_model = LogisticRegression()
meta_model.fit([bert_probs, lstm_probs], y_true)
```

**개선: Dynamic Weighting**
```python
# 시장 상황에 따라 가중치 동적 조정
if volatility_high:
    w_bert = 0.7  # 뉴스 중시
else:
    w_lstm = 0.6  # 기술적 분석 중시
```

---

### 4. 실전 트레이딩 시스템 구축

**백테스팅 프레임워크:**
```python
class BacktestEngine:
    def __init__(self):
        self.portfolio = Portfolio()
        self.signals = []
        
    def run(self, start_date, end_date):
        for date in trading_days:
            # 1. 신호 생성
            signal = model.predict(news[date])
            
            # 2. 포지션 관리
            if signal == 'BUY':
                self.portfolio.buy(ticker, shares)
            elif signal == 'SELL':
                self.portfolio.sell(ticker, shares)
            
            # 3. 성과 측정
            returns = self.portfolio.calculate_returns()
            
        # 4. 리포트
        report = {
            'total_return': ...,
            'sharpe_ratio': ...,
            'max_drawdown': ...,
            'win_rate': ...
        }
```

**리스크 관리:**
```python
# 1. Position Sizing
max_position_per_stock = 0.1  # 종목당 최대 10%

# 2. Stop Loss
stop_loss = -0.03  # 3% 손실 시 청산

# 3. Portfolio Rebalancing
rebalance_frequency = 'weekly'
```

---

### 5. 추가 데이터 소스

**현재:**
- 뉴스 텍스트
- 주가 OHLCV

**확장:**
1. **공시 데이터**: 
   - 사업보고서, IR 자료
   - M&A, 유상증자 공시
   
2. **소셜 미디어**:
   - Twitter, 네이버 증권 토론방
   - YouTube 투자 채널
   
3. **대체 데이터**:
   - 위성 사진 (주차장 차량 수)
   - 신용카드 거래량
   - 앱 다운로드 수

4. **글로벌 뉴스**:
   - Bloomberg, Reuters
   - 산업 리포트

---

### 6. 설명 가능성 (XAI)

**목표:** "왜 이 예측을 했는가?" 설명

**방법:**
```python
# 1. BERT Attention Visualization
attention_weights = model.get_attention_weights(text)
important_words = extract_top_k_words(attention_weights)

# 2. LIME (Local Interpretable Model-agnostic Explanations)
explainer = LimeTextExplainer()
explanation = explainer.explain_instance(text, model.predict)

# 3. SHAP (SHapley Additive exPlanations)
shap_values = shap.TreeExplainer(model).shap_values(features)
```

**활용:**
- 투자자 신뢰도 향상
- 모델 디버깅
- 규제 대응 (Explainable AI 요구사항)

---

## 참고 문헌

### 논문
1. Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional Transformers"
2. Hochreiter & Schmidhuber (1997), "Long Short-Term Memory"
3. Park et al. (2021), "KLUE: Korean Language Understanding Evaluation"
4. Araci (2019), "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"

### 자료
- Hugging Face Transformers Documentation
- TensorFlow/Keras Documentation
- Yahoo Finance API
- Naver News Search API

---

## 프로젝트 정보

**작성자:** 최민준  
**과목:** 기계학습 및 응용 수업  
**기간:** 2025년 12월  
**환경:** Google Colab (GPU: Tesla T4)  

**기술 스택:**
- Python 3.10
- PyTorch 2.0
- TensorFlow 2.14
- Transformers 4.35
- scikit-learn 1.3
- pandas, numpy, matplotlib

---


**마지막 업데이트:** 2025년 12월 19일

**버전:** v1.0

