# 🤖 CodeAlpha Machine Learning Internship

> **Intern:** Aastha Saxena
> **Domain:** Machine Learning
> **Organization:** [CodeAlpha](https://www.codealpha.tech)
> **Duration:** 20th April 2026 – 20th May 2026

---

## 📌 About This Repository

This repository contains my completed Machine Learning projects as part of the **CodeAlpha Internship Program**. Each task demonstrates a different area of machine learning — from classical algorithms to deep learning and audio intelligence.

---

## 🗂️ Tasks Completed

| # | Task | Techniques Used | Status |
|---|------|----------------|--------|
| 1 | 💳 Credit Scoring Model | Logistic Regression, Random Forest | ✅ Done |
| 2 | 🎤 Emotion Recognition from Speech | CNN, LSTM, MFCC, Deep Learning | ✅ Done |

---

## 💳 Task 1 — Credit Scoring Model

### 🎯 Objective
Predict whether a loan applicant will **repay** or **default** on their loan, based on their financial history.

### 🧠 How It Works
```
Financial Data (income, credit score, debts, etc.)
        ↓
Feature Engineering (loan-to-income ratio, payment history score)
        ↓
ML Models (Logistic Regression + Random Forest)
        ↓
Prediction: ✅ Will Repay  or  ❌ Will Default
```

### 📊 Features Used
| Feature | Description |
|---------|-------------|
| Age | Applicant's age |
| Income | Annual income |
| Loan Amount | Borrowed amount |
| Credit Score | 300–850 range |
| Late Payments | Number of past late payments |
| Debt-to-Income Ratio | Total debt / Annual income |
| Employment Years | Years of work experience |
| Education | Highest degree obtained |
| + Engineered Features | Loan-to-income ratio, Payment history score |

### 🤖 Models Trained
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~87% | ~0.955 |
| Random Forest | ~88% | ~0.953 |

### 🏆 Key Result
The **Random Forest** model achieved **88% accuracy** with an **AUC-ROC of 0.953**, correctly identifying high-risk applicants with **94.5% default probability** for risky profiles.

### 📁 Files
```
Task1_CreditScoring/
├── credit_scoring.py       ← Main script (heavily commented)
├── requirements.txt        ← Dependencies
└── README.md
```

### ▶️ How to Run
```bash
cd Task1_CreditScoring
pip install -r requirements.txt
python credit_scoring.py
```

---

## 🎤 Task 2 — Emotion Recognition from Speech

### 🎯 Objective
Listen to someone speaking and automatically predict their **emotion** — Happy, Sad, Angry, Fearful, Disgust, Surprised, or Neutral.

### 🧠 How It Works
```
🎙️ Raw Audio (.wav file)
        ↓
📊 Feature Extraction
   MFCC (40) + Chroma (12) + Mel Spectrogram (128) + Spectral Contrast (7)
   = 187 total features per audio clip
        ↓
🏗️ Deep Learning Model
   Conv1D(64) → Conv1D(128) → Conv1D(256) → LSTM(128) → Dense → Softmax
        ↓
😊 Predicted Emotion + Confidence %
```

### 🔊 What is MFCC?
MFCC (Mel-Frequency Cepstral Coefficients) converts raw audio into compact numerical features that capture the **texture and tone** of speech — similar to how human ears process sound. Different emotions produce different MFCC patterns:
- 😠 Angry → High energy, sharp frequency spikes
- 😢 Sad → Low energy, slow declining pattern
- 😊 Happy → Bright, varied, upbeat frequency spread

### 🏗️ Model Architecture
```
Input (187 features)
  → Conv1D(64)  + BatchNorm + MaxPool + Dropout(0.3)
  → Conv1D(128) + BatchNorm + MaxPool + Dropout(0.3)
  → Conv1D(256) + BatchNorm + MaxPool + Dropout(0.3)
  → LSTM(128) + Dropout(0.4)
  → Dense(256, relu) + Dropout(0.5)
  → Dense(128, relu) + Dropout(0.4)
  → Dense(7, softmax) → Emotion Probabilities
Total Parameters: 406,791
```

### 🗄️ Dataset
- **Recommended:** [RAVDESS](https://zenodo.org/record/1188976) — 1,440 real emotional speech audio files from 24 professional actors
- **Demo Mode:** Runs with synthetic data (no download needed)

### 📈 Emotions Recognized
`neutral` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

### 📁 Files
```
Task2_EmotionRecognition/
├── emotion_recognition.py       ← Main script (heavily commented)
├── requirements.txt             ← Dependencies
└── README.md
```

### ▶️ How to Run
```bash
cd Task2_EmotionRecognition
pip install -r requirements.txt
python emotion_recognition.py
```

**To predict on your own voice recording:**
```python
predict_emotion(audio_path="my_recording.wav")
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Python 3.x` | Programming language |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `tensorflow / keras` | Deep learning neural networks |
| `librosa` | Audio loading and MFCC feature extraction |
| `matplotlib` | Data visualization and charts |
| `seaborn` | Statistical data visualization |

---

## 📚 Key Concepts Learned

### Machine Learning Fundamentals
- **Feature Engineering** — Creating new features to improve model performance
- **Train/Test Split** — Why we never test on training data
- **Cross-Validation** — More robust model evaluation
- **Overfitting** — Why a model can "memorize" rather than "learn"

### Evaluation Metrics
| Metric | Meaning |
|--------|---------|
| **Accuracy** | % of correct predictions overall |
| **Precision** | Of all predicted positives, how many were actually positive? |
| **Recall** | Of all actual positives, how many did we catch? |
| **F1-Score** | Balance between Precision and Recall |
| **ROC-AUC** | How well the model ranks positives vs negatives (1.0 = perfect) |
| **Confusion Matrix** | Visual table of correct vs incorrect predictions |

### Deep Learning Concepts
| Concept | Meaning |
|---------|---------|
| **CNN (Conv1D)** | Detects local patterns in data using sliding filters |
| **LSTM** | Neural network with memory — understands time sequences |
| **Dropout** | Randomly disables neurons during training to prevent overfitting |
| **BatchNorm** | Normalizes layer outputs for faster, stable training |
| **Softmax** | Converts outputs to probabilities summing to 1.0 |
| **Early Stopping** | Automatically stops training when model stops improving |

---

## 📂 Repository Structure

```
CodeAlpha_MachineLearning/
│
├── Task1_CreditScoring/
│   ├── credit_scoring.py
│   ├── requirements.txt
│   └── README.md
│
├── Task2_EmotionRecognition/
│   ├── emotion_recognition.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md             ← This file
```

---

## 🚀 Getting Started

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/CodeAlpha_MachineLearning.git
cd CodeAlpha_MachineLearning

# 2. Run Task 1 (Credit Scoring)
cd Task1_CreditScoring
pip install -r requirements.txt
python credit_scoring.py

# 3. Run Task 2 (Emotion Recognition)
cd ../Task2_EmotionRecognition
pip install -r requirements.txt
python emotion_recognition.py
```

---

## 🎓 About CodeAlpha

[CodeAlpha](https://www.codealpha.tech) is a leading software development company providing internship opportunities in AI, Machine Learning, Web Development, and more. This internship provided hands-on experience building real-world ML solutions using Python and industry-standard libraries.

---

## 📞 Contact

- 🌐 **CodeAlpha Website:** [www.codealpha.tech](https://www.codealpha.tech)
- 📧 **Email:** services@codealpha.tech
- 💼 **LinkedIn:** [Your LinkedIn Profile URL]
- 🐙 **GitHub:** [Your GitHub Profile URL]

---

*Built with ❤️ during the CodeAlpha Machine Learning Internship*
