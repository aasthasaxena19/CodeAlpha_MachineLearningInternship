# 🎤 Emotion Recognition from Speech — CodeAlpha Internship Task 2

A Deep Learning model that listens to someone speaking and predicts their **emotion** (Happy, Sad, Angry, Fearful, Disgust, Surprised, Neutral).

---

## 🎯 Objective
Recognize human emotions from speech audio using MFCC feature extraction and a CNN + LSTM neural network.

## 📁 Project Structure
```
CodeAlpha_EmotionRecognition/
│
├── emotion_recognition.py         ← Main Python script (heavily commented)
├── emotion_recognition_results.png ← Auto-generated charts
├── emotion_model.keras            ← Saved trained model
├── requirements.txt               ← Libraries needed
└── README.md                      ← This file
```

---

## 🧠 How It Works — Step by Step

```
🎙️ Raw Audio (.wav file)
        ↓
📊 MFCC Feature Extraction     ← Converts sound into 187 numbers
   + Chroma + Mel + Contrast
        ↓
🔧 Preprocessing               ← Normalize, split into train/test
        ↓
🏗️ CNN + LSTM Model            ← Learn patterns from features
   Conv1D → Conv1D → Conv1D
      → LSTM → Dense → Softmax
        ↓
😊 Predicted Emotion
```

---

## 📊 What are MFCCs?

**Mel-Frequency Cepstral Coefficients** — the key technique in audio ML:

| Raw Audio | vs | MFCCs |
|-----------|----|-------|
| 66,150 numbers (3 seconds) | → | 40 compact numbers |
| Hard to use directly | → | Perfect for ML models |
| Stores every vibration | → | Stores sound "texture" |

Different emotions produce **different MFCC patterns**:
- 😠 Angry → High energy, sharp frequencies
- 😢 Sad → Low energy, slow frequencies  
- 😊 Happy → Bright, varied frequencies

---

## 🗄️ Dataset Options

### Option A: RAVDESS Dataset (Recommended for submission)
1. Download from: https://zenodo.org/record/1188976
2. Extract to `./ravdess/` folder
3. Set `USE_REAL_DATA = True` in the script
4. Run → trains on 1440 real audio files!

### Option B: Synthetic Demo (Default — works without download)
- Runs immediately without any dataset
- Uses mathematically simulated MFCC distributions per emotion
- Great for understanding the pipeline

---

## 🚀 How to Run

```bash
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Run the script
python emotion_recognition.py

# Optional: Predict on your own voice recording
```

To predict on your own audio:
```python
predict_emotion(audio_path="my_recording.wav")
```

---

## 🏗️ Model Architecture

```
Input: (187 features, 1)
  ↓
Conv1D(64, kernel=5) + BatchNorm + MaxPool + Dropout(0.3)
  ↓
Conv1D(128, kernel=5) + BatchNorm + MaxPool + Dropout(0.3)
  ↓
Conv1D(256, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
  ↓
LSTM(128) + Dropout(0.4)
  ↓
Dense(256, relu) + BatchNorm + Dropout(0.5)
  ↓
Dense(128, relu) + Dropout(0.4)
  ↓
Dense(7, softmax) → Emotion Probabilities
```

---

## 📚 Key Concepts (For Beginners)

| Concept | Simple Explanation |
|---------|-------------------|
| **MFCC** | Converts audio wave into compact numbers |
| **CNN** | Finds patterns in data using sliding filters |
| **LSTM** | Neural network with memory — understands sequences |
| **Dropout** | Randomly turns off neurons during training to prevent memorization |
| **Softmax** | Converts outputs to probabilities that sum to 1 |
| **Epoch** | One full pass through all training data |
| **Batch Size** | How many samples processed at once |

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading and MFCC extraction |
| `tensorflow/keras` | Building and training the neural network |
| `scikit-learn` | Data splitting, encoding, evaluation |
| `numpy` | Numerical computations |
| `matplotlib/seaborn` | Charts and visualizations |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Dataset | RAVDESS (real) / Synthetic |
| Emotions | 7 classes |
| Features | 187 (MFCC + Chroma + Mel + Contrast) |
| Model | CNN + LSTM |

---

## 👤 Author
CodeAlpha Machine Learning Internship — Task 2
