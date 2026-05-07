# ============================================================
# TASK 2: EMOTION RECOGNITION FROM SPEECH
# CodeAlpha Machine Learning Internship
# ============================================================
# 🎯 GOAL: Listen to someone speaking and predict their emotion
#          (Happy, Sad, Angry, Fearful, Disgusted, Surprised, Neutral)
#
# 📚 WHAT YOU'LL LEARN:
#   - How audio signals work (sound as numbers)
#   - MFCC feature extraction (converting audio → data)
#   - Building a CNN + LSTM deep learning model
#   - Training a neural network for classification
#
# 📂 DATASET: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)
#   Real dataset → download from:
#   https://zenodo.org/record/1188976
#   Or use: pip install ravdess  (if available)
#
#   This script works in TWO modes:
#   MODE A: If you have RAVDESS dataset → set USE_REAL_DATA = True
#   MODE B: Synthetic demo (runs without any download) → USE_REAL_DATA = False
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow startup messages

# Audio processing library
import librosa                   # Library for audio analysis
import librosa.display           # For visualizing audio

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Deep Learning (Neural Networks)
import tensorflow as tf
from tensorflow.keras.models import Sequential         # Stack layers one by one
from tensorflow.keras.layers import (
    Dense,          # Fully connected layer (standard NN layer)
    Dropout,        # Randomly turns off neurons to prevent overfitting
    Conv1D,         # 1D Convolution — detects patterns in sequences
    MaxPooling1D,   # Reduces size, keeps important features
    LSTM,           # Long Short-Term Memory — understands time sequences
    Flatten,        # Converts 2D data to 1D for Dense layers
    BatchNormalization  # Normalizes outputs, makes training stable
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical      # Converts labels to one-hot format

# ============================================================
# CONFIGURATION — Change these settings here
# ============================================================
USE_REAL_DATA   = False          # Set True if you downloaded RAVDESS dataset
RAVDESS_PATH    = "./ravdess/"   # Path to RAVDESS folder (if using real data)
SAMPLE_RATE     = 22050          # Audio sample rate (standard)
DURATION        = 3              # Seconds of audio to analyze per file
N_MFCC          = 40            # Number of MFCC features to extract
                                 # (Think: 40 different sound characteristics)
EMOTIONS = {                     # RAVDESS emotion codes → emotion names
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("=" * 60)
print("  EMOTION RECOGNITION FROM SPEECH - CodeAlpha Task 2")
print("=" * 60)


# ============================================================
# WHAT IS AN MFCC? — Beginner Explanation
# ============================================================
# Sound = air pressure waves → captured as numbers (waveform)
# Problem: Raw audio has 22,050 numbers per second → too many!
#
# MFCC (Mel-Frequency Cepstral Coefficients):
#   → Converts audio into a compact "fingerprint"
#   → 40 numbers that capture the "texture" of the sound
#   → Similar to how our ears actually process sound
#   → Happy voice has different MFCC pattern than Angry voice
#
# Think of it like this:
#   Audio waveform = raw ingredients
#   MFCC = nutritional facts label (compact summary)


def extract_features(audio_data, sample_rate=SAMPLE_RATE):
    """
    Extract MFCC features from audio data.
    
    INPUT:  audio_data = array of sound samples (numbers representing sound)
            sample_rate = how many samples per second (22050)
    OUTPUT: feature vector of shape (N_MFCC,) = 40 numbers
    """
    # Ensure consistent length (pad or trim to DURATION seconds)
    target_length = SAMPLE_RATE * DURATION
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]      # Trim if too long
    else:
        audio_data = np.pad(audio_data,              # Pad with zeros if too short
                            (0, max(0, target_length - len(audio_data))))

    # Extract MFCC features
    # n_mfcc=40 → gives us 40 coefficients
    # These 40 numbers summarize the sound's frequency characteristics
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=N_MFCC)

    # Also extract additional features for better accuracy:
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)  # Pitch info
    mel    = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate) # Mel spectrogram
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)

    # Take the mean across time → each feature becomes a single number
    features = np.concatenate([
        np.mean(mfcc, axis=1),       # 40 MFCC means
        np.mean(chroma, axis=1),     # 12 chroma means
        np.mean(mel, axis=1),        # 128 mel means
        np.mean(contrast, axis=1),   # 7 contrast means
    ])
    return features


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n📦 STEP 1: Loading Data...")

if USE_REAL_DATA and os.path.exists(RAVDESS_PATH):
    # -------------------------------------------------------
    # MODE A: Real RAVDESS Dataset
    # -------------------------------------------------------
    # RAVDESS filename format: 03-01-05-01-01-01-12.wav
    # Position 3 (index 2) = emotion code
    # -------------------------------------------------------
    print("📂 Loading real RAVDESS dataset...")
    X, y_labels = [], []

    for root, dirs, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                # Extract emotion from filename
                parts   = file.split('-')
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code, 'unknown')

                try:
                    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                    features = extract_features(audio, sr)
                    X.append(features)
                    y_labels.append(emotion)
                    print(f"  ✅ Loaded: {file} → {emotion}")
                except Exception as e:
                    print(f"  ⚠️  Skipped {file}: {e}")

    X = np.array(X)
    y_labels = np.array(y_labels)
    print(f"\n✅ Loaded {len(X)} real audio files")

else:
    # -------------------------------------------------------
    # MODE B: Synthetic Demo Dataset
    # (Simulates what MFCC features would look like per emotion)
    # -------------------------------------------------------
    print("🔬 Using synthetic demo data (real RAVDESS not found)")
    print("   To use real data: download RAVDESS & set USE_REAL_DATA=True")
    print("\n   Generating realistic synthetic MFCC feature distributions...")

    np.random.seed(42)
    emotions_list = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    n_per_emotion = 120   # 120 samples per emotion = 840 total
    feature_size  = 187   # 40 MFCC + 12 chroma + 128 mel + 7 contrast

    # Each emotion has a unique "mean" feature pattern
    # In real life, happy speech has different frequencies than sad speech
    emotion_means = {
        'neutral':   np.random.uniform(-0.1,  0.1,  feature_size),
        'happy':     np.random.uniform( 0.3,  0.6,  feature_size),
        'sad':       np.random.uniform(-0.5, -0.2,  feature_size),
        'angry':     np.random.uniform( 0.5,  0.9,  feature_size),
        'fearful':   np.random.uniform(-0.3,  0.0,  feature_size),
        'disgust':   np.random.uniform(-0.2,  0.2,  feature_size),
        'surprised': np.random.uniform( 0.1,  0.5,  feature_size),
    }

    X, y_labels = [], []
    for emotion in emotions_list:
        mean = emotion_means[emotion]
        # Add random noise around the emotion's "center"
        samples = mean + np.random.normal(0, 0.25, (n_per_emotion, feature_size))
        X.extend(samples)
        y_labels.extend([emotion] * n_per_emotion)

    X = np.array(X)
    y_labels = np.array(y_labels)
    print(f"✅ Synthetic dataset: {len(X)} samples, {X.shape[1]} features each")
    print(f"   Emotions: {emotions_list}")


# ============================================================
# STEP 2: PREPROCESS DATA
# ============================================================
print("\n🔧 STEP 2: Preprocessing Data...")

# Convert emotion text labels → numbers
# 'happy'→0, 'sad'→1, 'angry'→2, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
num_classes = len(le.classes_)
print(f"✅ Emotions found: {list(le.classes_)}")
print(f"✅ Number of emotion classes: {num_classes}")

# Convert to one-hot encoding
# Example: 'happy'(class 2) → [0, 0, 1, 0, 0, 0, 0]
# Neural networks work better with one-hot format for multi-class problems
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Scale features (normalize to same range)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape for CNN: CNN expects (samples, timesteps, features)
# We add a dimension: (samples, features, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Testing set:  {X_test.shape[0]} samples")
print(f"✅ Feature shape: {X_train_cnn.shape}")


# ============================================================
# STEP 3: BUILD THE DEEP LEARNING MODEL
# ============================================================
print("\n🧠 STEP 3: Building the CNN + LSTM Model...")

# -------------------------------------------------------
# WHY CNN + LSTM?
# -------------------------------------------------------
# CNN (Convolutional Neural Network):
#   → Great at finding LOCAL patterns in data
#   → Like finding "this part sounds sharp/angry"
#   → Uses sliding filters to detect features
#
# LSTM (Long Short-Term Memory):
#   → Great at understanding SEQUENCES over time
#   → Like remembering "the speech started soft then got loud"
#   → Has "memory" — remembers what came before
#
# Together: CNN extracts features → LSTM understands them over time
# -------------------------------------------------------

model = Sequential([

    # --- BLOCK 1: First CNN layer ---
    Conv1D(filters=64,          # 64 different filters (feature detectors)
           kernel_size=5,       # Each filter looks at 5 time-steps at once
           activation='relu',   # ReLU = "only keep positive values"
           padding='same',      # Keep same length after convolution
           input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),       # Normalize outputs → faster, stable training
    MaxPooling1D(pool_size=2),  # Shrink by half, keep important features
    Dropout(0.3),               # Randomly turn off 30% neurons → prevents overfitting

    # --- BLOCK 2: Second CNN layer ---
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # --- BLOCK 3: Third CNN layer ---
    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # --- BLOCK 4: LSTM layer ---
    # LSTM takes the CNN output and understands temporal patterns
    LSTM(128, return_sequences=False),
    Dropout(0.4),

    # --- BLOCK 5: Dense (fully connected) layers ---
    Dense(256, activation='relu'),  # 256 neurons
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),  # 128 neurons
    Dropout(0.4),

    # --- OUTPUT LAYER ---
    # num_classes neurons (one per emotion)
    # softmax = converts to probabilities that sum to 1
    # e.g., [0.05, 0.85, 0.02, 0.03, 0.02, 0.02, 0.01] → happy (index 1)
    Dense(num_classes, activation='softmax')
])

# Compile = tell model HOW to learn
model.compile(
    optimizer='adam',                       # Adam = smart learning rate algorithm
    loss='categorical_crossentropy',        # Loss function for multi-class problems
    metrics=['accuracy']                    # Track accuracy during training
)

print("\n--- Model Architecture ---")
model.summary()


# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================
print("\n🏋️ STEP 4: Training the Model...")

# Callbacks = helper functions that run during training
callbacks = [
    # Stop training early if model stops improving (saves time)
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True,
                  verbose=1),

    # Reduce learning rate if model gets stuck
    # (Like slowing down when you're close to the destination)
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7,
                      min_lr=0.0001, verbose=1),
]

history = model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,   # Use 20% of training data for validation
    epochs=60,              # Maximum training rounds
    batch_size=32,          # Process 32 samples at a time
    callbacks=callbacks,
    verbose=1
)

print("✅ Training complete!")


# ============================================================
# STEP 5: EVALUATE THE MODEL
# ============================================================
print("\n📈 STEP 5: Evaluating Model Performance...")

# Evaluate on test set (data model has never seen)
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\n  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print(f"  Test Loss:     {test_loss:.4f}")

# Get predictions
y_pred_proba = model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)   # Take class with highest probability
y_true = np.argmax(y_test, axis=1)         # Convert one-hot back to class index

print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))


# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================
print("\n📊 STEP 6: Creating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Emotion Recognition from Speech — Results Dashboard',
             fontsize=16, fontweight='bold')

# --- Chart 1: Training History (Accuracy) ---
ax1 = axes[0, 0]
ax1.plot(history.history['accuracy'],     label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy',   linewidth=2)
ax1.set_title('Training vs Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Chart 2: Training History (Loss) ---
ax2 = axes[0, 1]
ax2.plot(history.history['loss'],     label='Train Loss', linewidth=2, color='red')
ax2.plot(history.history['val_loss'], label='Val Loss',   linewidth=2, color='orange')
ax2.set_title('Training vs Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Chart 3: Emotion Distribution ---
ax3 = axes[0, 2]
emotion_counts = pd.Series(y_labels).value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_counts)))
emotion_counts.plot(kind='bar', ax=ax3, color=colors, edgecolor='black')
ax3.set_title('Emotion Distribution in Dataset')
ax3.set_xlabel('Emotion')
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)

# --- Chart 4: Confusion Matrix ---
ax4 = axes[1, 0]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax4.set_title('Confusion Matrix')
ax4.set_xlabel('Predicted Emotion')
ax4.set_ylabel('Actual Emotion')
ax4.tick_params(axis='x', rotation=45)
ax4.tick_params(axis='y', rotation=0)

# --- Chart 5: Per-Emotion Accuracy ---
ax5 = axes[1, 1]
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors2 = ['#2ecc71' if a >= 0.7 else '#e67e22' if a >= 0.5 else '#e74c3c'
           for a in per_class_acc]
bars = ax5.bar(le.classes_, per_class_acc * 100, color=colors2, edgecolor='black')
ax5.set_title('Accuracy per Emotion')
ax5.set_xlabel('Emotion')
ax5.set_ylabel('Accuracy (%)')
ax5.set_ylim(0, 110)
ax5.tick_params(axis='x', rotation=45)
ax5.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% threshold')
ax5.legend()
for bar, acc in zip(bars, per_class_acc):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{acc*100:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Chart 6: MFCC Feature Visualization ---
ax6 = axes[1, 2]
# Show average MFCC features per emotion (first 40 features = MFCCs)
emotion_feature_means = {}
for i, emotion in enumerate(le.classes_):
    mask = (y_true == i)
    if mask.sum() > 0:
        emotion_feature_means[emotion] = X_test[mask].mean(axis=0)[:40]

mfcc_df = pd.DataFrame(emotion_feature_means)
im = ax6.imshow(mfcc_df.T, aspect='auto', cmap='coolwarm')
ax6.set_title('Avg MFCC Features per Emotion')
ax6.set_xlabel('MFCC Coefficient Index')
ax6.set_ylabel('Emotion')
ax6.set_yticks(range(len(le.classes_)))
ax6.set_yticklabels(le.classes_)
plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.savefig('emotion_recognition_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Charts saved as 'emotion_recognition_results.png'")


# ============================================================
# STEP 7: PREDICT EMOTION FOR A NEW AUDIO FILE
# ============================================================
print("\n🔮 STEP 7: Predicting Emotion for New Audio...")

def predict_emotion(audio_path=None, audio_array=None, sample_rate=SAMPLE_RATE):
    """
    Predict emotion from either a file path or audio array.
    
    Usage:
        predict_emotion(audio_path="speech.wav")
        predict_emotion(audio_array=my_audio_array)
    """
    if audio_path and os.path.exists(audio_path):
        audio, sr = librosa.load(audio_path, sr=sample_rate)
    elif audio_array is not None:
        audio = audio_array
        sr = sample_rate
    else:
        print("❌ Provide audio_path or audio_array")
        return None

    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)

    probs = model.predict(features_cnn, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    predicted_emotion = le.classes_[predicted_idx]
    confidence = probs[predicted_idx]

    print(f"\n  Predicted Emotion: {predicted_emotion.upper()} ({confidence*100:.1f}% confidence)")
    print("  All probabilities:")
    for emotion, prob in sorted(zip(le.classes_, probs),
                                key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 20)
        print(f"    {emotion:10s}: {prob*100:5.1f}% {bar}")
    return predicted_emotion

# Demo: predict on a synthetic audio sample (simulates a "happy" pattern)
print("  Demo prediction on a synthetic audio sample...")
demo_audio = np.sin(2 * np.pi * 440 * np.linspace(0, DURATION, SAMPLE_RATE * DURATION))
demo_audio += np.random.normal(0, 0.05, len(demo_audio))  # Add slight noise
predict_emotion(audio_array=demo_audio)

print("\n  💡 For real audio files use:")
print('     predict_emotion(audio_path="your_speech.wav")')


# ============================================================
# SAVE THE MODEL
# ============================================================
model.save('emotion_model.keras')
print("\n✅ Model saved as 'emotion_model.keras'")
print("   Load later with: model = tf.keras.models.load_model('emotion_model.keras')")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  ✅ TASK 2 COMPLETE — EMOTION RECOGNITION FROM SPEECH")
print("=" * 60)
print(f"  Test Accuracy: {test_accuracy*100:.1f}%")
print(f"  Emotions Recognized: {list(le.classes_)}")
print(f"\n  Model Architecture:")
print("    Conv1D(64) → Conv1D(128) → Conv1D(256) → LSTM(128)")
print("    → Dense(256) → Dense(128) → Softmax Output")
print(f"\n  Key Learnings:")
print("  • MFCC converts audio → numerical features")
print("  • CNN detects local patterns in feature sequences")
print("  • LSTM understands temporal (time-based) patterns")
print("  • Use RAVDESS dataset for real-world performance")
print("=" * 60)
