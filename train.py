# =============================
# FULL ESC-50 TRAINING SCRIPT
# =============================

import os
import numpy as np
import pandas as pd
import librosa

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import json

# ===== CONFIG =====
BASE_DIR = os.path.join(os.getcwd(), "ESC-50-master")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
META_FILE = os.path.join(BASE_DIR, "meta", "esc50.csv")

SAMPLE_RATE = 22050
N_MELS = 128
MAX_FRAMES = 216   # width of mel-spectrogram
EPOCHS = 20
BATCH_SIZE = 16


# ============================
# FEATURE EXTRACTION FUNCTION
# ============================
def extract_mel_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize 0 → 1
    mel -= mel.min()
    mel /= mel.max() if mel.max() > 0 else 1

    # Pad / trim to MAX_FRAMES
    if mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel = mel[:, :MAX_FRAMES]

    return mel


# ====================
# LOAD ESC-50 DATASET
# ====================
print("\n📂 Loading ESC-50 metadata...\n")

df = pd.read_csv(META_FILE)

category_names = sorted(df["category"].unique())
NUM_CLASSES = len(category_names)

print(f"✅ Detected {NUM_CLASSES} classes.\n")

class_to_index = {cls: i for i, cls in enumerate(category_names)}
index_to_class = {i: cls for cls, i in class_to_index.items()}


# =====================
# BUILD DATASET
# =====================
X, y = [], []

print("🎵 Extracting spectrograms (this will take a few minutes)...\n")

for i, row in df.iterrows():
    file = row["filename"]
    label = row["category"]

    path = os.path.join(AUDIO_DIR, file)

    mel = extract_mel_spectrogram(path)

    X.append(mel)
    y.append(class_to_index[label])

X = np.array(X)[..., np.newaxis]   # (N, 128, 216, 1)
y = to_categorical(y, NUM_CLASSES)

print(f"✅ Dataset prepared: {X.shape[0]} samples\n")

# ===========================
# TRAIN / TEST SPLIT
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)


# ============================
# CNN MODEL
# ============================
print("🧠 Building CNN model...\n")

model = models.Sequential([
    layers.Input(shape=(N_MELS, MAX_FRAMES, 1)),

    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation='softmax')
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ====================
# TRAIN
# ====================
print("🚀 Training started...\n")

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# ====================
# EVALUATE + SAVE
# ====================
loss, acc = model.evaluate(X_test, y_test)

print(f"\n🎯 TEST ACCURACY: {acc*100:.2f}%")

model.save("forest_sound_model.h5")

with open("class_mapping.json", "w") as f:
    json.dump(index_to_class, f, indent=4)

print("\n✅ MODEL SAVED AS: forest_sound_model.h5")
print("✅ CLASS MAP SAVED AS: class_mapping.json")
