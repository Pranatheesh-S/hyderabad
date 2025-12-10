import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import json

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical

# ========================
# GPU SETUP
# ========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU detected and configured\n")
else:
    print("⚠️ GPU not detected, using CPU\n")

# ========================
# CONFIG
# ========================
AUDIO_DIR = "audio"
META_FILE = os.path.join("meta", "esc50.csv")

SAMPLE_RATE = 16000
N_MELS = 128
MAX_FRAMES = 216
NUM_CLASSES = 50
EPOCHS = 40
BATCH_SIZE = 32

# ========================
# LOAD METADATA
# ========================
df = pd.read_csv(META_FILE)

categories = sorted(df["category"].unique())
class_to_index = {cls: i for i, cls in enumerate(categories)}
index_to_class = {i: cls for cls, i in class_to_index.items()}

print(f"✅ Loaded {len(categories)} classes\n")

# ========================
# FEATURE EXTRACTION
# ========================
def extract_mel(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - np.mean(mel)) / np.std(mel)

    if mel.shape[1] < MAX_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, MAX_FRAMES - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :MAX_FRAMES]

    return mel

# ========================
# BUILD DATASET (NO AUG HERE)
# ========================
X, y = [], []

print("🎵 Extracting features...\n")

for _, row in df.iterrows():
    path = os.path.join(AUDIO_DIR, row["filename"])
    label = class_to_index[row["category"]]

    mel = extract_mel(path)

    X.append(mel)
    y.append(label)

X = np.array(X)[..., np.newaxis]
y = to_categorical(y, NUM_CLASSES)

print(f"✅ Dataset ready: {X.shape}")

# ========================
# SPLIT DATA
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
)

# ========================
# TF DATA (GPU PIPELINE)
# ========================
def augment_spec(spec):
    if tf.random.uniform(()) < 0.3:
        spec = tf.image.random_brightness(spec, 0.1)
    if tf.random.uniform(()) < 0.3:
        spec = tf.image.random_contrast(spec, 0.9, 1.1)
    return spec

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(
    lambda x, y: (augment_spec(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ========================
# STRONG GPU CNN MODEL
# ========================
model = models.Sequential([
    layers.Input(shape=(N_MELS, MAX_FRAMES, 1)),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================
# CALLBACKS
# ========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
]

# ========================
# TRAIN
# ========================
print("\n🚀 Training started...\n")

model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)

# ========================
# EVALUATE
# ========================
loss, acc = model.evaluate(test_ds)
print(f"\n🎯 FINAL TEST ACCURACY: {acc*100:.2f}%")

# ========================
# SAVE
# ========================
model.save("esc50_gpu_model.keras")

with open("class_mapping.json", "w") as f:
    json.dump(index_to_class, f, indent=4)

print("\n✅ Model saved: esc50_gpu_model.keras")
print("✅ Class map saved: class_mapping.json")
