import os
import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model

SAMPLE_RATE = 22050
N_MELS = 128
MAX_FRAMES = 216

def extract_mel_spectrogram(filepath, sr=SAMPLE_RATE, n_mels=N_MELS, max_frames=MAX_FRAMES):
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db -= mel_db.min()
    if mel_db.max() > 0:
        mel_db /= mel_db.max()
    if mel_db.shape[1] < max_frames:
        pad_width = max_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_frames]
    return mel_db

# Load model and class map
model = load_model("forest_sound_model.h5")
with open("class_mapping.json", "r") as f:
    index_to_class = json.load(f)

# === CHANGE THIS to any esc-50 audio you want to test ===
# BASE_DIR = os.path.join(os.getcwd(), "ESC-50-master")
AUDIO_FILE = os.path.join("audio", "1-103995-A-30.wav")

mel = extract_mel_spectrogram(AUDIO_FILE)
mel = mel[np.newaxis, ..., np.newaxis]  # shape (1, 128, 216, 1)

pred = model.predict(mel)
pred_idx = int(np.argmax(pred))
pred_class = index_to_class[str(pred_idx)]

print("Predicted class:", pred_class)
print("Raw probabilities:", pred)
