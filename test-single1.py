#!/usr/bin/env python3
# quick_predict_esc50.py — edit ONLY AUDIO_FILE, then run: python quick_predict_esc50.py

import os, json, numpy as np, librosa
from tensorflow.keras.models import load_model

# ======= EDIT ONLY THIS =======
AUDIO_FILE = os.path.join("audio", "1-137-A-32.wav")   # <-- change to your audio file in the project folder
MODEL_FILE = "esc50_gpu_model.keras"                     # <-- change if your ESC-50 model file has another name
CLASS_MAP_FILE = "class_mapping.json"
TOP_K = 5
# ==============================

# ESC-50 preprocessing (matches training)
SAMPLE_RATE = 16000
N_MELS = 128
MAX_FRAMES = 216

def extract_esc50_standard(filepath, sr=SAMPLE_RATE, n_mels=N_MELS, max_frames=MAX_FRAMES):
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-9)
    if mel_db.shape[1] < max_frames:
        mel_db = np.pad(mel_db, ((0,0),(0, max_frames - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :max_frames]
    return mel_db.astype(np.float32)

def load_class_map(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        try:
            return [data[str(i)] for i in range(len(data))]
        except Exception:
            items = sorted(data.items(), key=lambda kv: (int(kv[0]) if str(kv[0]).isdigit() else kv[0]))
            return [v for k, v in items]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported class map format")

def main():
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    if not os.path.exists(CLASS_MAP_FILE):
        raise FileNotFoundError(f"Class map not found: {CLASS_MAP_FILE}")

    spec = extract_esc50_standard(AUDIO_FILE)
    inp = spec[np.newaxis, ..., np.newaxis]

    print(f"Loading model: {MODEL_FILE}")
    model = load_model(MODEL_FILE, compile=False)
    class_map = load_class_map(CLASS_MAP_FILE)

    probs = model.predict(inp, verbose=0)[0]
    top_idx = probs.argsort()[-TOP_K:][::-1]

    print(f"\nFile: {AUDIO_FILE}\n")
    for i in top_idx:
        label = class_map[i] if i < len(class_map) else str(i)
        print(f"  {i:02d}  {label:30s}  {probs[i]:.6f}")

    best = int(np.argmax(probs))
    best_label = class_map[best] if best < len(class_map) else str(best)
    print(f"\n=> Top-1: [{best}] {best_label} ({probs[best]*100:.2f}%)\n")

if __name__ == "__main__":
    main()
