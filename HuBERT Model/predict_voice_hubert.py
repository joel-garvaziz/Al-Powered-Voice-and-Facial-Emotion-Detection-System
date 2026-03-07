# ============================================================
# ENV
# ============================================================

import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore")

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import librosa
import sounddevice as sd
import torch
import joblib

from keras.models import load_model
from transformers import HubertModel, Wav2Vec2Processor

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH   = "voice_emotion_detection_hubert_large.keras"
SCALER_PATH  = "scaler_hubert_large.pkl"
ENCODER_PATH = "encoder_hubert_large.pkl"

SAMPLE_RATE = 16000
DURATION = 20

# ============================================================
# LOAD HUBERT
# ============================================================

MODEL_NAME = "facebook/hubert-large-ls960-ft"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

hubert_model = HubertModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
)

hubert_model.to(device)
hubert_model.eval()

# ============================================================
# LOAD MODEL
# ============================================================

def load_all():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder

# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio():

    print("\n🎤 Speak now...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    print("✅ Recording finished")

    print("🔊 Playing back recorded audio...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()

    return audio.flatten()
# ============================================================
# HUBERT FEATURE EXTRACTION
# ============================================================

def extract_hubert(audio):

    audio = librosa.util.normalize(audio)

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hubert_model(**inputs)

    hidden_states = outputs.hidden_states

    stacked = torch.stack(hidden_states[-4:])
    layer_mean = stacked.mean(dim=0)

    pooled = layer_mean.mean(dim=1)

    return pooled.squeeze().cpu().numpy()

# ============================================================
# PREDICT
# ============================================================

def predict_emotion(model, scaler, encoder):

    audio = record_audio()

    features = extract_hubert(audio)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    probs = model.predict(features)[0]

    idx = np.argmax(probs)

    emotion = encoder.categories_[0][idx]

    confidence = probs[idx] * 100

    print("\n==============================")
    print("🎭 Emotion   :", emotion)
    print("🔥 Confidence:", f"{confidence:.2f}%")
    print("==============================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model, scaler, encoder = load_all()

    predict_emotion(model, scaler, encoder)