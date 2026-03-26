import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import cv2
import torch
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from transformers import HubertModel, Wav2Vec2Processor

# Load the Base Models first to construct the Fusion Pipeline Memory Graph
base_face = load_model(r"d:\S6 Mini Project\Facial Detection\emotion_model.keras")
for layer in base_face.layers: layer.trainable = False
base_voice = load_model(r"d:\S6 Mini Project\HuBERT Model\voice_emotion_detection_hubert_large.keras")
for layer in base_voice.layers: layer.trainable = False

from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Dropout, Lambda

face_extractor = Model(inputs=base_face.inputs, outputs=base_face.layers[-2].output)
voice_extractor = Model(inputs=base_voice.inputs, outputs=base_voice.layers[-2].output)

input_face = Input(shape=(48, 48, 1))
input_voice = Input(shape=(1024,))

feat_face = face_extractor(input_face)
feat_voice = voice_extractor(input_voice)
proj_face = Dense(256, activation='relu')(feat_face)
proj_voice = Dense(256, activation='relu')(feat_voice)

joint = Concatenate()([proj_face, proj_voice])
att_weights = Dense(128, activation='relu')(joint)
att_weights = Dense(2, activation='softmax')(att_weights)

# Keras functional graph manipulation
weight_face = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(att_weights)
weight_voice = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(att_weights)

attended_face = Multiply()([proj_face, weight_face])
attended_voice = Multiply()([proj_voice, weight_voice])

fused = Concatenate()([attended_face, attended_voice])
x = Dropout(0.4)(fused)
x = Dense(128, activation='relu')(x)
final_output = Dense(7, activation='softmax')(x) 

# Recreate the Fused Model exactly and Load ONLY the Weights!
fused_model = Model(inputs=[input_face, input_voice], outputs=final_output)
fused_model.load_weights(r"d:\S6 Mini Project\attention_fused_emotion_model.keras")

# Load Audio Encoders/Scalers
voice_scaler = joblib.load(r"d:\S6 Mini Project\HuBERT Model\scaler_hubert_large.pkl")

# We hardcode the 7 intersecting emotions we used during fusion training
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load HuBERT Large
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/hubert-large-ls960-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
hubert_model = HubertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
hubert_model.to(device)
hubert_model.eval()

# Face Cascade for cropping face from images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================================================
# 2. FEATURE EXTRACTION FUNCTIONS
# ==========================================================
def extract_face_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # If no face detected, just resize the whole image as a pure fallback
        face_img = gray
    else:
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    
    # Needs to be shape (1, 48, 48, 1) to pass into Keras Face Extractor Sub-Model
    return np.reshape(face_img, (1, 48, 48, 1))

def extract_audio_features(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    speech = librosa.util.normalize(speech)

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hubert_model(**inputs)

    hidden_states = outputs.hidden_states
    stacked = torch.stack(hidden_states[-4:])
    layer_mean = stacked.mean(dim=0)
    pooled = layer_mean.mean(dim=1)
    
    hubert_features = pooled.squeeze().cpu().numpy()
    
    # Scale and expand to shape (1, 1024) to pass into Keras Voice Extractor Sub-Model
    scaled_features = voice_scaler.transform([hubert_features])
    return scaled_features

# ==========================================================
# 3. PREDICT MULTIMODAL EMOTION
# ==========================================================
def predict_emotion(image_path, audio_path):
    try:
        print("\nExtracting Face Features...")
        face_input = extract_face_features(image_path)
        
        print("Extracting Audio Features (HuBERT)...")
        voice_input = extract_audio_features(audio_path)
        
        print("\nRunning Attention-Based Multimodal Prediction...")
        prediction = fused_model.predict([face_input, voice_input], verbose=0)[0]
        
        emotion_index = np.argmax(prediction)
        predicted_emotion = EMOTIONS[emotion_index]
        confidence = prediction[emotion_index] * 100
        
        print("="*40)
        print(f" FINAL FUSED EMOTION: {predicted_emotion.upper()} ")
        print(f" Fusion Confidence    : {confidence:.2f}%")
        print("="*40)
        
        # Optionally show all probability balances
        print("\nDetailed Probabilities:")
        for idx, emo in enumerate(EMOTIONS):
            print(f" - {emo.capitalize():<8}: {prediction[idx]*100:>5.2f}%")
            
    except Exception as e:
        print(f"[ERROR] Multimodal Prediction Failed: {e}")

# ==========================================================
# 4. RUN TEST
# ==========================================================
if __name__ == "__main__":
    # Replace these paths with paths to an actual testing Face Image and Audio WAV file!
    TEST_IMAGE = r"d:\S6 Mini Project\Datasets\train\happy\Training_10019449.jpg" # Example
    TEST_AUDIO = r"d:\S6 Mini Project\Datasets\audio_speech_actors_01-24\Actor_01\03-01-03-01-01-01-01.wav" # Example Happy Audio
    
    print(f"Testing Face : {TEST_IMAGE}")
    print(f"Testing Audio: {TEST_AUDIO}")
    
    if os.path.exists(TEST_IMAGE) and os.path.exists(TEST_AUDIO):
        predict_emotion(TEST_IMAGE, TEST_AUDIO)
    else:
        print("Please point TEST_IMAGE and TEST_AUDIO to valid files in your directory to run the final test.")
