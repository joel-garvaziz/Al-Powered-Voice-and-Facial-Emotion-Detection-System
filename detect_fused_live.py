import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time
import numpy as np
import cv2
import threading
import torch
import sounddevice as sd
import joblib

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Dropout, Lambda
import tensorflow as tf
from transformers import HubertModel, Wav2Vec2Processor

# ==========================================================
# 1. LOAD THE FUSED ARCHITECTURE AND SCALERS
# ==========================================================
print("Loading Models. Please wait for the application to start...")

base_face = load_model(r"d:\S6 Mini Project\Facial Detection\emotion_model.keras")
for layer in base_face.layers: layer.trainable = False
base_voice = load_model(r"d:\S6 Mini Project\HuBERT Model\voice_emotion_detection_hubert_large.keras")
for layer in base_voice.layers: layer.trainable = False

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
weight_face = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(att_weights)
weight_voice = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(att_weights)
attended_face = Multiply()([proj_face, weight_face])
attended_voice = Multiply()([proj_voice, weight_voice])
fused = Concatenate()([attended_face, attended_voice])
x = Dropout(0.4)(fused)
x = Dense(128, activation='relu')(x)
final_output = Dense(7, activation='softmax')(x) 

fused_model = Model(inputs=[input_face, input_voice], outputs=final_output)
fused_model.load_weights(r"d:\S6 Mini Project\attention_fused_emotion_model.keras")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
voice_scaler = joblib.load(r"d:\S6 Mini Project\HuBERT Model\scaler_hubert_large.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/hubert-large-ls960-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
hubert_model = HubertModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
hubert_model.eval()

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ==========================================================
# 2. NOISE-ROBUST AUDIO HELPERS
# ==========================================================

def estimate_noise_floor(audio_clip, noise_percentile=15):
    """Estimate background noise floor from the lower percentile of frame energies."""
    frame_size = 512
    hop_size = 256
    frames = []
    for start in range(0, len(audio_clip) - frame_size, hop_size):
        frame = audio_clip[start:start + frame_size]
        frames.append(float(np.sqrt(np.mean(frame ** 2))))
    if not frames:
        return 0.0
    return float(np.percentile(frames, noise_percentile))


def compute_snr(audio_clip, noise_floor):
    """Estimate SNR in dB given a noise floor measurement."""
    signal_rms = float(np.sqrt(np.mean(audio_clip ** 2)))
    if noise_floor < 1e-9:
        return 60.0  # Essentially clean signal
    return float(20 * np.log10(max(signal_rms, 1e-9) / max(noise_floor, 1e-9)))


# ==========================================================
# 3. APPLICATION GLOBAL STATE & THREADING ARRAYS
# ==========================================================
SAMPLE_RATE = 16000
DURATION = 3  
audio_buffer = np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)

latest_face_crop = None
latest_emotion = "Waiting for data..."
running = True

# Timing & Diagnostic Variables
timeline_data = [] # Stores (timestamp_seconds, raw_prediction_probabilities_array)
start_time = None

# ==========================================================
# 3. BACKGROUND WORKERS
# ==========================================================
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    shift = len(indata)
    audio_buffer = np.roll(audio_buffer, -shift)
    audio_buffer[-shift:] = indata[:, 0]

def extraction_worker():
    global latest_emotion, latest_face_crop, audio_buffer, running, timeline_data, start_time

    # Wait until start_time is officially set by the Main Loop
    while start_time is None and running: time.sleep(0.1)

    while running:
        try:
            audio_clip = audio_buffer.copy()
            max_amp = float(np.max(np.abs(audio_clip)))

            # 1. Dynamic Noise Floor & SNR Calculation
            # Robustly identifies constant background fan/ambient noise vs real speech
            noise_floor = estimate_noise_floor(audio_clip, noise_percentile=10)
            snr_db = compute_snr(audio_clip, noise_floor)
            
            # 2. Determine if it's genuine speech. (Tightened for noise robustness)
            # Speech has bursts of energy, so SNR > 8.5 dB. Constant background noise should stay below this.
            is_speech = (snr_db >= 8.5) and (max_amp >= 0.03)

            # 3. Handle background noise gracefully WITHOUT normalizing it!
            if is_speech:
                # Normalize only clear speech
                audio_clip = audio_clip / max_amp

            # 4. Extract Face Input
            is_masked = False
            if latest_face_crop is not None:
                face_img = latest_face_crop.copy()
            else:
                face_img = np.zeros((48, 48, 1), dtype='float32')
                is_masked = True

            face_input = np.reshape(face_img, (1, 48, 48, 1))

            # 5. HuBERT audio features
            inputs = processor(audio_clip, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = hubert_model(**inputs)

            hidden_states = outputs.hidden_states
            layer_mean = torch.stack(hidden_states[-4:]).mean(dim=0)
            pooled = layer_mean.mean(dim=1).squeeze().cpu().numpy()
            voice_input = voice_scaler.transform([pooled])

            # 6. Run all model predictions
            fused_pred = fused_model.predict([face_input, voice_input], verbose=0)[0]
            fused_idx = np.argmax(fused_pred)
            fused_conf = fused_pred[fused_idx] * 100

            bf_emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            base_face_pred = base_face.predict(face_input, verbose=0)[0] if not is_masked else np.zeros((7,))
            bf_idx = np.argmax(base_face_pred)

            base_voice_pred = base_voice.predict(voice_input, verbose=0)[0]
            bv_idx = np.argmax(base_voice_pred)

            # Remap face model label order -> fused model label order
            mapped_face_pred = np.zeros(7, dtype=float)
            mapped_face_pred[0] = base_face_pred[0]  # Angry
            mapped_face_pred[1] = base_face_pred[1]  # Disgust
            mapped_face_pred[2] = base_face_pred[2]  # Fear
            mapped_face_pred[3] = base_face_pred[3]  # Happy
            mapped_face_pred[4] = base_face_pred[6]  # Neutral
            mapped_face_pred[5] = base_face_pred[4]  # Sad
            mapped_face_pred[6] = base_face_pred[5]  # Surprise

            # Remap voice model (8 classes) to 7
            mapped_voice_pred = np.zeros(7, dtype=float)
            if len(base_voice_pred) == 8:
                mapped_voice_pred[0] = base_voice_pred[0]
                mapped_voice_pred[1] = base_voice_pred[2]
                mapped_voice_pred[2] = base_voice_pred[3]
                mapped_voice_pred[3] = base_voice_pred[4]
                mapped_voice_pred[4] = base_voice_pred[5] + base_voice_pred[1]
                mapped_voice_pred[5] = base_voice_pred[6]
                mapped_voice_pred[6] = base_voice_pred[7]
            else:
                mapped_voice_pred = base_voice_pred[:7]

            # 7. Clean binary routing
            if is_speech:
                if is_masked:
                    # Voice only -> use voice model
                    latest_emotion = f"VOICE ONLY: {EMOTIONS[np.argmax(mapped_voice_pred)].capitalize()} ({mapped_voice_pred[np.argmax(mapped_voice_pred)]*100:.1f}%)"
                    raw_pred_array = mapped_voice_pred
                else:
                    # Speech & Face -> logic check: if SNR is borderline and Face is strong, fallback to Face
                    fused_conf = float(np.max(fused_pred))
                    face_conf = float(np.max(mapped_face_pred))
                    
                    if fused_conf < 0.35 and face_conf > 0.60:
                        # Fusion is unsure (likely noise interference), Face is very sure. Trust Face.
                        latest_emotion = f"FUSED (VISUAL ASSET): {EMOTIONS[np.argmax(mapped_face_pred)].capitalize()} ({face_conf*100:.1f}%)"
                        raw_pred_array = mapped_face_pred
                    else:
                        # Speech & Face -> use Fused Model
                        latest_emotion = f"FUSED EMOTION: {EMOTIONS[fused_idx].capitalize()} ({fused_conf*100:.1f}%)"
                        raw_pred_array = fused_pred
            else:
                # NO SPEECH (Either pure silence or background noise)
                if is_masked:
                    # No face, no speech -> wait
                    latest_emotion = "AWAITING USER..."
                    raw_pred_array = np.zeros((7,))
                else:
                    # Face present, no speech -> Face Model ONLY
                    face_idx = np.argmax(mapped_face_pred)
                    face_conf = mapped_face_pred[face_idx] * 100
                    latest_emotion = f"FACE ONLY: {EMOTIONS[face_idx].capitalize()} ({face_conf:.1f}%)"
                    raw_pred_array = mapped_face_pred

            # Log for Timeline
            elapsed = time.time() - start_time
            if elapsed > 0:
                timeline_data.append((elapsed, raw_pred_array))

            print(f"[Debug] Face:{EMOTIONS[np.argmax(mapped_face_pred)]} | Voice:{EMOTIONS[bv_idx]} | SNR:{snr_db:.1f}dB | Speech:{is_speech} -> {latest_emotion}")

        except Exception as e:
            print(f"[Thread Error] Inference failed: {e}")

        time.sleep(0.5)

# ==========================================================
# 4. MAIN WEBCAM UI LOOP
# ==========================================================
if __name__ == "__main__":
    print("\n=======================================================")
    print(" 10-SECOND LIVE MULTIMODAL EVALUATION INITIATED ")
    print("=======================================================\n")
    
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    stream.start()
    
    ai_thread = threading.Thread(target=extraction_worker)
    ai_thread.daemon = True
    ai_thread.start()
    
    cap = cv2.VideoCapture(0)
    
    # Officially start the 10-second timer
    start_time = time.time()
    
    while cap.isOpened():
        elapsed_time = time.time() - start_time
        
        # 10-Second Window Break Condition
        if elapsed_time > 10.0:
            print("\nEvaluation Complete! Stopping sensors...")
            break
            
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        vol_norm = np.linalg.norm(audio_buffer[-3200:]) * 5 
        vol_bars = "|" * int(min(20, vol_norm))
        cv2.putText(frame, f"LIVE MIC: [{vol_bars:<20}]", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            face_crop = gray[y:y+h, x:x+w]
            latest_face_crop = cv2.resize(face_crop, (48, 48)).astype('float32') / 255.0
        else:
            latest_face_crop = None 
            
        cv2.putText(frame, latest_emotion, 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 3, cv2.LINE_AA)
                    
        # Add Timer formatting to bottom right
        time_left = max(0, 10.0 - elapsed_time)
        cv2.putText(frame, f"Timer: {time_left:.1f}s", (450, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("Multimodal 10-Second Diagnostic Evaluation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Graceful Shutdown
    running = False
    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
    
    # ==========================================================
    # 5. DIAGNOSTIC TIMELINE GENERATION
    # ==========================================================
    if len(timeline_data) > 0:
        # Calculate Dominant Emotion mathematically over the window
        all_probs = np.array([item[1] for item in timeline_data])
        mean_probs = np.mean(all_probs, axis=0)
        max_idx = np.argmax(mean_probs)
        dom_emo = EMOTIONS[max_idx].capitalize()
        dom_conf = mean_probs[max_idx] * 100
        
        print(f"\n=======================================================")
        print(f" FINAL DIAGNOSTIC RESULT (10-SEC WINDOW) ")
        print(f"=======================================================")
        print(f" >> EMOTION WITH MAXIMUM PROBABILITY: {dom_emo} ({dom_conf:.2f}%) <<")
        print(f"=======================================================\n")
        
        # Plot Matplotlib Timeline Graph
        try:
            import matplotlib.pyplot as plt
            times = [item[0] for item in timeline_data]
            
            plt.figure(figsize=(10, 5))
            for i, emo in enumerate(EMOTIONS):
                emo_probs = all_probs[:, i] * 100
                # Only plot lines that actively spiked or were considered to keep graph clean
                if np.max(emo_probs) > 5.0:
                    plt.plot(times, emo_probs, label=emo.capitalize(), linewidth=2, marker='o', alpha=0.8)
                    
            plt.fill_between(times, 0, 100, color='gray', alpha=0.1)
            plt.title(f"Live Evaluation Timeline (Dominant Emotion: {dom_emo})", fontsize=14, fontweight='bold')
            plt.xlabel("Elapsed Time (Seconds)", fontsize=11)
            plt.ylabel("Prediction Confidence (%)", fontsize=11)
            plt.ylim([0, 105])
            plt.xlim([0, 10.5])
            
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
            plt.tight_layout()
            
            # Save strictly to local directory
            graph_path = os.path.join(os.getcwd(), "timeline_graph.png")
            plt.savefig(graph_path, dpi=300)
            print(f"-> Beautiful Timeline Graph saved to: {graph_path}")
            
            # Show interactive window to user!
            plt.show()
            
        except ImportError:
            print("[Warning] Matplotlib is not installed. Run 'pip install matplotlib' to see the timeline graph.")
    else:
        print("Error: No timeline data collected. Did the audio buffer fail?")
