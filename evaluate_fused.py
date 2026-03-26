import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Dropout, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import joblib 
import cv2 

print("===============================================")
print("  EVALUATING MULTIMODAL FUSION METRICS         ")
print("===============================================")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
voice_scaler = joblib.load(r"d:\S6 Mini Project\HuBERT Model\scaler_hubert_large.pkl")

print("Locating Weights and rebuilding Fusion Topographies natively...")

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

class MultimodalTestGenerator(tf.keras.utils.Sequence):
    # Same as training generator, except minimal
    def __init__(self, face_dir, voice_npy_x, voice_npy_y, batch_size=32):
        self.face_dir = face_dir
        self.batch_size = batch_size
        self.voice_X = np.load(voice_npy_x)
        self.raw_voice_Y = np.load(voice_npy_y, allow_pickle=True)
        self.voice_Y_str = self.raw_voice_Y
        
        valid_indices = []
        for i, val in enumerate(self.voice_Y_str):
            if val == 'calm':
                val = 'neutral'
                self.voice_Y_str[i] = 'neutral'
            if val in EMOTIONS: valid_indices.append(i)
                
        self.voice_X = self.voice_X[valid_indices]
        self.voice_Y_str = self.voice_Y_str[valid_indices]
                
        self.face_paths = {emotion: [] for emotion in EMOTIONS}
        for emotion in EMOTIONS:
            folder_path = os.path.join(self.face_dir, emotion)
            if not os.path.exists(folder_path): folder_path = os.path.join(self.face_dir, emotion.capitalize())
            if os.path.exists(folder_path):
                self.face_paths[emotion] = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png','jpg'))]
                
    def __len__(self):
        return int(np.floor(len(self.voice_X) / self.batch_size))

    def __getitem__(self, index):
        batch_voice_x, batch_face_x, batch_y = [], [], []
        
        current_voice_x = self.voice_X[index*self.batch_size : (index+1)*self.batch_size]
        current_voice_y = self.voice_Y_str[index*self.batch_size : (index+1)*self.batch_size]
        
        for i in range(len(current_voice_x)):
            v_feature = current_voice_x[i]
            emotion_label = current_voice_y[i]
            if emotion_label in self.face_paths and len(self.face_paths[emotion_label]) > 0:
                random_face_path = np.random.choice(self.face_paths[emotion_label])
                img = cv2.imread(random_face_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue 
                img = cv2.resize(img, (48, 48)).astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)
                
                batch_voice_x.append(v_feature)
                batch_face_x.append(img)
                batch_y.append(EMOTIONS.index(emotion_label))
                
        if len(batch_voice_x) == 0:
            return (np.zeros((1, 48, 48, 1)), np.zeros((1, 1024))), to_categorical([0], num_classes=len(EMOTIONS))

        batch_voice_x = voice_scaler.transform(np.array(batch_voice_x))
        batch_face_x = np.array(batch_face_x)
        batch_y_cat = to_categorical(batch_y, num_classes=len(EMOTIONS))
        return (batch_face_x, batch_voice_x), batch_y_cat

# Run Evaluation
FACE_TEST_DIR = r"d:\S6 Mini Project\Datasets\test" 
VOICE_X_PATH = r"d:\S6 Mini Project\HuBERT Model\X_hubert_large.npy"
VOICE_Y_PATH = r"d:\S6 Mini Project\HuBERT Model\Y_hubert_large.npy"

if not os.path.exists(FACE_TEST_DIR): 
    print("Falling back to Train dir as Test dir not found.")
    FACE_TEST_DIR = r"d:\S6 Mini Project\Datasets\train"

print("\nInitializing Test Data Generator...")
test_gen = MultimodalTestGenerator(FACE_TEST_DIR, VOICE_X_PATH, VOICE_Y_PATH, batch_size=32)

print("Running Predictions over Dataset (This will take a few moments)...")
y_true = []
y_pred = []

# Evaluate enough batches to be statistically accurate (e.g. 1000 total samples)
total_batches = min(35, len(test_gen))

for i in range(total_batches):
    X, Y = test_gen[i]
    if len(X[0]) <= 1: continue 
    
    preds = fused_model.predict(X, verbose=0)
    
    y_true.extend(np.argmax(Y, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    
    print(f"\rProcessed batch {i+1}/{total_batches}", end="", flush=True)

print("\n\n===============================================")
print("  MULTIMODAL FUSION PERFORMANCE REPORT         ")
print("===============================================")
acc = accuracy_score(y_true, y_pred)
print(f"Overall Fused Accuracy: {acc * 100:.2f}%\n")

report = classification_report(y_true, y_pred, target_names=EMOTIONS)
print(report)
print("===============================================")
