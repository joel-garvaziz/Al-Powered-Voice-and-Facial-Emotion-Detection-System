import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Concatenate, Multiply, Dropout
from tensorflow.keras.utils import to_categorical
import joblib 
import os
import cv2 

print("===============================================")
print("  ATTENTION-BASED MULTIMODAL FUSION TRAINING   ")
print("===============================================")

# ==========================================================
# 1. LOAD YOUR PRETRAINED MODELS & PREPROCESSORS
# ==========================================================
print("Loading Base Models...")

base_face = load_model(r"d:\S6 Mini Project\Facial Detection\emotion_model.keras")
base_voice = load_model(r"d:\S6 Mini Project\HuBERT Model\voice_emotion_detection_hubert_large.keras")

# Load Audio Encoders/Scalers (using joblib as seen in your HuBERT script)
voice_scaler = joblib.load(r"d:\S6 Mini Project\HuBERT Model\scaler_hubert_large.pkl")
voice_encoder = joblib.load(r"d:\S6 Mini Project\HuBERT Model\encoder_hubert_large.pkl")

# ==========================================================
# 2. BUILD THE ATTENTION FUSION ARCHITECTURE
# ==========================================================
print("Building Attention Fusion Head...")

# To avoid Keras layer name collisions when fusing flat graphs, we build cleanly isolated Sub-Models
face_extractor = Model(inputs=base_face.inputs, outputs=base_face.layers[-2].output, name="isolated_face_extractor")
face_extractor.trainable = False

voice_extractor = Model(inputs=base_voice.inputs, outputs=base_voice.layers[-2].output, name="isolated_voice_extractor")
voice_extractor.trainable = False

# New entry points for the overarching Fusion Model
input_face = tf.keras.layers.Input(shape=(48, 48, 1), name="face_input")
input_voice = tf.keras.layers.Input(shape=(1024,), name="voice_input")

# Get deep representations through isolated graphs
feat_face = face_extractor(input_face)
feat_voice = voice_extractor(input_voice)

# Project to equal dimensions (256)
proj_face = Dense(256, activation='relu')(feat_face)
proj_voice = Dense(256, activation='relu')(feat_voice)

# Attention Calculation (Dynamic weights for modality)
joint = Concatenate()([proj_face, proj_voice])
att_weights = Dense(128, activation='relu')(joint)
from tensorflow.keras.layers import Dense, Concatenate, Multiply, Dropout, Lambda

# ... (down to where splits happen) ...
att_weights = Dense(2, activation='softmax')(att_weights)

# Safely extract and expand using Keras Lambda layers 
weight_face = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(att_weights)
weight_voice = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(att_weights)

# Apply Attention Multipliers
attended_face = Multiply()([proj_face, weight_face])
attended_voice = Multiply()([proj_voice, weight_voice])

# Final Classification Head
fused = Concatenate()([attended_face, attended_voice])
x = Dropout(0.4)(fused)
x = Dense(128, activation='relu')(x)

# 7 Output classes (matching Face classes + Voice overlapping classes)
final_output = Dense(7, activation='softmax', name="emotion_fusion_output")(x) 

attention_model = Model(inputs=[input_face, input_voice], outputs=final_output, name="AttentionFusionModel")
attention_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================================
# 3. CREATE A MULTIMODAL DATA GENERATOR (SIMULATED PAIRS)
# ==========================================================
class MultimodalGenerator(tf.keras.utils.Sequence):
    def __init__(self, face_dir, voice_npy_x, voice_npy_y, batch_size=32):
        self.face_dir = face_dir
        self.batch_size = batch_size
        
        # Load Pre-extracted HuBERT Audio Features
        self.voice_X = np.load(voice_npy_x)
        self.raw_voice_Y = np.load(voice_npy_y, allow_pickle=True)
        
        # The numpy array is already a flat array of string labels
        self.voice_Y_str = self.raw_voice_Y
        
        # The intersecting 7 emotions (Voice has 'calm', Face does not. We map calm->neutral or ignore it)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Gather all voice indices that map to these 7 emotions (exclude purely 'calm' if not mapping)
        valid_indices = []
        for i, val in enumerate(self.voice_Y_str):
            if val == 'calm':
                val = 'neutral' # Map calm to neutral to keep data volume high
                self.voice_Y_str[i] = 'neutral'
            if val in self.emotions:
                valid_indices.append(i)
                
        # Filter down Voice dataset
        self.voice_X = self.voice_X[valid_indices]
        self.voice_Y_str = self.voice_Y_str[valid_indices]
                
        # Load Face Image Paths into a Dictionary
        self.face_paths = {emotion: [] for emotion in self.emotions}
        for emotion in self.emotions:
            folder_path = os.path.join(self.face_dir, emotion)
            
            # Accommodate uppercase folders if they exist
            if not os.path.exists(folder_path):
                folder_path = os.path.join(self.face_dir, emotion.capitalize())
                
            if os.path.exists(folder_path):
                self.face_paths[emotion] = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png','jpg'))]
            else:
                print(f"Warning: Could not find face folder for emotion '{emotion}' at {folder_path}")
                
    def __len__(self):
        return int(np.floor(len(self.voice_X) / self.batch_size))

    def __getitem__(self, index):
        batch_voice_x = []
        batch_face_x = []
        batch_y = []
        
        current_voice_x = self.voice_X[index*self.batch_size : (index+1)*self.batch_size]
        current_voice_y = self.voice_Y_str[index*self.batch_size : (index+1)*self.batch_size]
        
        for i in range(len(current_voice_x)):
            v_feature = current_voice_x[i]
            emotion_label = current_voice_y[i]
            
            # Find a random Face Image matching the voice emotion
            if emotion_label in self.face_paths and len(self.face_paths[emotion_label]) > 0:
                random_face_path = np.random.choice(self.face_paths[emotion_label])
                
                # Preprocess Face Image exactly how detect_emotion.py does
                img = cv2.imread(random_face_path, cv2.IMREAD_GRAYSCALE)
                
                # In case an image is broken
                if img is None:
                    continue 

                img = cv2.resize(img, (48, 48)) 
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1) # Becomes (48, 48, 1)
                
                batch_voice_x.append(v_feature)
                batch_face_x.append(img)
                
                emotion_idx = self.emotions.index(emotion_label)
                batch_y.append(emotion_idx)
                
        # Prevent Scikit-Learn scaler crashes on un-paired skipped batches
        if len(batch_voice_x) == 0:
            dummy_face = np.zeros((1, 48, 48, 1), dtype='float32')
            dummy_voice = np.zeros((1, 1024), dtype='float32')
            dummy_y = to_categorical([0], num_classes=len(self.emotions))
            return (dummy_face, dummy_voice), dummy_y

        # Scale the Voice features
        batch_voice_x = voice_scaler.transform(np.array(batch_voice_x))
        # Input shape expected by HuBERT model is (1024,). No extra dimension needed for Keras.
        
        batch_face_x = np.array(batch_face_x)
        # Expected input shape Face: (48, 48, 1)
        
        batch_y_cat = to_categorical(batch_y, num_classes=len(self.emotions))
        
        return (batch_face_x, batch_voice_x), batch_y_cat

# ==========================================================
# 4. START TRAINING THE FUSION HEAD
# ==========================================================
FACE_TRAIN_DIR = r"d:\S6 Mini Project\Datasets\train" # Usually where FER2013 lands
VOICE_X_PATH = r"d:\S6 Mini Project\HuBERT Model\X_hubert_large.npy"
VOICE_Y_PATH = r"d:\S6 Mini Project\HuBERT Model\Y_hubert_large.npy"

if not os.path.exists(FACE_TRAIN_DIR) or not os.path.exists(VOICE_X_PATH):
    print(f"\n[ERROR] Ensure your Datasets are located exactly here:")
    print(f"FACE: {FACE_TRAIN_DIR}")
    print(f"VOICE: {VOICE_X_PATH}")
else:
    print("Building Data Generator...")
    train_gen = MultimodalGenerator(FACE_TRAIN_DIR, VOICE_X_PATH, VOICE_Y_PATH, batch_size=32)

    print("Starting Training of Attention Fusion Head...")
    history = attention_model.fit(
        train_gen,
        epochs=15, 
        verbose=1
    )

    attention_model.save(r"d:\S6 Mini Project\attention_fused_emotion_model.keras")
    print("Attention Fusion Model Saved Successfully!")
