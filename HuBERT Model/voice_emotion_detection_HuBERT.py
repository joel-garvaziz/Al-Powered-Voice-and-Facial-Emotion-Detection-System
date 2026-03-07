# ============================================================
# ENV
# ============================================================

import os
import warnings
from transformers.utils import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from transformers import HubertModel, Wav2Vec2Processor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization

# ============================================================
# LOAD HUBERT LARGE
# ============================================================

MODEL_NAME = "facebook/hubert-large-ls960-ft"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
hubert_model = HubertModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    use_safetensors=True
)
hubert_model.to(device)
hubert_model.eval()


# ============================================================
# DATA PATHS (UNCHANGED)
# ============================================================

Ravdess = r"D:\S6 Mini Project\Datasets\audio_speech_actors_01-24"
Crema   = r"D:\S6 Mini Project\Datasets\AudioWAV"
Tess    = r"D:\S6 Mini Project\Datasets\TESS Toronto emotional speech set data"
Savee   = r"D:\S6 Mini Project\Datasets\ALL"

# ============================================================
# ADDITIONAL DATASETS
# ============================================================

IEMOCAP = r"D:\S6 Mini Project\Datasets\IEMOCAP"
EmotionSpeech = r"D:\S6 Mini Project\Datasets\Emotion Speech Dataset"
MLEND = r"D:\S6 Mini Project\Datasets\MLEnd\MLEndSND_Public"
SYNTH = r"D:\S6 Mini Project\Datasets\synth_speech"


# ============================================================
# BUILD DATAFRAME (UNCHANGED)
# ============================================================

file_emotion = []
file_path = []

for d in os.listdir(Ravdess):
    for f in os.listdir(os.path.join(Ravdess, d)):
        emo = int(f.split("-")[2])
        file_emotion.append(emo)
        file_path.append(os.path.join(Ravdess, d, f))

rav = pd.DataFrame({"Emotions": file_emotion, "Path": file_path})
rav.Emotions.replace({
    1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fear',7:'disgust',8:'surprise'
}, inplace=True)

file_emotion, file_path = [], []
for f in os.listdir(Crema):
    file_path.append(os.path.join(Crema,f))
    emo = f.split("_")[2]
    file_emotion.append({
        "SAD":"sad","ANG":"angry","DIS":"disgust","FEA":"fear","HAP":"happy","NEU":"neutral"
    }.get(emo,"neutral"))

crema = pd.DataFrame({"Emotions":file_emotion,"Path":file_path})

file_emotion, file_path = [], []
for d in os.listdir(Tess):
    for f in os.listdir(os.path.join(Tess,d)):
        e = f.split("_")[2].split(".")[0]
        file_emotion.append("surprise" if e=="ps" else e)
        file_path.append(os.path.join(Tess,d,f))

tess = pd.DataFrame({"Emotions":file_emotion,"Path":file_path})

file_emotion, file_path = [], []
for f in os.listdir(Savee):
    ele = f.split("_")[1][:-6]
    file_emotion.append({
        "a":"angry","d":"disgust","f":"fear","h":"happy","n":"neutral","sa":"sad"
    }.get(ele,"surprise"))
    file_path.append(os.path.join(Savee,f))

savee = pd.DataFrame({"Emotions":file_emotion,"Path":file_path})

file_emotion, file_path = [], []

for emotion in os.listdir(IEMOCAP):
    emo_path = os.path.join(IEMOCAP, emotion)

    for f in os.listdir(emo_path):

        if not f.endswith(".wav"):
            continue
        file_path.append(os.path.join(emo_path,f))

        file_emotion.append({
            "Angry":"angry",
            "Disgust":"disgust",
            "Excited":"happy",
            "Fearful":"fear",
            "Frustration":"angry",
            "Happy":"happy",
            "Neutral":"neutral",
            "Sad":"sad",
            "Surprised":"surprise"
        }.get(emotion,"neutral"))

iemocap = pd.DataFrame({
    "Emotions": file_emotion,
    "Path": file_path
})

file_emotion, file_path = [], []

for speaker in os.listdir(EmotionSpeech):

    speaker_path = os.path.join(EmotionSpeech, speaker)

    if not os.path.isdir(speaker_path):
        continue

    for emotion in os.listdir(speaker_path):

        emo_path = os.path.join(speaker_path, emotion)

        if not os.path.isdir(emo_path):
            continue

        for f in os.listdir(emo_path):

            if not f.endswith(".wav"):
                continue

        file_path.append(os.path.join(emo_path,f))
        file_emotion.append(emotion.lower())

emotion_speech = pd.DataFrame({"Emotions":file_emotion,"Path":file_path})

mlend_attr = pd.read_csv(r"D:\S6 Mini Project\Datasets\MLEnd\MLEndSND_Public\MLEndSND_Audio_Attributes.csv")

emotion_map = {
    "excited":"happy",
    "neutral":"neutral",
    "bored":"sad",
    "question":"surprise"
}

file_emotion = []
file_path = []

for i,row in mlend_attr.iterrows():

    filename = str(row["Public filename"]).zfill(5)+".wav"
    emotion = emotion_map.get(row["Intonation"],"neutral")

    file_path.append(os.path.join(MLEND,filename))
    file_emotion.append(emotion)

mlend = pd.DataFrame({"Emotions":file_emotion,"Path":file_path})

file_emotion = []
file_path = []

for speaker in os.listdir(SYNTH):  # F1,F2,...M12

    speaker_path = os.path.join(SYNTH, speaker)

    if not os.path.isdir(speaker_path):
        continue

    for sentence in os.listdir(speaker_path):  # s1,s2,...s8

        sentence_path = os.path.join(speaker_path, sentence)

        if not os.path.isdir(sentence_path):
            continue

        for f in os.listdir(sentence_path):

            path = os.path.join(sentence_path, f)
            file_path.append(path)

            emotion = f.split("-")[-1].split(".")[0]

            emotion = {
                "anger": "angry",
                "calm": "neutral",
                "disgust": "disgust",
                "fear": "fear",
                "joy": "happy",
                "low": "sad",
                "neutral": "neutral",
                "sadness": "sad",
                "surprise": "surprise"
            }.get(emotion, "neutral")

            file_emotion.append(emotion)

synth = pd.DataFrame({
    "Emotions": file_emotion,
    "Path": file_path
})

# print("rav:", type(rav))
# print("crema:", type(crema))
# print("tess:", type(tess))
# print("savee:", type(savee))
# print("IEMOCAP:", type(IEMOCAP))
# print("emotion_speech:", type(emotion_speech))
# print("MLEND:", type(MLEND))
# print("synth:", type(synth))

data_path = pd.concat([
    rav,
    crema,
    tess,
    savee,
    iemocap,
    emotion_speech,
    mlend,
    synth
], axis=0)
print("Total files:",len(data_path))

data_path = data_path.sample(frac=1).reset_index(drop=True)
# ============================================================
# HUBERT FEATURE EXTRACTION
# ============================================================

def get_hubert_features(path):
    speech, sr = librosa.load(path, sr=16000)
    speech = librosa.util.normalize(speech)

    inputs = processor(speech,
                       sampling_rate=16000,
                       return_tensors="pt",
                       padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hubert_model(**inputs)

    hidden_states = outputs.hidden_states

    # Mean of last 4 layers
    stacked = torch.stack(hidden_states[-4:])
    layer_mean = stacked.mean(dim=0)

    # Global mean pooling (time axis)
    pooled = layer_mean.mean(dim=1)

    return pooled.squeeze().cpu().numpy()
# ============================================================
# BUILD FEATURE SET
# ============================================================

X,Y = [],[]

for i,(p,e) in enumerate(zip(data_path.Path,data_path.Emotions)):
    if i%50==0:
        print(i,"/",len(data_path))
    try:
        X.append(get_hubert_features(p))
        Y.append(e)
    except Exception as err:
        print("Skipping file:", p)

X = np.array(X)
Y = np.array(Y)

np.save("X_hubert_large.npy", X)
np.save("Y_hubert_large.npy", Y)

print("HuBERT features saved")


# ============================================================
# ENCODING + SPLIT
# ============================================================

encoder = OneHotEncoder()
Y = encoder.fit_transform(Y.reshape(-1,1)).toarray()

x_train,x_test,y_train,y_test = train_test_split(
    X,Y,
    test_size=0.2,
    stratify=Y,
    random_state=0
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

joblib.dump(scaler,"scaler_hubert_large.pkl")
joblib.dump(encoder,"encoder_hubert_large.pkl")


# ============================================================
# MODEL (INPUT = 1024 FOR HUBERT LARGE)
# ============================================================

model = Sequential([
    Dense(512, activation='relu', input_shape=(1024,)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ============================================================
# TRAIN
# ============================================================

rlrp = ReduceLROnPlateau(monitor='loss',factor=0.4,patience=2,min_lr=1e-7)

history = model.fit(
    x_train,y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test,y_test),
    callbacks=[rlrp]
)

# ============================================================
# TRAINING RESULTS
# ============================================================

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print("\n=========== TRAINING RESULTS ===========")
print(f"Training Accuracy     : {train_acc*100:.2f}%")
print(f"Validation Accuracy   : {val_acc*100:.2f}%")
print(f"Training Loss         : {train_loss:.4f}")
print(f"Validation Loss       : {val_loss:.4f}")
print("========================================")

# ============================================================
# SAVE
# ============================================================

model.save("voice_emotion_detection_hubert_large.keras")
print("Model saved")

# ============================================================
# EVALUATION
# ============================================================

test_loss, test_accuracy = model.evaluate(x_test, y_test)

pred = model.predict(x_test)

y_pred = encoder.inverse_transform(pred)
y_true = encoder.inverse_transform(y_test)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')

print("\n=========== MODEL PERFORMANCE ===========")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print("=========================================")

print("\nClassification Report\n")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true,y_pred)
sns.heatmap(cm,annot=True,fmt="d",
            xticklabels=encoder.categories_[0],
            yticklabels=encoder.categories_[0])
plt.show()
