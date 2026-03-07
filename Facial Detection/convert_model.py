from tensorflow.keras.models import load_model

# Load existing .h5 model
model = load_model("emotion_model.h5")

# Save as new .keras format
model.save("emotion_model.keras")

print("Model converted successfully to emotion_model.keras")
