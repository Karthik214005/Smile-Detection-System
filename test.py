# test_predict.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "models/final_model.pth"
IMG_SIZE = (160, 160)  # same as used during training
TEST_IMAGES = [
    "D:\\Mini_karthik\\test\\dogs\\dog (1).jpg",   # path to known cat
    "D:\\Mini_karthik\\test\\cats\\cat (3).jpg"    # path to known dog
]
# ----------------------------------------

# Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

def preprocess_image(path):
    """Load and preprocess an image for prediction"""
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(path):
    """Predict a single image and print result"""
    x = preprocess_image(path)
    pred = model.predict(x)[0][0]
    label = "Dog" if pred > 0.5 else "Cat"
    print(f"Image: {path}")
    print(f"  Sigmoid output: {pred:.4f}")
    print(f"  Predicted Label: {label}")
    print("-" * 40)

# Run predictions
for path in TEST_IMAGES:
    if os.path.exists(path):
        predict_image(path)
    else:
        print(f"[!] Image not found: {path}")
