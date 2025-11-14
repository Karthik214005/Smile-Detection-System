# app_gradio.py
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# -------- CONFIG - edit if needed --------
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pth")
CLASS_JSON = os.path.join(MODEL_DIR, "class_indices.json")
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Sanity checks (fail fast) --------
if not os.path.exists(CLASS_JSON):
    raise FileNotFoundError(f"Missing class mapping: '{CLASS_JSON}'. Train script should create this file.")

# Prefer checkpoint -> fallback to final state_dict
if not os.path.exists(BEST_MODEL_PATH) and not os.path.exists(FINAL_MODEL_PATH):
    raise FileNotFoundError(f"No model file found. Expected one of: '{BEST_MODEL_PATH}' or '{FINAL_MODEL_PATH}'")

# -------- Load class mapping --------
with open(CLASS_JSON, "r") as f:
    idx_to_class = json.load(f)
# convert keys to int if they are strings
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

num_classes = len(idx_to_class)

# -------- Build model architecture (must match training) --------
model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)

# -------- Load weights --------
if os.path.exists(BEST_MODEL_PATH):
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    # checkpoint might be a dict with 'model_state_dict' or might be a bare state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
elif os.path.exists(FINAL_MODEL_PATH):
    state_dict = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)
else:
    # already handled above, but keep explicit
    raise FileNotFoundError("No model file found to load.")

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# -------- Preprocessing pipeline (same as training) --------
preprocess = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- Predict function for Gradio --------
def predict(image: Image.Image):
    if image is None:
        return "No image", {}
    if image.mode != "RGB":
        image = image.convert("RGB")
    inp = preprocess(image).unsqueeze(0).to(DEVICE)  # shape: (1,3,H,W)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    top_label = idx_to_class[top_idx]
    top_prob = float(probs[top_idx])
    # build probability dict with class names
    prob_dict = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    return f"{top_label} ({top_prob*100:.2f}%)", prob_dict

# -------- Gradio UI --------
title = "Two-label classifier (ResNet18)"
description = "Upload an image and the model will predict the class and show class probabilities."

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Prediction"), gr.Label(num_top_classes=2, label="Probabilities")],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share = True)
