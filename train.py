# train.py (no try/except, resnet18, train-only)
import os
import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # progress bar

# -------- CONFIG (edit if needed) --------
DATA_DIR = ""               # expects data/train/<class>/images
TRAIN_DIR = os.path.join(DATA_DIR, "train")
MODEL_DIR = "models"
NUM_EPOCHS = 8
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 0                 # set to 0 for compatibility; increase if you know what you're doing
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------- Sanity checks (fail fast, clear messages) --------
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory not found: '{TRAIN_DIR}'. Create it with structure: data/train/<class>/images")

# ensure there is at least one class folder with images
class_dirs = [d for d in Path(TRAIN_DIR).iterdir() if d.is_dir()]
if not class_dirs:
    raise FileNotFoundError(f"No class subdirectories found under '{TRAIN_DIR}'. Each class should be a folder containing images.")

# -------- TRANSFORMS --------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- DATASETS & LOADERS --------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
if len(train_dataset) == 0:
    raise ValueError(f"No images found in ImageFolder at '{TRAIN_DIR}'. Check file extensions (.jpg/.png) and that files exist.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Save class index -> label map
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(idx_to_class, f)

print("Classes:", class_to_idx)
print("Device:", DEVICE)
print(f"Train samples: {len(train_dataset)}")

# -------- MODEL (resnet18 - stable attributes) --------
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
num_classes = len(class_to_idx)
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# Optional: freeze backbone for initial head-only training (uncomment to use)
# for param in model.layer1.parameters():
#     param.requires_grad = False
# for param in model.layer2.parameters():
#     param.requires_grad = False
# for param in model.layer3.parameters():
#     param.requires_grad = False
# for param in model.layer4.parameters():
#     param.requires_grad = False

# -------- LOSS & OPTIMIZER --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

# -------- TRAIN LOOP (no validation) --------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    since = time.time()

    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Train"):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total if total else 0.0
    epoch_acc = running_corrects / total if total else 0.0
    print(f" Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    scheduler.step()

    # Save best by training accuracy
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            "model_state_dict": best_model_wts,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_acc": best_acc
        }, os.path.join(MODEL_DIR, "best_model.pth"))
        print(" Saved best model (based on train acc).")

    elapsed = time.time() - since
    print(f" Epoch time: {elapsed:.1f}s")

print(f"\nTraining complete. Best train acc: {best_acc:.4f}")
# Save final model weights
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pth"))
print("Saved final_model.pth")
