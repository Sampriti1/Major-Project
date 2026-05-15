import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from loadData import get_dataloaders  # This already contains your AlbuDataset logic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataloaders (batch_size=1 is fine for a single prediction check)
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders(batch_size=1)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print("Classes:", class_names)

# --- CHANGE 1: MATCH THE ARCHITECTURE ---
# You were using ResNet18 here, but your train.py uses EfficientNet-B2
model = efficientnet_b2(weights=None) 
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# --- CHANGE 2: LOAD WEIGHTS ---
model.load_state_dict(torch.load("models/tea_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- CHANGE 3: INFERENCE ---
# Since get_dataloaders already uses AlbuDataset, the images coming out of 
# test_loader are already correctly augmented/normalized.
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad(): # Good practice for prediction
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

print(f"True label: {class_names[labels.item()]}")
print(f"Predicted: {class_names[predicted.item()]}")