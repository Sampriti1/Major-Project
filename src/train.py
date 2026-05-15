import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import json
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from loadData import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print("Classes:", class_names)


model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

model = model.to(device)

# Unfreeze only the classifier head for initial training
for param in model.classifier.parameters():   # FIXED: was model.fc
    param.requires_grad = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

epochs = 25
best_val_f1 = 0.0

print("\n--- STARTING TRAINING ---")

for epoch in range(epochs):
    start_time = time.time()

    if epoch == 5:
        print("\n>>> UNFREEZING ALL LAYERS (Fine-Tuning Mode) <<<")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total * 100

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct_val / total_val * 100

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | F1: {f1:.2f}")

    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), "models/tea_disease_model.pth")
        print(f" NEW RECORD! Model saved with F1-Score: {best_val_f1:.2f}")

        with open("models/class_names.json", "w") as f:
            json.dump(train_dataset.classes, f)

print(f"\nTraining finished. Best F1 Score: {best_val_f1:.2f}")


print("\n--- GENERATING FINAL REPORT ON TEST SET ---")

model.load_state_dict(torch.load("models/tea_disease_model.pth"))
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=class_names))

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Tea Disease Detection')
plt.savefig('models/confusion_matrix.png')
print("Confusion Matrix saved to 'models/confusion_matrix.png'")