import torch
import torch.nn as nn
import json
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights 
from loadData import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
   param.requires_grad = False
   #param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
for param in model.fc.parameters():
    param.requires_grad = True 
model = model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)


epochs = 25
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
for epoch in range(epochs):
    if epoch==5:
       for param in model.parameters():
          param.requires_grad = True #unfreezing the brain after  5th epoch
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
    
    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")
    # ✅ After training on this epoch, evaluate on validation set
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
     for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

    val_acc = correct_val / total_val * 100
    print(f"Validation Accuracy: {val_acc:.2f}%")


torch.save(model.state_dict(), "models/tea_disease_model.pth")
with open("models/class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)
print("Model training completed and saved!")

