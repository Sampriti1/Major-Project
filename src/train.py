import torch
import torch.nn as nn
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


model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
for epoch in range(epochs):
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


torch.save(model.state_dict(), "models/tea_disease_model.pth")
print("Model training completed and saved!")

