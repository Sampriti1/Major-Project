import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights  
from loadData import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders(batch_size=1)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print("Classes:", class_names)


model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)


model.load_state_dict(torch.load("models/tea_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()


images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print(f"True label: {class_names[labels.item()]}")
print(f"Predicted: {class_names[predicted.item()]}")

