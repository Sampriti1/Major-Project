import os
from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights 


app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from loadData import get_dataloaders
train_dataset, _, _, _, _, _ = get_dataloaders(batch_size=1)
class_names = train_dataset.classes



model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  
model.fc = nn.Linear(model.fc.in_features, len(class_names))


model.load_state_dict(torch.load("models/tea_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            result = "No file part"
            return render_template("index.html", result=result)

        file = request.files["file"]
        if file.filename == "":
            result = "No selected file"
            return render_template("index.html", result=result)

        if file:
   
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            
            image = Image.open(filepath).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0).to(device)

           
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                result = class_names[predicted.item()]

            
            os.remove(filepath)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

