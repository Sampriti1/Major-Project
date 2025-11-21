import os
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights 
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
import cv2
import numpy as np
from loadData import get_dataloaders

# ---------------- Flask setup ----------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Device setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Load class names ----------------
train_dataset, _, _, _, _, _ = get_dataloaders(batch_size=1)
class_names = train_dataset.classes
print("Classes:", class_names)

# ---------------- Load model ----------------
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/tea_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------- Transform (match training) ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize camera frames to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------- Web routes ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    """Main upload page."""
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result="No selected file")

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                result = f"{class_names[predicted.item()]} ({confidence.item()*100:.2f}%)"

            os.remove(filepath)

    return render_template("index.html", result=result)

# ---------------- Camera stream ----------------
def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert BGR -> RGB and then PIL Image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((224, 224))  # Resize to match model input

        # Apply transform
        image = transform(img).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            label = class_names[predicted.item()]

        # Overlay prediction and confidence
        text = f"{label} ({confidence.item()*100:.1f}%)"
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/camera')
def camera():
    """Live camera feed."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- Run the app ----------------
if __name__ == "__main__":
    app.run(debug=True)
