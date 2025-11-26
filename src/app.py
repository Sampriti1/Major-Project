import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights 
import torch.nn.functional as F


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

def get_curing_suggestions(disease_name):
    """Returns a list of suggestion objects for a given disease name."""
    suggestions_map = {
        "Anthracnose": [
            {"icon": "fa-fan", "text": "Improve **air circulation** around the plants."},
            {"icon": "fa-scissors", "text": "Promptly **prune and destroy** infected branches and leaves."},
            {"icon": "fa-spray-can", "text": "Apply recommended **copper-based fungicides** regularly."},
            {"icon": "fa-trash-can", "text": "Clear plant debris from the base of the tea plant."}
        ],
        "algal leaf": [
            {"icon": "fa-pump-medical", "text": "Apply liquid copper fungicide to affected areas."},
            {"icon": "fa-water", "text": "Ensure proper drainage to reduce humidity."},
            {"icon": "fa-sun", "text": "Increase sunlight and air exposure to dry the leaves."}
        ],
        "bird eye spot": [
            {"icon": "fa-sun", "text": "Ensure **proper sunlight** exposure, as infection is common in shaded areas."},
            {"icon": "fa-leaf", "text": "Remove and destroy severely spotted leaves."},
            {"icon": "fa-spray-can", "text": "Apply suitable **fungicides** (e.g., copper oxychloride) in severe cases."},
            {"icon": "fa-tree", "text": "Promote overall plant vigor with balanced nutrition."}
        ],
        "brown blight": [
            {"icon": "fa-leaf", "text": "Remove and destroy **all infected leaves** to prevent spread."},
            {"icon": "fa-wind", "text": "Avoid overcrowding to improve air circulation and reduce moisture."},
            {"icon": "fa-fire-extinguisher", "text": "Use effective **chemical treatments** targeting fungal pathogens."},
        ],
        "gray light": [
            {"icon": "fa-hand-scissors", "text": "Prune severely affected parts and destroy them."},
            {"icon": "fa-temperature-low", "text": "Maintain proper **humidity and temperature** control, especially during storage."},
            {"icon": "fa-spray-can", "text": "Apply a fungicide early upon detection to halt progression."},
        ],
        "red leaf spot": [
            {"icon": "fa-tint", "text": "Ensure **good drainage** and avoid standing water near the roots."},
            {"icon": "fa-tree", "text": "Apply **urea** or other nitrogenous fertilizers carefully, as excess can worsen it."},
            {"icon": "fa-scissors", "text": "Remove and dispose of all fallen, infected leaves."},
        ],
        "white spot": [
            {"icon": "fa-shield-halved", "text": "Apply **sulfur-based fungicides** as a primary defense."},
            {"icon": "fa-brush", "text": "In non-severe cases, wiping the leaves with a damp cloth may help."},
            {"icon": "fa-clock", "text": "Ensure proper spacing and prune to maximize airflow."},
        ],
        "healthy": [
            {"icon": "fa-check-circle", "text": "The leaf appears **perfectly healthy**! Keep up the good agricultural practice."},
            {"icon": "fa-hand-holding-water", "text": "Maintain a balanced **irrigation and nutrient** schedule."},
        ]
    }
    
    # This ensures a clean fallback message if a class name is missed or misspelled
    return suggestions_map.get(disease_name, 
        [{"icon": "fa-search", "text": f"No specific treatment found for {disease_name}. Consult an agricultural expert."}])

@app.route("/")
def index():

    return render_template("index.html")

@app.route("/api/detect", methods=["POST"])
def detect_disease():
    if "file" not in request.files:
        return jsonify({"error": "No file part received"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # 1. Save and Load Image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            image = Image.open(filepath).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # 2. Prediction
            with torch.no_grad():
                output = model(image_tensor)
                
                probabilities = F.softmax(output, dim=1)[0]
                confidence_score = round(torch.max(probabilities).item() * 100)
                predicted_index = torch.argmax(probabilities).item()
                disease_name = class_names[predicted_index]
            suggestions = get_curing_suggestions(disease_name)
            os.remove(filepath)
            return jsonify({
                "disease_name": disease_name,
                "sub_details": f"Probable Severity: {('High' if confidence_score > 70 else 'Medium')}",
                "confidence": confidence_score,
                "suggestions": suggestions
            })    
        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({"error": "Failed during model prediction or file handling."}), 500

    return jsonify({"error": "An unknown error occurred"}), 500 
import base64
from io import BytesIO

@app.route("/api/detect_live", methods=["POST"])
def detect_live():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image received"}), 400

        # Remove base64 prefix
        base64_str = data["image"].split(",")[1]

        # Decode
        img_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence_score = round(torch.max(probabilities).item() * 100)
            predicted_index = torch.argmax(probabilities).item()
            disease_name = class_names[predicted_index]

        suggestions = get_curing_suggestions(disease_name)

        return jsonify({
            "disease_name": disease_name,
            "sub_details": f"Probable Severity: {('High' if confidence_score > 70 else 'Medium')}",
            "confidence": confidence_score,
            "suggestions": suggestions
        })

    except Exception as e:
        print("Live camera error:", e)
        return jsonify({"error": "Live detection failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
