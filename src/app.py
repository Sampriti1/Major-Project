import os
import json
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from rembg import remove

app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load Class Names ---
try:
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)
    print("Loaded class names:", class_names)
except FileNotFoundError:
    print("Warning: class_names.json not found! Using default list.")
    class_names = [
        'Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 
        'gray light', 'healthy', 'red leaf spot', 'white spot'
    ]

# --- 2. Initialize Model (EfficientNet-B2) ---
# Weights=None because we are loading our custom trained weights
model = efficientnet_b2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Load your trained weights
model.load_state_dict(torch.load("models/tea_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- 3. Synchronized Inference Transform ---
# Matches val_test_transform in loadData.py exactly
inference_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_gradcam(model, image_tensor, image_pil):
    """Generates Grad-CAM heatmap using the last convolutional layer of EfficientNet."""
    # EfficientNet target layer is different from ResNet
    target_layer = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    
    grayscale_cam = cam(input_tensor=image_tensor)[0]
    
    # Prepare original image for overlay
    rgb_img = np.array(image_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    pil_vis = Image.fromarray(visualization)
    buffer = BytesIO()
    pil_vis.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def get_curing_suggestions(disease_name):
    """Full mapping for all tea disease classes."""
    suggestions_map = {
        "Anthracnose": {
            "practical": [
                {"icon": "fa-fan", "text": "Improve **air circulation** around the plants."},
                {"icon": "fa-scissors", "text": "Prune and destroy infected branches and leaves."},
                {"icon": "fa-spray-can", "text": "Apply copper-based fungicides regularly."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "**Nitrogen:** Apply urea to boost leaf growth."},
                {"icon": "fa-leaf", "text": "**Potassium:** Use potassium sulfate for resistance."}
            ]
        },
        "algal leaf": {
            "practical": [
                {"icon": "fa-pump-medical", "text": "Apply liquid copper fungicide to affected areas."},
                {"icon": "fa-sun", "text": "Increase sunlight and air exposure to dry the leaves."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "**Magnesium:** Use magnesium sulfate foliar spray if chlorosis occurs."}
            ]
        },
        "bird eye spot": {
            "practical": [
                {"icon": "fa-sun", "text": "Ensure adequate sunlight; disease thrives in shade."},
                {"icon": "fa-leaf", "text": "Remove and destroy heavily infected leaves."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "Apply balanced NPK fertilizer to improve resistance."}
            ]
        },
        "brown blight": {
            "practical": [
                {"icon": "fa-leaf", "text": "Collect and destroy all infected leaves regularly."},
                {"icon": "fa-wind", "text": "Reduce humidity by improving air circulation."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "Ensure adequate potassium supply to strengthen tissues."}
            ]
        },
        "gray light": {
            "practical": [
                {"icon": "fa-hand-scissors", "text": "Prune and destroy severely affected parts immediately."},
                {"icon": "fa-spray-can", "text": "Apply systemic fungicide at early stages."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "Maintain balanced nitrogen; avoid excessive soft growth."}
            ]
        },
        "red leaf spot": {
            "practical": [
                {"icon": "fa-tint", "text": "Ensure proper drainage; avoid water stagnation near roots."},
                {"icon": "fa-scissors", "text": "Remove and dispose of fallen infected leaves."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "Supplement potassium to improve leaf strength."}
            ]
        },
        "white spot": {
            "practical": [
                {"icon": "fa-shield-halved", "text": "Apply sulfur-based fungicides to control spread."},
                {"icon": "fa-brush", "text": "Clean mildly infected leaves using a damp cloth."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "Apply balanced micronutrients to support healthy development."}
            ]
        },
        "healthy": {
            "practical": [
                {"icon": "fa-check-circle", "text": "The tea leaf is healthy. Continue good practices."},
                {"icon": "fa-seedling", "text": "Regular monitoring helps early detection of issues."}
            ],
            "fertilizer_nutrients": [
                {"icon": "fa-leaf", "text": "Follow a balanced NPK schedule for optimal growth."}
            ]
        }
    }
    return suggestions_map.get(disease_name, {"practical": [], "fertilizer_nutrients": []})

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/detect", methods=["GET"])
def detect_page():
    return render_template("index.html")

@app.route("/api/detect", methods=["POST"])
def detect_disease():
    if "file" not in request.files:
        return jsonify({"error": "No file part received"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # 1. Save File Temporarily
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # 2. Background Removal (Matching User Strategy)
        with open(filepath, "rb") as f:
            input_data = f.read()
        output_data = remove(input_data)
        
        # 3. Prepare Image
        image_pil = Image.open(BytesIO(output_data)).convert("RGB")
        image_np = np.array(image_pil)
        
        # 4. Apply Synchronized Preprocessing
        augmented = inference_transform(image=image_np)
        image_tensor = augmented["image"].unsqueeze(0).to(device)

        # 5. Model Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence_score = round(torch.max(probabilities).item() * 100)
            
            predicted_index = torch.argmax(probabilities).item()
            disease_name = class_names[predicted_index]

        # 6. Generate Grad-CAM for the processed leaf
        gradcam_base64 = get_gradcam(model, image_tensor, image_pil)
        suggestions = get_curing_suggestions(disease_name)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            "disease_name": disease_name,
            "sub_details": f"Probable Severity: {('High' if confidence_score > 70 else 'Medium')}",
            "confidence": confidence_score,
            "practical_suggestions": suggestions["practical"],
            "fertilizer_nutrient_suggestions": suggestions["fertilizer_nutrients"],
            "gradcam_image": gradcam_base64
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/api/detect_live", methods=["POST"])
def detect_live():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image received"}), 400

        # Decode base64 frame from webcam
        base64_str = data["image"].split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        
        # Apply Background removal to the live frame
        output_data = remove(img_bytes)
        image_pil = Image.open(BytesIO(output_data)).convert("RGB")

        # Preprocess
        image_np = np.array(image_pil)
        augmented = inference_transform(image=image_np)
        image_tensor = augmented["image"].unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence_score = round(torch.max(probabilities).item() * 100)
            predicted_index = torch.argmax(probabilities).item()
            disease_name = class_names[predicted_index]

        suggestions = get_curing_suggestions(disease_name)

        return jsonify({
            "disease_name": disease_name,
            "sub_details": f"Confidence: {confidence_score}%",
            "confidence": confidence_score,
            "suggestions": suggestions
        })

    except Exception as e:
        print("Live camera error:", e)
        return jsonify({"error": "Live detection failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)