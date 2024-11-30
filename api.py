from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
import io
import pandas as pd

app = FastAPI(title="Snake Species Classifier API")

# Constants
IMAGE_DIM = 480
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load species info
species_info = pd.read_csv('data/snake_species_info.csv')
class_mapping = {idx: row for idx, row in species_info.iterrows()}

# Initialize model
def load_model():
    model = resnet50()
    num_classes = len(species_info)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the trained weights
    checkpoint = torch.load('models/model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and transform image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top 3 predictions
            top_probs, top_classes = torch.topk(probabilities, 3)
            
            results = []
            for prob, class_idx in zip(top_probs[0], top_classes[0]):
                snake_info = class_mapping[int(class_idx)]
                results.append({
                    "species": snake_info['binomial'],
                    "family": snake_info['family'],
                    "genus": snake_info['genus'],
                    "poisonous": bool(snake_info['poisonous']),
                    "primary_continent": snake_info['primary_continent'],
                    "primary_country": snake_info['primary_country'],
                    "confidence": float(prob) * 100
                })
        
        return JSONResponse({
            "status": "success",
            "predictions": results
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "Snake Species Classification API is running. Use /predict endpoint with an image file to classify snakes."
    }
