from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import pandas as pd
from typing import Dict, List

app = FastAPI(
    title="Snake Species Classifier API",
    description="API for classifying snake species from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to store model and species data
model = None
species_db = None
valid_class_ids = None

def load_model():
    """Load the trained model"""
    return tf.keras.models.load_model('models/model.keras')

def load_species_info():
    """Load the species information database"""
    return pd.read_csv('data/snake_species_info.csv')

def get_valid_class_ids(species_db):
    """Get sorted list of valid class IDs from the database"""
    return sorted(species_db['class_id'].unique())

def find_nearest_class_id(predicted_class, valid_class_ids):
    """Find the nearest valid class ID to the predicted class"""
    valid_class_ids = np.array(valid_class_ids)
    idx = (np.abs(valid_class_ids - predicted_class)).argmin()
    return valid_class_ids[idx]

def preprocess_image(image_bytes):
    """Preprocess the image bytes for model prediction"""
    # Convert bytes to numpy array
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = np.array(image)
    
    # Resize to match training dimensions
    image = cv2.resize(image, (480, 480))
    # Convert to float and normalize
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.on_event("startup")
async def startup_event():
    """Load model and species data on startup"""
    global model, species_db, valid_class_ids
    try:
        model = load_model()
        species_db = load_species_info()
        valid_class_ids = get_valid_class_ids(species_db)
    except Exception as e:
        print(f"Error loading model or species data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load model or species data")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Snake Species Classifier API",
        "status": "active",
        "endpoints": [
            "/predict",
            "/species"
        ]
    }

@app.get("/species")
async def get_species_list() -> Dict[str, List[Dict]]:
    """Get list of all species in the database"""
    species_list = species_db.to_dict('records')
    return {"species": species_list}

@app.post("/predict")
async def predict_species(file: UploadFile = File(...)):
    """Predict snake species from uploaded image"""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = preprocess_image(contents)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        
        # Map to nearest valid class ID
        mapped_class = find_nearest_class_id(predicted_class, valid_class_ids)
        confidence = float(prediction[0][predicted_class])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        alternatives = []
        
        for idx in top_3_indices:
            mapped_idx = find_nearest_class_id(idx, valid_class_ids)
            species_info = species_db[species_db['class_id'] == mapped_idx].iloc[0]
            
            alternatives.append({
                "class_id": int(mapped_idx),
                "binomial": species_info['binomial'],
                "confidence": float(prediction[0][idx]),
                "poisonous": bool(species_info['poisonous']),
                "mapped_from": int(idx) if mapped_idx != idx else None
            })
        
        # Get primary prediction details
        species_info = species_db[species_db['class_id'] == mapped_class].iloc[0]
        
        return {
            "prediction": {
                "class_id": int(mapped_class),
                "binomial": species_info['binomial'],
                "confidence": confidence,
                "genus": species_info['genus'],
                "family": species_info['family'],
                "snake_sub_family": species_info['snake_sub_family'],
                "poisonous": bool(species_info['poisonous']),
                "primary_continent": species_info['primary_continent'],
                "primary_country": species_info['primary_country'],
                "mapped_from": int(predicted_class) if mapped_class != predicted_class else None
            },
            "alternatives": alternatives
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
