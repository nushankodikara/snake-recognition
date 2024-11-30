import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon="🐍",
    layout="wide"
)

# Constants
IMAGE_DIM = 480
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    """Load the trained PyTorch model"""
    model = resnet50()
    num_classes = len(load_species_info())
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load('models/model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_data
def load_species_info():
    """Load the species information database"""
    return pd.read_csv('data/snake_species_info.csv')

def display_species_info(species_info, confidence):
    """Display detailed information about the predicted species"""
    if species_info is None:
        st.error("Unable to find species information for this prediction.")
        return
        
    st.write(f"**Scientific Name:** {species_info['binomial']}")
    st.write(f"**Confidence:** {confidence:.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Taxonomic Information**")
        st.write(f"- Family: {species_info['family']}")
        st.write(f"- Genus: {species_info['genus']}")
        st.write(f"- Subfamily: {species_info['snake_sub_family']}")
        
    with col2:
        st.write("**Distribution**")
        st.write(f"- Continent: {species_info['primary_continent']}")
        st.write(f"- Country: {species_info['primary_country']}")
    
    if species_info['poisonous']:
        st.warning("⚠️ **VENOMOUS SPECIES** - This snake is venomous and should be considered dangerous!", icon="⚠️")
    
    st.write("**Dataset Statistics**")
    st.write(f"- Training samples: {species_info['train_samples']}")
    st.write(f"- Test samples: {species_info['test_samples']}")
    st.write(f"- Total samples: {species_info['total_samples']}")

def get_prediction(image, model):
    """Get model prediction for an image"""
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities[0]

def display_predictions(predictions, species_db):
    """Display predictions with proper DataFrame indexing"""
    top_3_probs, top_3_indices = torch.topk(predictions, 3)
    
    # Convert tensor indices to Python integers for DataFrame indexing
    indices = [idx.item() for idx in top_3_indices]
    probs = [prob.item() for prob in top_3_probs]
    
    # Display top prediction
    top_species = species_db.iloc[indices[0]]
    display_species_info(top_species, probs[0])

    # Display alternative predictions
    st.write("**Alternative Predictions:**")
    for prob, idx in zip(probs[1:], indices[1:]):
        species = species_db.iloc[idx]
        st.write(f"- {species['binomial']}: {prob:.2%}")

def main():
    st.title("🐍 Snake Species Classifier")
    st.write("Upload an image of a snake to identify its species!")

    try:
        model = load_model()
        species_db = load_species_info()
        st.success("Model and species database loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model or species database: {str(e)}")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)

        try:
            with st.spinner('Analyzing image...'):
                predictions = get_prediction(image, model)

            with col2:
                st.subheader("Prediction Results")
                display_predictions(predictions, species_db)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please ensure the image is valid and try again.")

    with st.expander("About this model"):
        st.write("""
        This model was trained to identify different snake species using deep learning. 
        The model uses ResNet50 architecture and was trained on a diverse dataset of snake images.
        
        **Important Notes:**
        - This is an experimental model and should not be used as the sole method for identifying snakes
        - Always maintain a safe distance from snakes and contact professional snake handlers if needed
        - Some species may look similar, so consider the confidence score and alternative predictions
        - The model's performance may vary based on image quality and angle
        """)

        st.write("\n**Dataset Overview:**")
        total_species = len(species_db)
        venomous_count = len(species_db[species_db['poisonous'] == 1])
        
        st.write(f"- Total number of species: {total_species}")
        st.write(f"- Venomous species: {venomous_count}")
        st.write(f"- Non-venomous species: {total_species - venomous_count}")

if __name__ == "__main__":
    main()
