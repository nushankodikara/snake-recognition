import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon="🐍",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    return tf.keras.models.load_model('models/model.keras')

@st.cache_data
def load_species_info():
    """Load the species information database"""
    return pd.read_csv('data/snake_species_info.csv')

def preprocess_image(image):
    """Preprocess the image to match model requirements"""
    # Resize to match training dimensions
    image = cv2.resize(image, (480, 480))
    # Convert to float and normalize
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def display_species_info(species_info, confidence):
    """Display detailed information about the predicted species"""
    if species_info is None:
        st.error("Unable to find species information for this prediction.")
        return
        
    st.write(f"**Scientific Name:** {species_info['binomial']}")
    st.write(f"**Confidence:** {confidence:.2%}")
    
    # Create two columns for species information
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
    
    # Warning for venomous species
    if species_info['poisonous']:
        st.warning("⚠️ **VENOMOUS SPECIES** - This snake is venomous and should be considered dangerous!", icon="⚠️")
    
    # Dataset statistics
    st.write("**Dataset Statistics**")
    st.write(f"- Training samples: {species_info['train_samples']}")
    st.write(f"- Test samples: {species_info['test_samples']}")
    st.write(f"- Total samples: {species_info['total_samples']}")

def get_species_info(species_db, class_id):
    """Safely get species information from database"""
    species_info = species_db[species_db['class_id'] == class_id]
    if len(species_info) == 0:
        return None
    return species_info.iloc[0]

def get_valid_class_ids(species_db):
    """Get sorted list of valid class IDs from the database"""
    return sorted(species_db['class_id'].unique())

def find_nearest_class_id(predicted_class, valid_class_ids):
    """Find the nearest valid class ID to the predicted class"""
    valid_class_ids = np.array(valid_class_ids)
    idx = (np.abs(valid_class_ids - predicted_class)).argmin()
    return valid_class_ids[idx]

def main():
    st.title("🐍 Snake Species Classifier")
    st.write("Upload an image of a snake to identify its species!")

    # Load model and species information
    try:
        model = load_model()
        species_db = load_species_info()
        valid_class_ids = get_valid_class_ids(species_db)
        st.success("Model and species database loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model or species database: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        # Preprocess image and make prediction
        try:
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            processed_image = preprocess_image(image_array)

            # Make prediction
            with st.spinner('Analyzing image...'):
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction[0])
                
                # Map to nearest valid class ID
                mapped_class = find_nearest_class_id(predicted_class, valid_class_ids)
                confidence = prediction[0][predicted_class]

            with col2:
                st.subheader("Prediction Results")
                
                if mapped_class != predicted_class:
                    st.info(f"Original prediction (class {predicted_class}) was mapped to nearest valid class ({mapped_class})")
                
                # Get species information
                species_info = get_species_info(species_db, mapped_class)
                display_species_info(species_info, confidence)

                # Display top 3 predictions
                st.write("**Alternative Predictions:**")
                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                for idx in top_3_indices[1:]:  # Skip the first one as it's already displayed
                    # Map each alternative prediction to valid class IDs
                    mapped_idx = find_nearest_class_id(idx, valid_class_ids)
                    alt_species = get_species_info(species_db, mapped_idx)
                    if alt_species is not None:
                        if mapped_idx != idx:
                            st.write(f"- {alt_species['binomial']}: {prediction[0][idx]:.2%} (mapped from class {idx})")
                        else:
                            st.write(f"- {alt_species['binomial']}: {prediction[0][idx]:.2%}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please ensure the image is valid and try again.")

    # Add information about the model
    with st.expander("About this model"):
        st.write("""
        This model was trained to identify different snake species using deep learning. 
        The model uses EfficientNetB4 architecture and was trained on a diverse dataset of snake images.
        
        **Important Notes:**
        - This is an experimental model and should not be used as the sole method for identifying snakes
        - Always maintain a safe distance from snakes and contact professional snake handlers if needed
        - Some species may look similar, so consider the confidence score and alternative predictions
        - The model's performance may vary based on image quality and angle
        """)

        # Display dataset statistics
        st.write("\n**Dataset Overview:**")
        total_species = len(species_db)
        venomous_count = len(species_db[species_db['poisonous'] == 1])
        
        st.write(f"- Total number of species: {total_species}")
        st.write(f"- Venomous species: {venomous_count}")
        st.write(f"- Non-venomous species: {total_species - venomous_count}")

if __name__ == "__main__":
    main()
